import os

import torch
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
login(os.getenv("HF_KEY"))

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = "antonlabate/synthetic-simple-text-to-AMR-SQL-pt-glg-grn-cym"
LANG = "galician"

OUTPUT_DIR = f"./results_{LANG}_mistral_7b_semantic_aware"
ADAPTER_NAME = f"mistral-7b-{LANG}-semantic-aware"
TRAINING_LOG_PATH = f"training_history_{LANG}_semantic-aware.xlsx"

MAX_SEQ_LENGTH = 1024
TEST_SPLIT_SIZE = 0.05
RANDOM_SEED = 42

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
]

RESPONSE_TEMPLATE = "[/INST]"

SYSTEM_MESSAGE = (
    f"You are given a sentence in {LANG}. Your goal is to DEEPLY UNDERSTAND "
    "and CAPTURE the TRUE MEANING of this phrase, by identifying the intrinsic relationships "
    "between the terms it contains. FOCUS ON HOW THE ELEMENTS WITHIN THE SENTENCE INTERACT "
    "TO CONVEY MEANING. Also, BE ATTENTIVE TO NUANCES and CUES from the phrase that contribute "
    "to its interpretation. Once you have truly understood the true meaning of the phrase, output "
    "ONLY the semantic structure of the phrase enclosed within <semantics> and </semantics> tags. "
    "Do not include any explanations or additional comments."
)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    max_length=MAX_SEQ_LENGTH,
    padding="max_length",
    padding_side="left",
    truncation=True,
)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# ---------------------------------------------------------------------------
# Model (4-bit quantization via BitsAndBytes)
# ---------------------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=True,
)
model.config.use_cache = False

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model.is_parallelizable = True
    model.model_parallel = True

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset = load_dataset(DATASET_PATH, split="train").train_test_split(
    test_size=TEST_SPLIT_SIZE,
    seed=RANDOM_SEED,
)
print(dataset)

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_amr_prompts(examples):
    """Format each example as a chat-template prompt for AMR generation.

    The assistant is trained to output the AMR semantic structure of an input
    sentence (Galician or Guarani), wrapped in <semantics>…</semantics> tags.
    Only the assistant turn is used for loss computation (completion-only LM).
    """
    prompts = []
    for i in range(len(examples["id"])):
        question = examples[f"{LANG}_prompt"][i]
        amr_structure = examples["amr_phrase"][i]

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": f"<semantics> {amr_structure} </semantics>"},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompts.append(prompt)

    return prompts

# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    bf16=True,
    learning_rate=2e-5,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    max_seq_length=MAX_SEQ_LENGTH,
    logging_steps=0.1,
    save_strategy="epoch",
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=0.25,
    do_eval=True,
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    formatting_func=build_amr_prompts,
    args=training_args,
    data_collator=DataCollatorForCompletionOnlyLM(
        RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
    ),
)

# ---------------------------------------------------------------------------
# Train and save
# ---------------------------------------------------------------------------

trainer.train()

trainer.model.save_pretrained(ADAPTER_NAME)
print(f"LoRA adapter saved to '{ADAPTER_NAME}'.")

log_df = pd.DataFrame(trainer.state.log_history)
print(log_df)
log_df.to_excel(TRAINING_LOG_PATH)
print(f"Training history saved to '{TRAINING_LOG_PATH}'.")
