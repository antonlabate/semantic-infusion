import os

import torch
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
login(os.getenv("HF_KEY"))

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LANG = "galician"
ADAPTER_CHECKPOINT = "1439"

# Previously trained AMR adapter to export before the new training run
PREV_ADAPTER_PATH = f"results_{LANG}_mistral_7b_semantic_aware/checkpoint-{ADAPTER_CHECKPOINT}"
MERGED_MODEL_EXPORT_NAME = f"{LANG}-semantic-Mistral"

# Spider dataset (Galician + Guarani translations)
DATASET_PATH = "antonlabate/spider-glg-grn"


OUTPUT_DIR = f"./results_{LANG}_mistral_7b_ft_on_spider-1-epoch"
ADAPTER_NAME = f"mistral-7b-{LANG}-ft-on-spider-1-epoch"
TRAINING_LOG_PATH = f"training_history_{LANG}-ft-spider-1-epoch.xlsx"

MAX_SEQ_LENGTH = 1024

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
]

RESPONSE_TEMPLATE = "[/INST]"

SYSTEM_MESSAGE = (
    f"You are a great software engineer and SQL expert. You are given a question in {LANG}. "
    "Your task is, given the user question and schema of a database, generate ONLY the true SQL "
    "answer to the question, enclosed within <sql> and </sql> tags. "
    "However, before generating any code, make sure you deeply understand and capture the true "
    "meaning of this phrase. Once you have understood the intent of the sentence, output the SQL answer. "
    "You should pay attention to the following guidelines while generating your answer:\n"
    "  - Pay attention to the relationships between the terms in the question and to the meaning "
    "of the user request which they convey;\n"
    "  - Your answer must query only the columns that are needed to answer the question, with the "
    "appropriate operators. Be careful to not query for columns that do not exist in the schema;\n"
    "  - Pay attention to which column is in which table and use the proper columns for joins;\n"
    "  - Provide as an answer ONLY the SQL query that correctly answers the question, within the "
    "<sql> and </sql> tags without explanations or notes."
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
# Export previously trained AMR adapter
#
# Load the base model, attach the previously trained LoRA adapter, and save
# the merged weights before starting a new training run on the Spider dataset.
# ---------------------------------------------------------------------------

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    token=True,
)
peft_model = PeftModel.from_pretrained(base_model, PREV_ADAPTER_PATH)
peft_model.save_pretrained(MERGED_MODEL_EXPORT_NAME)
print(f"AMR adapter exported to '{MERGED_MODEL_EXPORT_NAME}'.")

del base_model, peft_model  # free GPU memory before loading the training model

# ---------------------------------------------------------------------------
# Model for training (the merged semantically-aware model, no quantization)
# ---------------------------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_EXPORT_NAME,
    device_map="auto",
    token=True,
)
model.config.use_cache = False

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model.is_parallelizable = True
    model.model_parallel = True

# ---------------------------------------------------------------------------
# Dataset (Spider, Galician + Guarani translations)
# ---------------------------------------------------------------------------

dataset = load_dataset(DATASET_PATH)
print(f"Train: {len(dataset["train"])} examples | Eval: {len(dataset["test"])} examples")

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_spider_sql_prompts(examples):
    """Format each Spider example as a chat-template prompt for SQL generation.

    The 'galician_prompt' field encodes the question and DB schema as:
        "{question}CREATE{schema_tables}|{foreign_key_relations}|..."

    The assistant is trained to output the correct SQL query wrapped in
    <sql>…</sql> tags. Only the assistant turn is used for loss computation
    (completion-only LM).
    """
    prompts = []
    for i in range(len(examples[f"{LANG}_prompt"])):
        raw_prompt = examples[f"{LANG}_prompt"][i]

        # Split question from schema (everything after the first "CREATE")
        question, schema_after_create = raw_prompt.split("CREATE", 1)

        # Extract optional foreign-key relations encoded after the first "|"
        if "|" in schema_after_create:
            schema_tables, fk_block = schema_after_create.split("|", 1)
            foreign_keys = fk_block.replace("|", ";")
        else:
            schema_tables = schema_after_create
            foreign_keys = ""

        answer = examples["output_sequence"][i].split("|", 1)[1]

        user_content = (
            f"Schema: CREATE {schema_tables}\n"
            f"Foreign keys relations: {foreign_keys}\n\n"
            f"Question: {question}\n"
        )
        assistant_content = f"<sql> {answer} </sql>"

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
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
    formatting_func=build_spider_sql_prompts,
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
