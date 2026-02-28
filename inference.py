import os
import time
from contextlib import nullcontext
from functools import partial

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
login(os.getenv("HF_KEY"))

LANG = "galician"
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_CHECKPOINT = "438"

# Model to load: the merged (base + AMR adapter) model, fine-tuned on Spider
MERGED_MODEL_PATH = f"{LANG}-semantic-Mistral"
ADAPTER_CHECKPOINT = f"results_{LANG}_mistral_7b_semantic_ft_on_spider-1-epoch/checkpoint-{ADAPTER_CHECKPOINT}"

EVAL_DATASET_PATH = (
    "training_data/spider_galician_and_guarani_translations_updated/"
    "resdsql_dev_Guarani_and_Galician.json"
)


OUTPUT_SQL_PATH = f"answer_files_sql/semantic-text2sql-{LANG}-mistral.sql"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 1024
INFERENCE_BATCH_SIZE = 32

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
# Model and tokenizer
# ---------------------------------------------------------------------------

base_model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    device_map="auto",
    token=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_CHECKPOINT)
model.eval()
model.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    max_length=MAX_SEQ_LENGTH,
    padding="max_length",
    padding_side="left",
    truncation=True,
)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset_eval = load_dataset("json", data_files=EVAL_DATASET_PATH, split="train")

# ---------------------------------------------------------------------------
# Prompt formatting (inference — no assistant turn)
# ---------------------------------------------------------------------------

def format_for_inference(example):
    """Build the chat-template prompt for a single Spider eval example.

    Parses the '{LANG}_prompt' field (format: "{question}CREATE{schema}|{fk_relations}|...")
    and produces a prompt ending with [/INST] so the model generates the SQL answer.
    """
    raw_prompt = example[f"{LANG}_prompt"]

    # Split question from schema block (everything after the first "CREATE")
    question, schema_after_create = raw_prompt.split("CREATE", 1)

    # Extract optional foreign-key relations encoded after the first "|"
    if "|" in schema_after_create:
        schema_tables, fk_block = schema_after_create.split("|", 1)
        foreign_keys = fk_block.replace("|", ";")
    else:
        schema_tables = schema_after_create
        foreign_keys = ""

    user_content = (
        f"Schema: CREATE {schema_tables}\n"
        f"Foreign keys relations: {foreign_keys}\n\n"
        f"Question: {question}\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # appends [/INST] so the model generates the answer
    )
    return {"messages": prompt}


dataset_eval = dataset_eval.map(format_for_inference, desc="Formatting dataset")

# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_sql_batch(examples, model):
    """Run batched inference and extract SQL answers from the model output.

    Returns a dict with key "sql_answer" for each example in the batch.
    Falls back to "invalid sql" when the model produces no <sql> content.
    """
    tokenized_inputs = tokenizer(
        examples["messages"],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    # Use autocast when the model runs in a reduced-precision dtype
    ctx = (
        torch.autocast(device_type=model.device.type, dtype=torch.bfloat16)
        if model.dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )

    with ctx:
        generation_output = model.generate(
            **tokenized_inputs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.unk_token_id,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

    raw_outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=False)

    sql_answers = []
    for raw in raw_outputs:
        # Trim at the end-of-sequence token, then extract content between <sql> tags
        raw = raw.split("</s>", 1)[0]
        sql = raw.split("<sql>")[-1].replace("</sql>", "").replace("</s>", "")
        sql_answers.append(sql if sql else "invalid sql")

    return {"sql_answer": sql_answers}


print("Starting batch inference...")
start_time = time.time()

sql_answers = dataset_eval.map(
    partial(generate_sql_batch, model=model),
    batched=True,
    batch_size=INFERENCE_BATCH_SIZE,
    desc="Generating SQL answers",
)

print(f"Inference completed in {time.time() - start_time:.2f} seconds.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

with open(OUTPUT_SQL_PATH, "w") as f:
    for row in sql_answers:
        f.write(row["sql_answer"] + "\n")

print("Results saved successfully.")
