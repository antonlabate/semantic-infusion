# Semantic Infusion for Low-Resource Language Text-to-SQL

## Overview

**Semantic Infusion** is a three-step training pipeline that improves Text-to-SQL accuracy for **low-resource languages** (Galician and Guarani) by first teaching a large language model to understand the *semantic structure* of those languages before fine-tuning it to generate SQL.

The core hypothesis is that a model already familiar with the meaning of sentences in a low-resource language — their entities, relations, and intent — will produce more accurate SQL translations than one fine-tuned on SQL directly from a cold start. This familiarity is injected by training an Abstract Meaning Representation (AMR) parser as an intermediate step, then **merging** that semantic knowledge into the base model before the SQL fine-tuning begins.

The approach is evaluated on the **Spider** Text-to-SQL benchmark using Galician and Guarani translations of the dataset.

---

## The Three-Step Pipeline

```
┌─────────────────────────────────────────────────────────────────────────----┐
│  Step 1  │  train_general.py                                                │
│          │  Fine-tune a LoRA adapter that parses the AMR semantic           │
│          │  structure of Galician / Guarani input sentences.                │
│          │                                                                  │
│          │  Input:  sentence in Galician / Guarani                          │
│          │  Output: AMR tree  →  <semantics> (a / amr-tree ...) </semantics>│
└──────────────────────────────┬──────────────────────────────────────────----┘
                               │  merge AMR adapter into Mistral-7B
                               ▼
┌────────────────────────────────────────────────────────────────────────----─┐
│  Step 2  │  continued_training.py                                           │
│          │  1. Merge the Step 1 adapter → "{LANG}-semantic-Mistral"         │
│          │     (a Mistral-7B model now semantically aware of the            │
│          │      low-resource language)                                      │
│          │  2. Fine-tune a second LoRA adapter on top of this merged        │
│          │     model for Text-to-SQL using the translated Spider dataset.   │
│          │                                                                  │
│          │  Input:  Galician / Guarani question + DB schema                 │
│          │  Output: SQL query  →  <sql> SELECT ... </sql>                   │
└──────────────────────────────┬──────────────────────────────────────────----┘
                               │  load merged model + SQL adapter
                               ▼
┌────────────────────────────────────────────────────────────────────────----─┐
│  Step 3  │  inference.py                                                    │
│          │  Batch inference on the Spider dev set.                          │
│          │  Writes one predicted SQL query per line to a .sql file.         │
└─────────────────────────────────────────────────────────────────────────----┘
```

### Why AMR?

Abstract Meaning Representation (AMR) is a graph-based semantic formalism that encodes *who did what to whom* in a sentence, independent of surface phrasing. By first training the model to parse AMR graphs from low-resource language inputs, we give it a structured understanding of those languages. The subsequent SQL fine-tuning then builds on top of this richer internal representation rather than starting from a model that has little exposure to the target language.

---

## Hardware Requirements

Training requires a CUDA-capable GPU. The recommended minimum is:

| Step | Script | Min. VRAM | Notes |
|------|--------|-----------|-------|
| 1 — AMR fine-tuning    | `train_general.py`      | 24 GB | Mistral-7B in 4-bit (NF4) + LoRA |
| 2 — SQL fine-tuning    | `continued_training.py` | 40 GB | Mistral-7B in fp16 + LoRA (no quantization) |
| 3 — Inference          | `inference.py`          | 24 GB | fp16 model, no gradients |

Multi-GPU setups are detected and enabled automatically.

---

## Installation

```bash
git clone https://github.com/<your-org>/semantic-infusion.git
cd semantic-infusion
pip install -r requirements.txt
```

Then copy the environment template and add your Hugging Face access token
(needed to download `mistralai/Mistral-7B-Instruct-v0.3`):

```bash
cp .env.example .env
# Edit .env and set:  HF_KEY=hf_your_token_here
```

Get a token at <https://huggingface.co/settings/tokens>. Read access is sufficient.
You must also accept the [Mistral-7B-Instruct-v0.3 licence](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
on the model page before the download will succeed.

---

## Data

Two datasets are required. Place them under `data/` as shown below.

### 1. AMR Semantic Parsing Dataset — used in Step 1

A Hugging Face dataset saved to disk (`datasets.save_to_disk` format). Each
example must contain:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique example identifier |
| `galician_prompt` | string | Input sentence in Galician (or Guarani) |
| `amr_phrase` | string | Gold AMR tree string for that sentence |

Expected path (set via `DATASET_PATH` in `train_general.py`):

```
data/synthetic_text2sql_AMR_simple_SQL_glg_and_grn/
```

### 2. Spider Dataset — Galician & Guarani Translations — used in Steps 2 & 3

Two JSON files (train and dev splits). Each line-delimited JSON object must
contain:

| Field | Type | Description |
|-------|------|-------------|
| `galician_prompt` | string | `"{question}CREATE {schema_ddl}\|{foreign_keys}\|..."` |
| `output_sequence` | string | `"...\|{gold_sql}"` — the reference SQL query |

The `galician_prompt` field encodes both the natural-language question *and* the
database schema in a single string. The scripts split on `"CREATE"` to separate
them, and on `"|"` to extract foreign-key relations.

Expected paths (set via `TRAIN_DATASET_PATH` / `EVAL_DATASET_PATH`):

```
data/spider_galician_and_guarani_translations_updated/
    resdsql_train_Guarani_and_Galician.json
    resdsql_dev_Guarani_and_Galician.json
```

---

## Running the Pipeline

Every script has a clearly labelled `# Configuration` block at the top. Adjust
the constants there before running — **do not edit the rest of the file**.

---

### Step 1 — AMR Semantic Parsing Fine-Tuning

**Script:** `train_general.py`

Trains a LoRA adapter that maps an input sentence (Galician or Guarani) to its
AMR semantic tree. Only the assistant turn (the `<semantics>` output) contributes
to the training loss (completion-only LM).

**Configuration:**

| Variable | Default | What to change |
|----------|---------|----------------|
| `LANG` | `"galician"` | Set to `"galician"` or `"guarani"` |
| `DATASET_PATH` | `"data/..."` | Path to the AMR dataset (HF disk format) |
| `TEST_SPLIT_SIZE` | `0.05` | Fraction of data held out for evaluation |
| `LORA_RANK` | `32` | LoRA rank — higher rank = more capacity, more VRAM |
| `OUTPUT_DIR` | auto-named | Directory where training checkpoints are saved |
| `ADAPTER_NAME` | auto-named | Local folder for the final saved adapter |

```bash
python train_general.py
```

**Outputs:**

| Path | Description |
|------|-------------|
| `results_{LANG}_mistral_7b_semantic_aware/` | Intermediate checkpoints |
| `mistral-7b-{LANG}-semantic-aware/` | Final LoRA adapter (use in Step 2) |
| `training_history_{LANG}_semantic-aware.xlsx` | Per-step loss log |

---

### Step 2 — Merge Adapter + Text-to-SQL Fine-Tuning

**Script:** `continued_training.py`

This script does two things in sequence:

1. **Merges** the AMR adapter from Step 1 into the base Mistral-7B weights,
   producing a self-contained model (`{LANG}-semantic-Mistral`) that no longer
   needs the adapter at inference time.
2. **Fine-tunes** a new LoRA adapter on top of this merged model for the
   Text-to-SQL task using the translated Spider dataset.

**Configuration:**

| Variable | Default | What to change |
|----------|---------|----------------|
| `LANG` | `"galician"` | Must match the language used in Step 1 |
| `ADAPTER_CHECKPOINT` | `"1439"` | **Update** to the checkpoint number produced by Step 1 (see `results_{LANG}_mistral_7b_semantic_aware/`) |
| `TRAIN_DATASET_PATH` | `"data/..."` | Spider train JSON |
| `EVAL_DATASET_PATH` | `"data/..."` | Spider dev JSON |
| `OUTPUT_DIR` | auto-named | Directory for SQL fine-tuning checkpoints |
| `ADAPTER_NAME` | auto-named | Local folder for the final SQL adapter |

> **Important:** After Step 1 completes, check `results_{LANG}_mistral_7b_semantic_aware/`
> for the final checkpoint folder (e.g., `checkpoint-1439`) and update `ADAPTER_CHECKPOINT`
> in `continued_training.py` accordingly.

```bash
python continued_training.py
```

**Outputs:**

| Path | Description |
|------|-------------|
| `{LANG}-semantic-Mistral/` | Merged semantically-aware base model |
| `results_{LANG}_mistral_7b_ft_on_spider-1-epoch/` | SQL fine-tuning checkpoints |
| `mistral-7b-{LANG}-ft-on-spider-1-epoch/` | Final SQL LoRA adapter (use in Step 3) |
| `training_history_{LANG}-ft-spider-1-epoch.xlsx` | Per-step loss log |

---

### Step 3 — Inference

**Script:** `inference.py`

Loads the merged semantically-aware model with the SQL adapter from Step 2,
then runs batched greedy decoding over the Spider dev set. Predicted SQL queries
are extracted from between `<sql>` and `</sql>` tags and written one per line.

**Configuration:**

| Variable | Default | What to change |
|----------|---------|----------------|
| `LANG` | `"galician"` | Must match Steps 1–2 |
| `CHECKPOINT_STEP` | `"438"` | **Update** to the checkpoint number from the SQL fine-tuning in Step 2 |
| `MERGED_MODEL_PATH` | `"{LANG}-semantic-Mistral"` | Path to the merged model from Step 2 |
| `EVAL_DATASET_PATH` | `"data/..."` | Spider dev JSON |
| `OUTPUT_SQL_PATH` | auto-named | Path for the predicted `.sql` file |
| `INFERENCE_BATCH_SIZE` | `32` | Reduce if you run out of VRAM during inference |

> **Important:** After Step 2 completes, find the final checkpoint folder under
> `results_{LANG}_mistral_7b_ft_on_spider-1-epoch/` and update `CHECKPOINT_STEP`
> in `inference.py`.

```bash
python inference.py
```

**Outputs:**

| Path | Description |
|------|-------------|
| `answer_files_sql/semantic-text2sql-{LANG}-mistral.sql` | One predicted SQL query per line, aligned with the dev set |

---

## Project Structure

```
semantic-infusion/
├── train_general.py          # Step 1: AMR semantic parsing fine-tuning
├── continued_training.py     # Step 2: Merge adapter + Text-to-SQL fine-tuning
├── inference.py              # Step 3: Batch inference on Spider dev set
├── requirements.txt          # Pinned Python dependencies
├── .env.example              # Template for the HF_KEY environment variable
└── data/
    ├── synthetic_text2sql_AMR_simple_SQL_glg_and_grn/    # AMR dataset (Step 1)
    └── spider_galician_and_guarani_translations_updated/  # Spider translations (Steps 2–3)
        ├── resdsql_train_Guarani_and_Galician.json
        └── resdsql_dev_Guarani_and_Galician.json
```

---

## Acknowledgements

- **Base model:** [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **SQL benchmark:** [Spider](https://yale-seas.yale.edu/spider/)
- **Semantic formalism:** [Abstract Meaning Representation (AMR)](https://amr.isi.edu/)
- **Fine-tuning framework:** [TRL SFTTrainer](https://huggingface.co/docs/trl) + [PEFT LoRA](https://huggingface.co/docs/peft)
