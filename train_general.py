import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
from huggingface_hub import login
import pandas as pd

login('HF_LOGIN_KEY')

model_id="mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=1024, padding="max_length", padding_side='left', truncation=True)
tokenizer.pad_token = tokenizer.unk_token


bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
      model_id, quantization_config=bnb_config, device_map="auto"
)
model.config.use_cache=False
model.config.pretraining_tp=1
type_info = "syntax"


if torch.cuda.device_count() > 1: # If more than 1 GPU
    print(torch.cuda.device_count())
    model.is_parallelizable = True
    model.model_parallel = True

dataset_name = f"data/resdsql_train_grnspider_llama_{type_info}.json"
#dataset_name = f"data/resdsql_train_wspider_llama_infusion.json"
dataset = load_dataset("json", data_files = dataset_name,split="train")

# Validation dataset
dataset_eval_name = f"data/resdsql_dev_grnspider_llama_{type_info}.json"
#dataset_name = f"data/resdsql_train_wspider_llama_infusion.json"
dataset_eval = load_dataset("json", data_files = dataset_eval_name,split="train")

def formatting_prompts_func(example):
    output_texts = []
    system_message = """You are a great software engineer and SQL expert. Your task is, given the user question and schema of a database, generate ONLY the true SQL answer to the question. 
    You should pay attention to the following guidelines while generating your answer:
      - Your answer must query only the columns that are needed to answer the question, with the appropriate operators. Be careful to not query for columns that do not exist in the schema. 
      - Pay attention to which column is in which table and use the proper columns for joins. 
      - Provide as an answer ONLY the true SQL answer without explanations or notes."""

    for i in range(len(example["input_sequence"])):
        print(i)
        print(example['input_sequence'][i])
        schema = example['input_sequence'][i].split("CREATE",1)[1] #.split("|",1)[0]
        
        try:
            foreign_keys= schema.split("|",1)[1].replace("|",";")
            schema = schema.split("|", 1)[0]
        except:
            foreign_keys = ""
      
        answer = example['output_sequence'][i].split("|",1)[1]
        question = example['input_sequence'][i].split("CREATE",1)[0].split("[row]", 1)[0]
        phrase_structure = example['input_sequence'][i].split("CREATE",1)[0].split("[row]", 1)[1]
        q_user = f"""Schema: CREATE  {schema} \nForeign keys relations: {foreign_keys}\n\nQuestion: {question}\n"""
       

        #Uncomment the following (next) line for training with syntax or semantics appended to the input question, instead of making the model generate it
        #q_user = f"""### Question: {example['input_sequence'][i].split("CREATE",1)[0]} \n ### Schema: CREATE  {schema} \n ### Foreign keys relations: {foreign_keys}\n\n"""
        q_assistant =  f"Syntactic structure: [row] {phrase_structure}\nTrue SQL answer: {answer}" #</s>"
        messages = [
        {"role": "system", "content":system_message},
        {"role":"user", "content":q_user},
        {"role":"assistant", "content": q_assistant }
        ]

        prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
	    )
        
        #prompt = prompt+q_assistant
       
        output_texts.append(prompt)

    return output_texts

# Load LoRA configuration
peft_config = LoraConfig(
     r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules='all-linear',
    #modules_to_save=["embed_tokens","lm_head"],
    bias="none",
    task_type="CAUSAL_LM",
)

response_template = "[/INST]"

training_arguments = TrainingArguments(
    output_dir=f"./results_guarani_mistral_7b_gen_{type_info}_assistant_comp_only_all_quant",
    warmup_ratio=0.03,
    learning_rate=2e-5,
    #max_steps = max_steps,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="constant",
    group_by_length=True,
    logging_steps=0.01,
    save_strategy="steps",
    save_steps=0.2,
    save_total_limit=2,
    #weight_decay=0.001,
    #max_grad_norm=0.3,
    evaluation_strategy = "steps",
    eval_steps = 0.1,
    do_eval=True
    #disable_tqdm=disable_tqdm,
#    report_to="tensorboard",
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset_eval,
    peft_config=peft_config,
    #dataset_text_field="text",
    max_seq_length=1024, #None: 1024 or model's max seq length
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    args=training_arguments,
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False),
)

# Train model
trainer.train()
new_model = f"mistral-7b-guarani-gen-{type_info}-assistant_comp_only_all"
#trainer.save_model()
# Save trained model
trainer.model.save_pretrained(new_model)

print("Log history")
df = pd.DataFrame(trainer.state.log_history)
print(df)
df.to_excel(f"training_history_gen_{type_info}_assistant_comp_only_all_guarani.xlsx")


# Save checkpoint every X updates steps
#save_steps = 3000

# Log every X updates steps
#logging_steps = 100

