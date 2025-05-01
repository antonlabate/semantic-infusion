import datasets
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import time
from transformers import (
    AutoModelForCausalLM,
   
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from huggingface_hub import login

login('hf_KEY')
model_name = "mistralai/Mistral-7B-Instruct-v0.3"#"ibm-granite/granite-3.0-8b-instruct"  """meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=False,
   bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(model_name,
  quantization_config=bnb_config,
  device_map="auto",
  trust_remote_code=True,
  use_auth_token=True
)

type_info = "syntax"
model = PeftModel.from_pretrained(base_model, f"./mistral-7b-guarani-gen-{type_info}-assistant_comp_only/")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")
tokenizer.pad_token = tokenizer.unk_token

#load test set
dataset_name = f"data/resdsql_dev_grnspider_llama_{type_info}.json" #resdsql_dev_ayr_Latnspider_llama.json
dataset = load_dataset("json", data_files = dataset_name,split="train")
prompts = []

def create_prompt(example):
  #bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = """You are a great software engineer and SQL expert. Your task is, given the user question and schema of a database, generate ONLY the true SQL answer to the question. 
    You should pay attention to the following guidelines while generating your answer:
      - Your answer must query only the columns that are needed to answer the question, with the appropriate operators. Be careful to not query for columns that do not exist in the schema. 
      - Pay attention to which column is in which table and use the proper columns for joins. 
      - Provide as an answer ONLY the true SQL answer without explanations or notes."""
  schema = example['input_sequence'].split("CREATE",1)[1]
  try:
        foreign_keys= schema.split("|",1)[1].replace("|",";")
        schema = schema.split("|",1)[0]
  except:
            foreign_keys = ""
  
  question = example['input_sequence'].split("CREATE",1)[0].split("[row]", 1)[0]
  phrase_structure = example['input_sequence'].split("CREATE",1)[0].split("[row]", 1)[1]
  answer = example['output_sequence'].split("|",1)[1]

  #q_user = f"""### Question: {question} \n ### Schema: CREATE  {schema} \n ### Foreign keys relations: {foreign_keys}\n\n"""
  q_user = f"""Schema: CREATE  {schema} \nForeign keys relations: {foreign_keys} \n\nQuestion: {question}\n"""
  
  #Uncomment the following (next) line for training with syntax or semantics appended to input question, instead of making the model generate it
  #q_user = f"""### Question: {example['input_sequence'].split("CREATE",1)[0]} \n ### Schema: CREATE {schema} \n ### Foreign keys relations: {foreign_keys} \n\n"""
  
  q_assistant = f"Syntactic structure: [row] "#{phrase_structure}\nTrue SQL answer: "

  messages = [
        {"role": "system", "content":system_message},
        {"role":"user", "content":q_user}
       
      ]

  prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
	    )
        
  prompt = prompt+q_assistant
  
  example["question"] =  prompt 

  return (example)

#map test set  
test = dataset.map(create_prompt) 
print(len(test))       

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=False)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs,
                                 max_new_tokens=150,
                                 do_sample=True,
                                 #repetition_penalty = 1.0,
                                 temperature = 0.3,
                                 num_beams = 3,
                                 #no_repeat_ngram_size = 3,
                                 pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids) #.detach().cpu().numpy())

  return decoded_output[0]


tam = len(dataset)

prompts = test["question"]
    
def get_sql_no_hal(sample):
    prompt = sample["question"]
    #print(prompt)
    gen = generate_response(prompt, model)
    
    print("Total")
    
    print(gen)
    try:
        gen = gen.split("True SQL answer: ", 1)[1].strip()
    
    except:

        gen= ""
    

    try:
      gen = gen.replace("</s>", "")
    except:
      gen = gen
    
    
    
    sample["sql"] = gen
    
    return sample
    
sql_answers = test.map(get_sql_no_hal)
sql_answers.save_to_disk(f"answers/guarani_model_generating_{type_info}_assistant_no_ind_comp_only")

with open(f"pred_guarani_with_{type_info}_assistant_no_ind_comp_only.sql",'w') as file:
  for k in range(len(sql_answers["sql"])):
    file.write(sql_answers[k]["sql"])
    file.write("\n")


