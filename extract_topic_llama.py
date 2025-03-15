import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json
import random

# Set device for processing
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'

llm = "meta-llama/Llama-3.1-8B-Instruct"
#llm = "google/gemma-2-9b-it"
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(llm)
model = AutoModelForCausalLM.from_pretrained(llm).to(device)

# Set pad_token to eos_token or add a new pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  
tokenizer.padding_side = "left"  
# Define prompt and examples
#prompt = "You are a helpful AI assistant. Please extract the potential topic entities from the question."
prompt = '''
Given a question and the topic entities within it, 
please extract any additional topic entities from the question, including the provided topic entities. 
If you determine there are no additional topic entities worth extracting, respond with only the given topic entities. 
However, if there are additional topic entities, respond with both the given and the additional entities.
'''

examples = '''
**Example 1:**

Topic Entity: ['Continent', 'Central Time Zone']

Question: What Central Time Zone continent do the Falkland Islands belong to?

Answer: ['Continent', 'Central Time Zone', 'Falkland Islands']

**Example 2:**

Topic Entity: ['Lou Seal', 'mascot', 'team', 'World Series']

Question: Lou Seal is the mascot for the team that last won the World Series when?

Answer: ['Lou Seal', 'mascot', 'team', 'World Series']

**Example 3:**

Topic Entity: ['France', 'Nijmegen']

Question: What country bordering France contains an airport that serves Nijmegen?

Answer: ['country', 'France', 'airport', 'Nijmegen']

**Example 4:**

Topic Entity: ['Country Nation World Tour', "Bachelor's degree"]

Question: Where did the \"Country Nation World Tour\" concert artist go to college?

Answer: ['Country Nation World Tour', "Bachelor's degree", 'College/University']
'''



# Load dataset and set batch size
dataset = load_dataset("rmanluo/RoG-cwq", split='test')
idx_list = [3164, 3280, 3344, 2029, 2649, 645, 546, 2134, 2633, 1514, 604, 213, 2799, 2204, 3239, 2565, 543, 2692, 1286, 1174, 407, 1483, 1654, 2947, 2741, 746, 1822, 214, 2957, 2122, 2485, 81, 3007, 2765, 1221, 2049, 1531, 1515, 3173, 3401, 2126, 362, 3418, 3459, 39, 2767, 1778, 2205, 1991, 3035]
#idx_list = random.sample(range(3,len(split_q)), 50)
#print(idx_list)

# Accumulate results in a list
all_results = []
i = True
while i:
    for i in tqdm(range(len(dataset))):
        q = dataset[i]['question']
        id = dataset[i]['id']
        given_entity = dataset[i]['q_entity']
        input_texts = f"{prompt} \n{examples} \nTopic Entity: {given_entity} \nQuestion: {q} \nAnswer: " 

        input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        #context = tokenizer.batch_decode(input_ids['input_ids'], return_tensors="pt", padding=True, truncation=True, clean_up_tokenization_spaces=False)[0]
    
        # Generate output in batch
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids['input_ids'], 
                attention_mask=input_ids['attention_mask'],
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and process each generated output in the batch
        generated_texts = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #gresponse = generated_texts.replace(context, "").strip()
        
        try:
            a = generated_texts.split(prompt)[1]
            topic_entity = generated_texts.split(q)[1]
        except IndexError:
            topic_entity = generated_texts  

        result = {
            "id": id,
            "question": q,
            "topic_entity": topic_entity
            }
        all_results.append(result)


output_path = '/home/hyemin/model/my_model/output/topic/CWQ_Llama_ver2.json'
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(all_results, file, ensure_ascii=False, indent=4)
