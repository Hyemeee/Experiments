from graph_util import eval_hit, eval_f1
from tqdm import tqdm
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import OrderedDict
from graph_util import *


dataset = load_dataset("rmanluo/RoG-webqsp", split='test')
file_path = '/home/hyemin/model/my_model/KGQA/output/trie/webqsp/pruning50_weight2.0_multiple_choice.jsonl'
prediction = []
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        prediction.append(data)
 
graph = get_undirected_graph()        
total_hit = 0
total_f1 = 0 
cnt = 0

#**Example**: 
#Reasoning Paths: ["'Lou Seal' -> 'sports.mascot.team' -> 'San Francisco Giants' -> 'sports.sports_championship_event.champion' -> '2014 World Series'"]
#Question:  'Lou Seal is the mascot for the team that last won the World Series when?'
#Answer: "2014"

prompt = '''
Based on the reasoning paths, please answer the given question.Please keep the answer as simple as possible and return all the possible answers as a list.

Reasoning Paths: {pred}

Question: {Q}

Answer:
'''
device = 'cuda:1'
llm = "meta-llama/Llama-3.1-8B-Instruct"
llm = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(llm)
model = AutoModelForCausalLM.from_pretrained(llm).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  
tokenizer.padding_side = "left"  

with open("gcr_rel_reasoning_output_ver2.jsonl", "w", encoding="utf-8") as f:
    for idx in tqdm(range(len(dataset))):
        line = OrderedDict()
        line['id'] = dataset[idx]['id']
        preds = []
        for ent in dataset[idx]['q_entity']:
            if ent not in prediction[idx]:
                continue
            paths = prediction[idx][ent]
            pred = [p.split(' -> ')[1] for p in paths]
            paths = get_tail(graph, ent, pred)
            preds.extend(paths)

        input = prompt.format(pred=preds, Q=dataset[idx]['question'])
        input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                    input_ids=input_ids['input_ids'], 
                    attention_mask=input_ids['attention_mask'],
                    max_new_tokens= 512,
                    pad_token_id=tokenizer.eos_token_id
                )
        generated_text = tokenizer.decode(outputs[0])
        try: 
            generated = generated_text.split('Answer: ')[1]
        except IndexError:
            generated = generated_text 
        
        line['output'] = generated 
        
        json.dump(line, f, ensure_ascii=False) 
        f.write("\n")

        if idx == 10:
            break