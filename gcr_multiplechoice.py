import gc
import json
import torch
from tqdm import tqdm 
from datasets import load_dataset
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from constrained_decoding_prompt_list import *
from SentenceBERT_pruning import SentenceBERTPruning
from graph_util import get_undirected_graph, get_relation_graph,  get_tail_graph

device = 'cuda:2'
model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
generation_cfg = GenerationConfig.from_pretrained(model_name)

d = 'webqsp'
#d = 'cwq'
path = f'/home/hyemin/model/my_model/data/{d}/total_graph_{d}.json'
triples = json.load(open(path, 'r', encoding='utf-8'))
graph = get_undirected_graph(triples)
rel_graph = get_relation_graph(triples)
tail_graph = get_tail_graph(triples)
dataset = load_dataset(f"rmanluo/RoG-{d}", split='test')

pruner = SentenceBERTPruning(device = device, tail_graph=tail_graph,rel_graph = rel_graph, topk=50)
one_hop_path = f'/home/hyemin/model/my_model/data/{d}/onehop_question.jsonl'
one_hop_ids = set()
with open(one_hop_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        one_hop_ids.add(data['id'])

generation_cfg.do_sample = False
generation_cfg.num_beams = 3
generation_cfg.num_return_sequences = 3

weight = 1.0
with open(f"output/trie/{d}/givetail_pruning50_3shot_onlyllm.jsonl", "w", encoding="utf-8") as f:
    for idx in tqdm(range(len(dataset))):
        if dataset[idx]['id'] not in one_hop_ids:
            continue
        line = OrderedDict()
        entity_list = dataset[idx]['q_entity']
        question = dataset[idx]['question']
        line['id'] = dataset[idx]['id']
        for ent in entity_list:
            if ent in rel_graph:
                relation_list, _, pruning_scores= pruner.pruning([ent], question)
                if len(relation_list) < 4:
                    line[ent] = relation_list
                    continue
                '''
                #generate cot process
                prompt_a = COT_PROMPT2.format(Q=question, ent=ent, rels=relation_list)
                inputs_a = tokenizer(prompt_a, return_tensors="pt", add_special_tokens=False)
                inputs_a_ids = inputs_a.input_ids.to(device)
                inputs_a_attention_mask = inputs_a.attention_mask.to(device)
                response = model.generate(input_ids=inputs_a_ids, attention_mask=inputs_a_attention_mask, max_new_tokens=150, pad_token_id= tokenizer.eos_token_id)
                cot = tokenizer.decode(response[0][inputs_a_ids.shape[1]:], skip_special_tokens=True)
                try: 
                    cot = cot.split('#Question:')[0]
                except:
                    pass
                print(cot)
                #####
                '''
                valid_numbers = range(1,len(relation_list)+1)
                valid_tokens = [tokenizer.encode(str(num), add_special_tokens=False)[0] for num in valid_numbers]
                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    return valid_tokens
                #rels = '\n'.join(f'{idx+1}. {rel}' for idx, rel in enumerate(relation_list))
                temp= []
                for i, r in enumerate(relation_list):
                    tails = tail_graph[(ent, r)]
                    temp.append(f'{i+1}. {r} {tails[:5]}')
                rels = '\n'.join(temp)
                prompt = multiplechoice_tail.format(shot1=mp_tail_shot1, shot2=mp_tail_shot2, shot3=mp_tail_shot3, Q=question, ent=ent, rels=rels)
                #prompt = multiplechoice.format(shot1=mp_shot4, shot2=mp_shot2, shot3=mp_shot3, Q=question, ent=ent, rels=rels)
                
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs.input_ids.to(device)

                attention_mask = inputs.attention_mask.to(device)
                response = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                generation_config= generation_cfg,
                                max_new_tokens = 1,
                                prefix_allowed_tokens_fn= prefix_allowed_tokens_fn,
                                return_dict_in_generate=True,
                                output_scores=True,
                                pad_token_id= tokenizer.eos_token_id,
                            )
                #logits = response.scores[0]
                #logits = torch.nn.functional.softmax(logits, dim=-1) 
                '''
                logits_mask = torch.full_like(logits, -float('inf'))
                pruning_scores = torch.tensor(pruning_scores, device=device)
                pruning_scores =  torch.nn.functional.softmax(pruning_scores, dim=-1)

                for i, t in enumerate(valid_tokens):
                    logits_mask[0, t] = weight * pruning_scores[i]
                logits = torch.nn.functional.softmax(logits, dim=-1) + logits_mask
                '''
                #top_k = 3
                #top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
                #decoded_tokens = [tokenizer.decode(idx.item()).strip() for idx in top_k_indices[0]]
                decoded_tokens = [tokenizer.decode(r[input_ids.shape[1]:]) for r in response.sequences]
                pred_rel = [relation_list[int(rel)-1] for rel in decoded_tokens]
                line[ent] = pred_rel
                print(question, pred_rel)
            else:
                pass

        json.dump(line, f, ensure_ascii=False) 
        f.write("\n")
        
        gc.collect()