from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from datasets import load_dataset

#setting
pretrained_model = 'bert-base-uncased'
max_num_tokens = 512
batch_size = 1024
use_cuda = True
top_k = 3
device = 'cuda:1'

#get extracted topics
path1 = '/home/hyemin/model/my_model/output/topic/CWQ_Llama_ver2_p.json'
llama_topics = json.load(open(path1, 'r', encoding='utf-8')) 

#load pretrained bert 
t_bert = AutoModel.from_pretrained(pretrained_model)
e_bert = AutoModel.from_pretrained(pretrained_model)

#Activate for using multi-gpu
#t_bert = nn.DataParallel(t_bert, device_ids=[0,1]).cuda(device)
#e_bert = nn.DataParallel(e_bert, device_ids=[0,1]).cuda(device)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

t_bert.to(device)
e_bert.to(device)
t_bert.eval()
e_bert.eval()

#get entity in total graph
path2 = '/home/hyemin/model/my_model/data/total_graph_cwq.json'
triples = json.load(open(path2, 'r', encoding='utf-8')) 
entities = sorted(set(t[0] for t in triples).union(set(t[2] for t in triples)))
print('start encoding entities')

def parse_topics(topics):
    # 불필요한 공백 제거 및 대괄호 제거
    topics = topics.strip()[1:-1]
    # 요소 분리 및 양쪽 공백 제거
    #topic_list = [item.strip().strip("'\"") for item in topics.split(",")]
    topic_list = [item for item in topics.split(",")]
    return topic_list

#Activate for get entity embeddings
#dataset = load_dataset("rmanluo/RoG-cwq", split='test')
#idx_list = [3164, 3280, 3344, 2029, 2649, 645, 546, 2134, 2633, 1514, 604, 213, 2799, 2204, 3239, 2565, 543, 2692, 1286, 1174, 407, 1483, 1654, 2947, 2741, 746, 1822, 214, 2957, 2122, 2485, 81, 3007, 2765, 1221, 2049, 1531, 1515, 3173, 3401, 2126, 362, 3418, 3459, 39, 2767, 1778, 2205, 1991, 3035]
#topics = [dataset[i]['q_entity'] for i in idx_list]

def get_ent_embedding(entities,  batch_size):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(entities), batch_size)):
            batch_data = entities[i:i+batch_size]
            encoded_inputs = tokenizer(text=batch_data,
                               add_special_tokens=True,
                               max_length=max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True,
                               padding=True,
                               return_tensors='pt')
            encoded_inputs = {key: tensor.to(device) for key, tensor in encoded_inputs.items()}
            outputs = e_bert(**encoded_inputs)
            e_emb = outputs.last_hidden_state[:, 0, :]
            normed_e_emb = nn.functional.normalize(e_emb, dim=1)
            embeddings.append(normed_e_emb)
            torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)

#get topic entity embeddings
def get_topic_embedding(entities):
    with torch.no_grad():
        encoded_inputs = tokenizer(text=entities,
                               add_special_tokens=True,
                               max_length=max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True,
                               padding=True,
                               return_tensors='pt')
        encoded_inputs = {key: tensor.to(device) for key, tensor in encoded_inputs.items()}
        outputs = e_bert(**encoded_inputs)
        e_emb = outputs.last_hidden_state[:, 0, :]
        normed_e_emb = nn.functional.normalize(e_emb, dim=1)
        torch.cuda.empty_cache()
    return normed_e_emb

ent_embeddings = get_ent_embedding(entities, batch_size)

query_topic_map = {}
with torch.no_grad():
    for start in tqdm(range(len(llama_topics))):
        topic_list = llama_topics[start]['topic_entity']
        #topic_list = ast.literal_eval(topics)
        #topic_list = parse_topics(topics)
        encoded = tokenizer(text=topic_list, add_special_tokens=True, max_length = max_num_tokens, return_token_type_ids=True, truncation=True, padding=True, return_tensors='pt')
        encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
        outputs = t_bert(**encoded)
        t_emb = outputs.last_hidden_state[:,0,:]
        normed_t_emb = nn.functional.normalize(t_emb, dim=1)

        #ent_embeddings = get_topic_embedding(topics[start])

        #calculate cosine similarity
        probs = normed_t_emb.mm(ent_embeddings.t())
        #select top-k entities
        top_k_values, top_k_indices = torch.topk(probs, top_k, dim=1)

        #scores = []
        top_k_entities = []
        for i in range(len(topic_list)):
            #score = top_k_values[i].tolist()
            #scores.append(score)
            top_ent = [entities[idx] for idx in top_k_indices[i].tolist()]
            #golden
            #top_ent = [topics[start][idx] for idx in top_k_indices[i].tolist()]
            top_k_entities.append(top_ent)
        
        query_id = llama_topics[start]['id']
        query_topic_map[query_id] = {
                #"similarity_scores":scores,
                "llama topic": topic_list, 
                "topic entity": top_k_entities
            }

with open('/home/hyemin/model/my_model/output/mapping/CWQ_Llama_ver2.json', 'w', encoding='utf-8') as file:
    json.dump(query_topic_map, file, ensure_ascii=False, indent=4)