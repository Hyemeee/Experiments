from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import ast

#setting
pretrained_model = 'bert-base-uncased'
max_num_tokens = 512
batch_size = 1024
use_cuda = True
top_k = 5
device = 'cuda:1'

#get extracted topics
path1 = '/home/hyemin/model/my_model/cwq_topic_entity_keybert.json'
keybert_topics = json.load(open(path1, 'r', encoding='utf-8')) 
keybert_topics = keybert_topics[:50]
#load pretrained bert 
t_bert = AutoModel.from_pretrained(pretrained_model)
e_bert = AutoModel.from_pretrained(pretrained_model)

t_bert = nn.DataParallel(t_bert, device_ids=[1,2]).cuda(device)
e_bert = nn.DataParallel(e_bert, device_ids=[1,2]).cuda(device)

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

#get entity embeddings
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

ent_embeddings = get_ent_embedding(entities, batch_size)

query_topic_map = {}
with torch.no_grad():
    for start in tqdm(range(len(keybert_topics))):
        topics_1 = keybert_topics[start]['topic_entity']
        topics_2 = keybert_topics[start]['diverse']
        topic_list1 = [i[0] for i in topics_1]
        topic_list2 = [i[0] for i in topics_2]
        #naive topics
        encoded1 = tokenizer(text=topic_list1, add_special_tokens=True, max_length = max_num_tokens, return_token_type_ids=True, truncation=True, padding=True, return_tensors='pt')
        encoded1 = {key: tensor.to(device) for key, tensor in encoded1.items()}
        outputs1 = t_bert(**encoded1)
        t_emb1 = outputs1.last_hidden_state[:,0,:]
        normed_t_emb1 = nn.functional.normalize(t_emb1, dim=1)

        #diverse topics
        encoded2 = tokenizer(text=topic_list2, add_special_tokens=True, max_length = max_num_tokens, return_token_type_ids=True, truncation=True, padding=True, return_tensors='pt')
        encoded2 = {key: tensor.to(device) for key, tensor in encoded2.items()}
        outputs2 = t_bert(**encoded2)
        t_emb2 = outputs2.last_hidden_state[:,0,:]
        normed_t_emb2 = nn.functional.normalize(t_emb2, dim=1)
        
        #calculate cosine similarity
        probs1 = normed_t_emb1.mm(ent_embeddings.t())
        probs2 = normed_t_emb2.mm(ent_embeddings.t())
        #select top-k entities
        top_k_values_1, top_k_indices_1 = torch.topk(probs1, top_k, dim=1)
        top_k_values_2, top_k_indices_2 = torch.topk(probs2, top_k, dim=1)
        
 
        top_k_entities_1 = []
        top_k_entities_2 = []
        
        for i in range(len(topic_list1)):
            top_ent = [entities[idx] for idx in top_k_indices_1[i].tolist()]
            top_k_entities_1.append(top_ent)
        
        for i in range(len(topic_list2)):
            top_ent = [entities[idx] for idx in top_k_indices_2[i].tolist()]
            top_k_entities_2.append(top_ent)
        
        query_id = keybert_topics[start]['id']
        query_topic_map[query_id] = {
                "topic entity_naive": top_k_entities_1,
                "topic entity_diverse" : top_k_entities_2
            }

with open('cwq_topic_entity_map_keybert.json', 'w', encoding='utf-8') as file:
    json.dump(query_topic_map, file, ensure_ascii=False, indent=4)