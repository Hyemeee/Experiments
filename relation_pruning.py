from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

from datasets import load_dataset
from graph_util import *
from tqdm import tqdm


class RoBERTaPruning:
    def __init__(self, model = None, tokenizer=None, device='cuda:0', rel_graph=None, golden_rel=None, topk=50):
        if model is None:
            model_name = "roberta-base"
            model = AutoModel.from_pretrained(model_name) 
        if tokenizer is None:    
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if rel_graph is None:
            #path = '/home/hyemin/model/my_model/data/total_graph_cwq.json'
            path = '/home/hyemin/model/my_model/data/total_graph_webqsp.json'
            triples = json.load(open(path, 'r', encoding='utf-8')) 
            rel_graph = get_relation_graph(triples)
        if golden_rel is None:
            #golden_rel = json.load(open('/home/hyemin/model/my_model/data/add_reverse_golden.json', 'r', encoding='utf-8'))
            golden_rel = json.load(open('/home/hyemin/model/my_model/data/WebQSP_add_reverse_golden.json', 'r', encoding='utf-8'))
            
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.rel_graph = rel_graph
        self.golden_rel = golden_rel
        self.device = device
        self.topk = topk   
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        normed_emb = nn.functional.normalize(cls_embedding, dim=1)
        return normed_emb

    def pruning(self, ent, question, elim_wh = False):
        if elim_wh:
            question_words = ["what", "who", "where", "when", "which", "how"]
            question = question.rstrip('?')    
            question = question.split()
            for i, que in enumerate(question):
                if que.lower() in question_words:
                    question.pop(i)
            question = " ".join(question)
        
        q_embedding = self.get_embedding(question)
        cand_relations = list(self.rel_graph[ent])
        if len(cand_relations) <= self.topk:
            return cand_relations      
        normed_relations = normalize_relation(cand_relations, ent)
        relation_embeddings = self.get_embedding(normed_relations)
        probs = q_embedding.mm(relation_embeddings.t())
        top_k_values, top_k_indices = torch.topk(probs, self.topk, dim=1)
        topk_relation_list = [cand_relations[idx] for idx in top_k_indices[0]]
        
        return topk_relation_list
    
    def evaluate(self, answer, prediction):   
        acc = eval_acc(prediction, answer)
        hit = eval_hit(prediction, answer)
        hit1 = eval_hit1(prediction, answer)
        f1, _, _ = eval_f1(prediction, answer)
               

        return hit, hit1, f1, acc



def main():
    total_acc = 0
    total_hit = 0
    total_hit1 = 0
    total_f1 = 0
    count = 0
    pruner = RoBERTaPruning(topk=50)
    
    dataset = load_dataset("rmanluo/RoG-webqsp", split='test')
    for idx in tqdm(range(len(dataset))):
        for ent in dataset[idx]['q_entity']:
            answer = pruner.golden_rel[idx]['golden_rel']
            if ent not in pruner.rel_graph:
                    continue
            
            if not len(pruner.rel_graph[ent]) > 50 :
                continue
            
            if len(answer) == 0:
                continue
            
            prediction = pruner.pruning(ent, dataset[idx]['question'])
            hit, hit1, f1, acc = pruner.evaluate(answer, prediction)
            count+=1   
            total_acc += acc
            total_hit += hit
            total_hit1 +=hit1
            total_f1 +=f1
            
    print('count: ', count)
    print('hit: ', total_hit/count)
    print('hit1: ', total_hit1/count)
    print('f1: ', total_f1/count)
    print('acc: ', total_acc/count)
    
#main()