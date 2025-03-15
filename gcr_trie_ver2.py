from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from graph_util import get_undirected_graph, get_relation_graph,  get_tail
from datasets import load_dataset
import json
from collections import OrderedDict
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F
from relation_pruning import RoBERTaPruning
import gc
from SentenceBERT_pruning import SentenceBERTPruning
from trie import Trie

device = 'cuda:2'
weight = 2.0

def beam_search(logits, trie, pruning_scores, constrained_ids, beam_size=3, max_length=78, tokenizer=None, end_token_id=None):

    batch_size, seq_length, vocab_size = logits.shape
    assert batch_size == 1
    gcr = GraphConstrainedDecoding(tokenizer, trie)
    beams = [([], 0)]  

    logits_mask = torch.full_like(logits, -float('inf'))
    for idx, seq in enumerate(trie):
        for l, token_id in enumerate(seq):
            if logits_mask[:,l, token_id].item() == -float('inf'):
                logits_mask[:,l, token_id] = 0 

    logits = logits + logits_mask  
    for step in range(seq_length):
        new_beams = []
        
        for tokens, score in beams:
            if len(tokens) > 0 and tokens[-1] == end_token_id:
                new_beams.append((tokens, score))
                continue
            
            if step == 0:
                step_allowed_tokens = [seq[0] for seq in constrained_ids]
                step_pruning_scores = pruning_scores
            else:
                step_pruning_scores = []
                step_allowed_tokens = []
                for p, seq in enumerate(constrained_ids):
                    if len(seq) < step:
                        continue
                    if seq[step-1] == tokens[-1]:
                        step_allowed_tokens.append([seq[step]])
                        step_pruning_scores.append(pruning_scores[p].item())

            
            step_logits = logits[0, step]  
            allowed_step_logits = torch.zeros(len(step_allowed_tokens), device=step_logits.device)  # 결과 저장 텐서
            for i, token in enumerate(step_allowed_tokens):  # 각 allowed_token에 대해
                allowed_step_logits[i] = step_logits[token]
            
            final_step_logits = allowed_step_logits + torch.tensor(step_pruning_scores, dtype=torch.long, device=step_logits.device)
            log_probs = torch.nn.functional.log_softmax(final_step_logits, dim=-1) 
            indices = torch.tensor(step_allowed_tokens, dtype=torch.float, device=step_logits.device)
            #allowed_log_probs = log_probs[allowed_tokens] 
            #allowed_indices = torch.tensor(allowed_tokens, dtype=torch.long)

            if len(step_allowed_tokens) < beam_size:
                topk_log_probs, topk_indices = torch.topk(log_probs, len(step_allowed_tokens))
            else:
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            # 새로운 빔 확장
            for log_prob, index in zip(topk_log_probs, topk_indices):
                token_id = indices[index].item()
                new_tokens = tokens + [token_id]
                new_score = score + log_prob.item()
                new_beams.append((new_tokens, new_score))

        # 빔 스코어 기준 상위 `beam_size` 빔 선택
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        beams = new_beams

    # 상위 빔 반환
    best_sequences = sorted(beams, key=lambda x: x[1], reverse=True)[:beam_size]
    return best_sequences

class GraphConstrainedDecoding:
    def __init__(self, tokenizer, trie, start_token_ids=None, end_token_ids=None, enable_constrained_by_default=True):
        self.tokenizer = tokenizer
        self.trie = trie
        self.start_token = start_token_ids
        self.end_token = end_token_ids
        self.all_tokens = list(range(len(tokenizer)))
        self.constrained_flag = enable_constrained_by_default
        self.L_input = None

    def check_constrained_flag(self, sent: torch.Tensor):
        # Check start
        matched_start_token = torch.where(sent == self.start_token)[0]
        if len(matched_start_token) == 0:
            return False, len(sent)
        last_start_tokens = torch.where(sent == self.start_token)[0][-1]
        end_token_number = len(torch.where(sent[last_start_tokens:] == self.end_token)[0])
        # GCR not closed
        if end_token_number == 0:
            self.last_start_token = last_start_tokens
            return True, last_start_tokens
        else:
            self.last_start_token = None
            return False, len(sent)

    def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor):
        constrained_flag = self.constrained_flag
        # Check if enter the constrained decoding
        if self.start_token is not None and self.end_token is not None:
            constrained_flag, L_input = self.check_constrained_flag(sent)
        # Assign self.L_input with the input length
        else:
            if self.L_input is None:
                self.L_input = len(sent)
            L_input = self.L_input
            

        allow_tokens = self.trie.get(sent.tolist()[L_input:])
        if len(allow_tokens) == 0:
                return [self.tokenizer.eos_token_id]
        return allow_tokens



# 생성 모델 클래스 정의
class GraphConstrainedDecodingModel(AutoModelForCausalLM):
    def __init__(self, model, tokenizer, trie, start_token_ids=None, end_token_ids=None, print_scores=False):
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.start_token_ids = start_token_ids
        self.end_token_ids = end_token_ids
        self.print_scores = print_scores

    def generate(self, llm_input, pruning_scores, constrained_ids, max_length=100, **kwargs):
        
        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) 
        logits = outputs.logits
        
        responses= beam_search(logits, self.trie,  pruning_scores, constrained_ids, tokenizer=self.tokenizer, end_token_id=self.end_token_ids)
        generated_sentences = [self.tokenizer.decode(res[0]) for res in responses]
        print("Best sequence:", generated_sentences)
    
        return generated_sentences



prompt = "Generate Freebase relations when the head entity is {ent} which can be helpful to answer the question: {Q}"

prompt1 = '''
Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities.
Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.  

#Question:
{Q}

#Topic entities:
{ent}

#Reasoning path:
'''

model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
generation_cfg = GenerationConfig.from_pretrained(model_name)


generation_cfg.do_sample = False
generation_cfg.num_beams = 3
generation_cfg.num_return_sequences = 3
generation_cfg.output_scores = True

#path = '/home/hyemin/model/my_model/data/total_graph_cwq.json'
path = '/home/hyemin/model/my_model/data/total_graph_webqsp.json'
triples = json.load(open(path, 'r', encoding='utf-8'))
graph = get_undirected_graph()
rel_graph = get_relation_graph(triples)
dataset = load_dataset("rmanluo/RoG-webqsp", split='test')
one_hop_path = '/home/hyemin/model/my_model/data/webqsp/onehop_question.jsonl'
one_hop_ids = set()
with open(one_hop_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        one_hop_ids.add(data['id'])
        
#pruner = RoBERTaPruning(device = device, rel_graph=rel_graph, topk=50)
pruner = SentenceBERTPruning(device = device, rel_graph = rel_graph, topk=50)

end_token = tokenizer.eos_token
end_token_id = tokenizer.eos_token_id

   
#with open("output/trie/webqsp/add_pruningscores_weight_2.0.jsonl", "w", encoding="utf-8") as f:
for idx in tqdm(range(len(dataset))):
        if dataset[idx]['id'] not in one_hop_ids:
            continue
        line = OrderedDict()
        entity_list = dataset[idx]['q_entity']
        question = dataset[idx]['question']
        line['id'] = dataset[idx]['id']
        for ent in entity_list:
            if ent in rel_graph:
                relation_list, pruning_scores = pruner.pruning(ent, question)
                if len(relation_list) == 0:
                    line[ent] = []
                    continue
                pruning_scores = torch.tensor(pruning_scores, device='cuda:2')
                new_rel = []
                #new_rel = get_tail(graph, ent, relation_list)
                #pruning_scores =  F.softmax(torch.tensor(pruning_scores), dim=0)

                for rel in relation_list:
                    new_rel.append(f'{ent} -> {rel} {end_token}')
                relation_list = new_rel
            
                input_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in relation_list]

                trie = Trie(input_token_ids)
                gcd_model = GraphConstrainedDecodingModel(model, tokenizer, trie, end_token_ids = end_token_id)

                # 생성 문장
                prompt = prompt1.format(Q = question, ent = ent)
                
                generated_sentence = gcd_model.generate(prompt, pruning_scores, input_token_ids)
                
                #print("Generated sentence:", generated_sentence)
                line[ent] = generated_sentence
            
            else:
                pass
        
        #json.dump(line, f, ensure_ascii=False) 
        #f.write("\n")
        
        gc.collect()
        