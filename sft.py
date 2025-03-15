from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, HfArgumentParser, Trainer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from dataclasses import dataclass, field
from typing import Optional, List
from utils import *
from datasets import Dataset
from random import shuffle
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

N_CPUS = 1

SUBQ_ZERO_SHOT_PROMPT = """You are an expert of world knowledge with strong logical skills.
Given the provided candidate reasoning paths, answer the given question based on the tail entity of the reasoning paths.
The answer must be the tail entity of a candidate reasoning path.
If there is no suitable entity, return ‘None’.

Question: 
{question}
Candidate reasoning paths: 
{entities}"""

SUBQ_ANS_TEMPLATE = """
Return : {answer}"""

TOTALQ_ZERO_SHOT_PROMPT = """You are an expert of world knowledge with strong logical skills.
When a question and candidate reasoning paths are given, determine whether the provided reasoning paths are sufficient to answer the question.
If the candidate reasoning paths are enough to answer the question, return “yes”. If they are not sufficient, return “no”.
You must return either “yes” or “no”, and nothing else.

Question: 
{question}
Candidate reasoning paths: 
{entities}"""

TOTALQ_ANS_TEMPLATE = """
Return : {answer}"""

def load_dataset(x):
    paths = ["/home/minbae/KGQA/webqsp_sft_train_path_subQ.jsonl", "/home/minbae/KGQA/sft_train_totalq1000.jsonl"]
    data_list = []
    with open(paths[0], 'r', encoding='utf-8') as f:
        for line in f:
            temp = json.loads(line)
            data_list.append({'category' : temp['category'], 'question' : temp['question'], 'a_entity' : temp['a_entity'], 'topic' : temp['topic'], 'cand_paths' : temp['cand_paths'], 'a_path' : temp['a_path'], 'answer' : "None"})
    with open(paths[1], 'r', encoding='utf-8') as f:
        for line in f:
            temp = json.loads(line)
            data_list.append({'category' : temp['category'], 'question' : temp['question'], 'a_entity' : ["None"], 'topic' : ['None'], 'cand_paths' : temp['cand_path'], 'a_path' : [['None']], 'answer' : temp['answer']})
    return Dataset.from_list(data_list)

def get_undirected_graph(triples):
    G = nx.MultiGraph()
    for triple in triples:
        h, r, t = triple
        G.add_edge(h, t, relation=r.strip())
    return G

@dataclass
class ScriptArguments:
    # data_path_list: Optional[str] = field(
    #     default=["/home/minbae/KGQA/sft_train.jsonl"],
    #     metadata={"help": "Path to the training data."}
    # )
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    rel_dict_path: List[str] = field(
        default=None, metadata={"help": "Path to the relation dictionary."}
    )
    add_rel_token: Optional[bool] = field(
        default=False, metadata={"help": "Wether to add relation token or not"}
    )
    prompt_path: str = field(
        default="prompts/llama2.txt",
        metadata={"help": "Path to the prompt template"},
    )
    use_peft: Optional[bool] = field(
        # default=False,
        default=True,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "the lora r parameter"}
    )
    response_template: Optional[str] = field(
        default="<start_of_turn>model\n", metadata={"help": "Response template"}
    )
    graph_path: str = field(
        default="/home/minbae/KGQA/total_graph_webqsp.json",
    )

@dataclass
class ScriptTrainingArguments(SFTConfig):
    output_dir: str = field(
        default="saved_models/llama2_align",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=3072,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # max_new_tokens : int = field(
    #     default=100,
    #     metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    # )
    ddp_find_unused_parameters: bool = field(default=False)


def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained( # 인자값 수정하기
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",
        use_auth_token=True,
        # low_cpu_mem_usage=True,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            # target_modules=["q_proj", "v_proj"],
            # target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
            target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True,
        )
    else:
        peft_config = None

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    # if tokenizer.pad_token is None:
    #     assert tokenizer.unk_token is not None
    #     tokenizer.pad_token = tokenizer.unk_token

    # tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    
    # graph = get_undirected_graph(json.load(open(script_args.graph_path, 'r', encoding='utf-8')))
    
    # data_list =  [load_dataset(data_path) for data_path in script_args.data_path_list]
    # dataset = load_dataset(["/home/minbae/KGQA/webqsp_sft_train_path_subQ.jsonl", "/home/minbae/KGQA/sft_train_totalq1000.jsonl"])
    dataset = load_dataset('A')

    def input_formatter(examples):
        chunks = []
        for i in range(len(examples["question"])):
            question = examples["question"][i]
            # answer = examples["a_entity"][i]
            # cand_ents = examples['cand_ents'][i]
            answer = examples["a_entity"][i]
            yesorno = examples["answer"][i]
            cand_ents = examples['cand_paths'][i]
            category = examples['category'][i]
            shuffle(cand_ents)

            cand_ents = [" -> ".join(sublist) for sublist in cand_ents]

            if not question.endswith("?"):
                question += "?"
            
            if category == 'SubQ':
                raw_input = SUBQ_ZERO_SHOT_PROMPT.format(
                    # question=question, entities=",".join(cand_ents)
                    question=question, entities="\n".join(cand_ents)
                )
                response = SUBQ_ANS_TEMPLATE.format(
                    answer=answer
                )

            elif category == 'TotalQ':
                raw_input = TOTALQ_ZERO_SHOT_PROMPT.format(
                    question=question, entities=cand_ents
                )
                response = TOTALQ_ANS_TEMPLATE.format(
                    answer=yesorno
                )

            chat = [
                {"role": "user", "content": raw_input},
                {"role": "assistant", "content": response},
            ]
            final_input = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
            chunks.append(final_input)
        return {"text": chunks}

    train_dataset = dataset.map(
        input_formatter,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=N_CPUS,
    )

    response_template = script_args.response_template
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer, mlm=False
    )

    trainer = SFTTrainer(
        train_dataset=train_dataset,
        model=model, 
        peft_config = peft_config,
        dataset_text_field = 'text',
        max_seq_length = 2024,
        tokenizer=tokenizer, 
        args=training_args, 
        data_collator=data_collator,
    )

    trainer.train()
    torch.cuda.empty_cache()
    
    trainer.save_model(training_args.output_dir)
    # trainer.model.save_pretrained(training_args.output_dir)



if __name__ == "__main__":
    train()
    
    
     