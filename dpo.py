import logging
import json
import os
import torch
import sys
import random
from typing import Optional, List
from datasets import Dataset
from random import shuffle
from dataclasses import dataclass, field
import transformers
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, HfArgumentParser, set_seed


SUBQ_ZERO_SHOT_PROMPT = """You are an expert of world knowledge with strong logical skills.
Given the provided candidate reasoning paths, answer the given question based on the tail entity of the reasoning paths.
The answer must be the tail entity of a candidate reasoning path.
If there is no suitable entity, return ‘None’.

Question: 
{question}
Candidate reasoning paths: 
{paths}"""

SUBQ_ANS_TEMPLATE = """
Return : {answer}"""

TOTALQ_ZERO_SHOT_PROMPT = """You are an expert of world knowledge with strong logical skills.
When a question and candidate reasoning paths are given, determine whether the provided reasoning paths are sufficient to answer the question.
If the candidate reasoning paths are enough to answer the question, return “yes”. If they are not sufficient, return “no”.
You must return either “yes” or “no”, and nothing else.

Question: 
{question}
Candidate reasoning paths: 
{paths}"""

TOTALQ_ANS_TEMPLATE = """
Return : {answer}"""

EPRUNING_ZERO_SHOT_PROMPT = """You are an expert of world knowledge with strong logical skills.
When a question and candidate answer entity list are given, determine which answer entities can be used to answer the question.
You must return answer from the candidate entity list. If there are no suitable entities, return "None".
Please provide the minimum possible number of entities.

Question: 
{question}
Candidate answer entity: 
{entities}"""

TOTALQ_ANS_TEMPLATE = """
Return : {answer}"""

def load_dataset(paths):
    data_list = []
    # with open(paths[0], 'r', encoding='utf-8') as f:
    #     for line in f:
    #         temp = json.loads(line)
    #         data_list.append({'category' : "SubQ", 'question' : temp['question'], 'a_entity' : temp['a_entity'], 'topic' : temp['topic'], 'cand_paths' : temp['cand_paths'], 'a_path' : [["None"]], 'answer' : "None"})
    # if len(paths) == 2:
    with open(paths[0], 'r', encoding='utf-8') as f:
            for line in f:
                temp = json.loads(line)
                data_list.append({'id' : temp['id'], 'question' : temp['question'], 'topic' : temp['topic'], 'a_entity' : temp['a_entity'], 'category' : temp['category'], 'neg_ent' : temp['neg_ent']})
                #data_list.append({'category' : temp['category'], 'question' : temp['question'], 'a_entity' : ["None"], 'topic' : ['None'], 'cand_paths' : temp['cand_path'], 'a_path' : [['None']], 'answer' : temp['answer']})
    return Dataset.from_list(data_list)



from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="fine-tuning/sft/BackPack", metadata={"help": "Path to the pretrained model"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default="fine-tuning/sft/BackPack", metadata={"help": "Path to the pretrained tokenizer"}
    )
    use_peft: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use PEFT (LoRA) for training adapters"},
    )
    lora_alpha: Optional[int] = field(
        default=128, metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "LoRA dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=64, metadata={"help": "LoRA rank parameter"}
    )
    modules_to_save: Optional[str] = field(
        default="o_proj,q_proj,up_proj,v_proj,k_proj,down_proj,gate_proj",
        metadata={"help": "Modules to apply LoRA on"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    

@dataclass
class DataTrainingArguments(DPOConfig):
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    optimize_cuda_cache: Optional[bool]= field(default=True)
    fp16: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use BFloat16 training"}
    )
    # deepspeed: Optional[str] = field(
    #     default="config/deepspeed_config.json",
    #     metadata={"help": "Path to DeepSpeed configuration file"}
    # )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "Log every N steps"}
    )
    # save_total_limit: Optional[int] = field(
    #     default=10, metadata={"help": "Maximum number of saved models"}
    # )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    n_cpu: Optional[int] = field(
        default=1, metadata={"help": "Number of CPU workers for data preprocessing"}
    )

    output_dir: Optional[str] = field(
        default="fine-tuning/dpo/saved_models", metadata={"help": "Path to save the fine-tuned model"}
    )
    dataset_dir: Optional[str] = field(
        default="/home/hyemin/model/my_model/data/webqsp",
        metadata={"help": "Directory containing the dataset"}
    )
    train_file: Optional[List[str]] = field(
        default_factory=lambda: [
            "/home/hyemin/model/my_model/data/webqsp/sft_train_path_subQ.jsonl",
            "/home/hyemin/model/my_model/data/cwq/sft_train_totalq1000.jsonl"
        ],
        metadata={"help": "List of training data files"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=3, metadata={"help": "Batch size per device during training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=3, metadata={"help": "Batch size per device during evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=3, metadata={"help": "Number of gradient accumulation steps"}
    )
    num_train_epochs: Optional[int] = field(
        default=10, metadata={"help": "Number of training epochs"}
    )
    validation_split_percentage: Optional[float] = field(
        default=5, metadata={"help": "Percentage of training set used as validation set"}
    )
    max_seq_length: Optional[int] = field(
        default=2048, metadata={"help": "Maximum sequence length for training"}
    )
    max_prompt_length: Optional[int] = field(
        default=2048, metadata={"help": "Maximum sequence length for training"}
    )
    max_target_length: Optional[int] = field(
        default=200, metadata={"help": "Maximum sequence length for training"}
    )



def input_formatter(samples):
    prompt_list = list()
    chosen_list = list()
    rejected_list = list()
      
    for i in range(len(samples["question"])):
        question = samples["question"][i]
        if not question.endswith("?"):
            question += "?"
        category = samples['category'][i]
        neg_entity = samples["neg_ent"][i]
        answer = samples["a_entity"][i]
        # paths = samples["cand_paths"][i]
        # cand_paths = [" -> ".join(sublist) for sublist in paths]
        # cand_paths = '\n'.join(cand_paths)
        
        # if samples["category"][i] == 'SubQ':
        #     prompt = SUBQ_ZERO_SHOT_PROMPT.format(question=question, paths=cand_paths)
        #     chosen = SUBQ_ANS_TEMPLATE.format(answer=samples["a_entity"][i]) #or samples["a_entity"][0]
        #     rejected = SUBQ_ANS_TEMPLATE.format(answer=list(set([p[-1] for p in paths]) - set(samples["a_entity"][i]))) 
        # else:
        #     prompt = TOTALQ_ZERO_SHOT_PROMPT.format(question=question, paths=cand_paths)
        #     chosen = TOTALQ_ANS_TEMPLATE.format(answer=samples["answer"][i])
        #     rejected = TOTALQ_ANS_TEMPLATE.format(answer="no" if samples["answer"][i] == "yes" else "yes")
        
        if category == 'ent':
            for iter in range(12):
                if iter == 0:
                    if len(answer) < 20:
                        fin_cand_ents = answer
                    else:
                        fin_cand_ents = random.sample(answer, k=20)
                    total_answer = fin_cand_ents
                elif iter == 1 and len(answer) > 2:
                    if len(answer) < 20:
                        sel_num = random.sample(range(1, len(answer)), k=1)[0]
                        fin_cand_ents = random.sample(answer, k=sel_num)
                    else:
                        fin_cand_ents = random.sample(answer, k=20)
                    total_answer = fin_cand_ents
                elif iter < 6:
                    max_len = 10 if len(answer) > 10 else len(answer)
                    sel_num = random.sample(range(1, max_len+1), k=1)[0]
                    pos_cand_ents = random.sample(answer, k=sel_num)

                    max_len = 10 if len(neg_entity) > 10 else len(neg_entity)
                    sel_num = random.sample(range(1, max_len+1), k=1)[0]
                    neg_cand_ents = random.sample(neg_entity, k=sel_num)
                    fin_cand_ents = pos_cand_ents + neg_cand_ents
                    total_answer = pos_cand_ents
                else:
                    max_len = 20 if len(neg_entity) > 20 else len(neg_entity)
                    sel_num = random.sample(range(1, max_len+1), k=1)[0]
                    fin_cand_ents = random.sample(neg_entity, k=sel_num)
                    total_answer = ["None"]

                fin_cand_ents.append("None")
                shuffle(fin_cand_ents)

                prompt = EPRUNING_ZERO_SHOT_PROMPT.format(question=question, entities=fin_cand_ents)
                chosen = TOTALQ_ANS_TEMPLATE.format(answer=total_answer)
                rejected = TOTALQ_ANS_TEMPLATE.format(answer= list(set(fin_cand_ents)-set(total_answer)))
                
                prompt_list.append(prompt)
                chosen_list.append(chosen)
                rejected_list.append(rejected)
        
    return {
        "prompt": prompt_list,
        "chosen": chosen_list,
        "rejected": rejected_list,
    }


logger = logging.getLogger(__name__)

def train():
    parser = HfArgumentParser((ScriptArguments, DataTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)
    
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    dataset = load_dataset(training_args.train_file)
    
    train_dataset = dataset.map(
        input_formatter,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,
    )
    
    logger.info(f"Num train_samples  {len(train_dataset)}")
    logger.info("Training example:")
    logger.info(train_dataset[0])
        
    if training_args.validation_split_percentage > 0:
        split_dataset = train_dataset.train_test_split(
        test_size=training_args.validation_split_percentage / 100, seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(eval_dataset[0])
    else:
        eval_dataset = None  
    
    
    torch_dtype = (
        script_args.torch_dtype
        if script_args.torch_dtype in ["auto", None]
        else getattr(torch, script_args.torch_dtype)
    )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))       
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained( # 인자값 수정하기
        "google/gemma-2-9b-it",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",
        use_auth_token=True,
        cache_dir='/home/huggingface',
        torch_dtype=torch_dtype #torch.float16  
        # low_cpu_mem_usage=True,
    )
    
    logger.info(f"quantization_config:{bnb_config.to_dict()}")
    
    #model = get_peft_model(model, peft_config)
    model = PeftModel.from_pretrained(base_model, script_args.model_name_or_path, is_trainable=True, adapter_name="DPO")
    #model = peft_model.merge_and_unload()
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

   
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, use_fast=False)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    #model.to(device)
    
    trainer = DPOTrainer(
        model=model,
        #ref_model=ref_model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=training_args.max_prompt_length,
        max_target_length=training_args.max_target_length,
        max_length=training_args.max_seq_length,
        peft_config=peft_config
    )

    trainer.train()
    torch.cuda.empty_cache()
    
    trainer.save_model(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    # trainer.model.save_pretrained(training_args.output_dir)

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    print(metrics)  
        
if __name__ == "__main__":
    train()