import os
import json
from tqdm import trange
from llm_engine import FlanLLMAnswer, LlamaLLMAnswer, FairLlamaLLMAnswer, AlpacaLlamaLLMAnswer, VicunaLLMAnswer
from gpt_engine import GPTAnswer
from global_config import QUERY_STRUCTS
import logging
import argparse
import deepspeed
from compute_scores import clean_string

logging.basicConfig(level=logging.INFO)
def main(processed_path, batch_size=10, 
         modelname="google/flan-t5-xxl", 
         ckpt_dir="", tokenizer_path="",
         lora_weights=""):
    if "flan" in modelname.lower():
        engine = FlanLLMAnswer(modelname)
        logging.info(f"Flan LLM: {modelname}")
    elif "gpt" in modelname.lower():
        engine = GPTAnswer(modelname)
        logging.info(f"GPT LLM: {modelname}")
    elif "fair-llama" in modelname.lower():
        engine = FairLlamaLLMAnswer(ckpt_dir,tokenizer_path,batch_size)
        logging.info(f"FAIR LLaMA LLM: {modelname}")
    elif "alpaca" in modelname.lower():
        engine = AlpacaLlamaLLMAnswer(modelname.split(":")[1], lora_weights)
        logging.info(f"Alpaca LLM: {modelname}")
    elif "vicuna" in modelname.lower():
        engine = VicunaLLMAnswer(modelname.split(":")[1])
        logging.info(f"Vicuna LLM: {modelname}")
    elif "llama" in modelname.lower():
        engine = LlamaLLMAnswer(modelname)
        logging.info(f"LLaMA LLM: {modelname}")
    for qtype, _ in QUERY_STRUCTS.items():
        logging.info(f"Generating predictions for query type {qtype}")
        idx  = 0
        question_path = os.path.join(f"{processed_path}","step_questions",f"{qtype}_{idx}_question.json")
        premise_questions = {}
        while os.path.exists(question_path):
            with open(question_path) as q_f:
                question = json.load(q_f)
            premise_questions[idx] = question
            idx  += 1
            question_path = os.path.join(f"{processed_path}","step_questions",f"{qtype}_{idx}_question.json")
        logging.info(f"Finished loading premise questions for query type {qtype}")
        li_premise_questions = list(premise_questions.items())
        predictions_path = os.path.join(f"{processed_path}","step_predictions",f"{modelname}")
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        for i in trange(len(premise_questions)//batch_size+1):
            pq_subset = dict(li_premise_questions[i*batch_size:(i+1)*batch_size])
            engine.log_step_answer(qtype, pq_subset, output_path=predictions_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', type=str, required=True, help="Path to processed files.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size of the model")
    parser.add_argument('--model_name', type=str, required=True, help="Model Id from Huggingface")
    parser.add_argument('--ckpt_dir', type=str, default="", help="Checkpoint dir for FAIR LLM")
    parser.add_argument('--tokenizer_path', type=str, default="", help="Tokenizer dir for FAIR LLM")
    parser.add_argument('--lora_weights',type=str, default="", help="Path to lora weights for Alpaca LLM")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args.processed_path, args.batch_size, 
         modelname=args.model_name, ckpt_dir=args.ckpt_dir,
         tokenizer_path=args.tokenizer_path,
         lora_weights=args.lora_weights)
    # deepspeed --include localhost:1,2,3 generate_llm_answers.py \
    #          --processed_path ../processed_data/FB15k-237 --batch_size 10 \
    #          --model_name google/flan-t5-xxl \
    #          --deepspeed --deepspeed_config ds_config.json
    # torchrun --nproc_per_node 1 --master_port=32384 generate_llm_answers.py \
    #          --processed_path ../processed_data/FB15k-237 \
    #          --batch_size 32 \
    #          --model_name fair/fair-llama-65b \
    #          --ckpt_dir ../../LLaMA/LLaMA/65B/ \
    #          --tokenizer_path ../../LLaMA/LLaMA/tokenizer.model

    