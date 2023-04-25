import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
import deepspeed
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from peft import PeftModel
import os
import sys
import time
import json
from compute_scores import clean_string

from pathlib import Path
class BaseLLMAnswer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_source_length = 512
        self.max_new_tokens = 64
        self.tokenizer = None
        self.model = None

    def process_input(self, premise_question):
        input_ids = self.tokenizer(premise_question, 
                        padding="longest",
                        max_length=self.max_source_length,
                        truncation=True,
                        return_tensors="pt").input_ids.to(self.device)
        return input_ids

    def generate_answer(self, premise_questions):
        if len(premise_questions)<1: return []
        input_ids = self.process_input(premise_questions)
        outputs = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
        
    def log_answer(self, qtype, premise_questions={}, output_path=""):
        question_ids = list(premise_questions.keys())
        premise_questions = list(premise_questions.values())
        predicted_answers = self.generate_answer(premise_questions)
        for idx, prediction in enumerate(predicted_answers):
            with open(os.path.join(f"{output_path}",f"{qtype}_{question_ids[idx]}_predicted_answer.txt"),"w") as prediction_file:
                print(prediction, file=prediction_file)

    def process_step_question(self, qtype, premise_questions):
        if len(premise_questions)<1: return None
        explain_tag = premise_questions[0]["question"]["explain_tag"]
        question_tag = premise_questions[0]["question"]["question_tag"]
        step_questions = {}
        for premise_question in premise_questions:
            questions = premise_question["question"]["question"][qtype]
            premise = premise_question["premise"]
            phase = 1
            for question in questions:
                enhanced_premise_question = "".join([premise,question_tag,question,explain_tag])
                if phase in step_questions: step_questions[phase].append(enhanced_premise_question)
                else: step_questions[phase] = [enhanced_premise_question]
                phase += 1
        return step_questions
    
    def swap_question_placeholders(self, questions, step_answers):
        question_with_swaps = []
        for i in range(len(questions)):
            question = questions[i]
            if "[PP1]" in question:
                sai = clean_string(step_answers[1][i])
                question = question.replace("[PP1]", "{"+sai+"}")
            if "[PP2]" in question:
                sai = clean_string(step_answers[2][i])
                question = question.replace("[PP2]", "{"+sai+"}")    
            if "[PP3]" in question:
                sai = clean_string(step_answers[3][i])
                question = question.replace("[PP3]", "{"+sai+"}")
            question_with_swaps.append(question)
        return question_with_swaps

    def generate_step_answer(self, qtype, premise_questions):
        step_questions = self.process_step_question(qtype, premise_questions)
        if step_questions == None: return []
        step_answers = {}
        final_phase = 1
        for phase in range(1,len(step_questions)+1):
            if phase == 1:
                step_answers[phase] = self.generate_answer(step_questions[phase])
            else:
                question_with_swaps = self.swap_question_placeholders(step_questions[phase], step_answers)
                step_answers[phase] = self.generate_answer(question_with_swaps)
            final_phase = phase
        return step_answers[final_phase]

    def log_step_answer(self, qtype, premise_questions={}, output_path=""):
        question_ids = list(premise_questions.keys())
        premise_questions = list(premise_questions.values())
        predicted_answers = self.generate_step_answer(qtype, premise_questions)
        for idx, prediction in enumerate(predicted_answers):
            with open(os.path.join(f"{output_path}",f"{qtype}_{question_ids[idx]}_predicted_answer.txt"),"w") as prediction_file:
                print(prediction, file=prediction_file)

class FlanLLMAnswer(BaseLLMAnswer):
    def __init__(self, modelname="google/flan-t5-xxl"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(modelname)
        self.model = T5ForConditionalGeneration.from_pretrained(modelname,
                                                           load_in_8bit=True, 
                                                           device_map="auto")
        # self.model = deepspeed.init_inference(model,
        #                          dtype=torch.int8,
        #                          checkpoint=None,
        #                          replace_with_kernel_inject=True)    

class LlamaLLMAnswer(BaseLLMAnswer):
    def __init__(self, modelname="decapoda-research/llama-7b-hf"):
        super().__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(modelname)
        self.tokenizer.pad_token='[PAD]'
        model = LlamaForCausalLM.from_pretrained(modelname)
        self.model = deepspeed.init_inference(model,
                                 dtype=torch.int8,
                                 checkpoint=None,
                                 replace_with_kernel_inject=True)  
        
class FairLlamaLLMAnswer(BaseLLMAnswer):
    def __init__(self, ckpt_dir, tokenizer_path, max_batch_size):
        super().__init__()
        local_rank, world_size = self.setup_model_parallel()
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")
        self.max_source_length = 1024
        self.max_new_tokens = 100
        self.temperature = 0.2
        self.top_p = 0.95
        self.model = self.load(ckpt_dir, tokenizer_path, 
                               local_rank, world_size, 
                               self.max_source_length, 
                               max_batch_size)
        
    def setup_model_parallel(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

        torch.manual_seed(1)
        return local_rank, world_size

    def load(self,
            ckpt_dir,
            tokenizer_path,
            local_rank,
            world_size,
            max_seq_len,
            max_batch_size):
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        generator = LLaMA(model, tokenizer)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator

    def generate_answer(self, premise_question):
        if len(premise_question)<1: return []
        outputs = self.model.generate(premise_question, max_gen_len=self.max_new_tokens,
                                      temperature=self.temperature, top_p=self.top_p)
        return outputs
    
class AlpacaLlamaLLMAnswer(BaseLLMAnswer):
    def __init__(self, base_model="decapoda-research/llama-7b-hf", 
                  lora_weights="tloen/alpaca-lora-7b"):
        super().__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token='[PAD]'
        model = LlamaForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        self.model = deepspeed.init_inference(model,
                                 dtype=torch.int8,
                                 checkpoint=None,
                                 replace_with_kernel_inject=True)  
        self.generation_config = GenerationConfig(
            temperature =0.2,
            top_p = 0.95,
            top_k = 40,
            num_beams = 1
        )
    
    def generate_answer(self, premise_question):
        if len(premise_question)<1: return []
        input_ids = self.process_input(premise_question)
        outputs = self.model.generate(input_ids=input_ids,
                                    generation_config=self.generation_config, 
                                    max_new_tokens=self.max_new_tokens)
        return self.tokenizer.batch_decode(outputs[input_ids.shape[0]:], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
        
class VicunaLLMAnswer(BaseLLMAnswer):
    def __init__(self, modelname=""):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(modelname, 
                                                       #repo_type="model", 
                                                       use_fast=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(modelname, #repo_type="model", 
                                                     trust_remote_code=True,
                                                     load_in_8bit=True, 
                                                     device_map="auto")
        self.generation_config = GenerationConfig(
            temperature =0.2,
            top_p = 0.95,
            top_k = 40,
            num_beams = 1
        )
        # self.model = deepspeed.init_inference(model,
        #                          dtype=torch.int8,
        #                          checkpoint=None,
        #                          replace_with_kernel_inject=True)  
    
    def generate_answer(self, premise_question):
        if len(premise_question)<1: return []
        input_ids = self.process_input(premise_question)
        outputs = self.model.generate(input_ids=input_ids,
                                    generation_config=self.generation_config, 
                                    max_new_tokens=self.max_new_tokens)
        return self.tokenizer.batch_decode(outputs[input_ids.shape[0]:], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)