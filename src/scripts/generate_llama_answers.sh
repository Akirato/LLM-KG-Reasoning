CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=32384 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k-237 \
         --batch_size 32 \
         --model_name fair/fair-llama-30b \
         --ckpt_dir ../../../LLaMA/LLaMA/30B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model & \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=32385 ../generate_llm_answers.py \
         --processed_path ../../processed_data/NELL \
         --batch_size 32 \
         --model_name fair/fair-llama-30b \
         --ckpt_dir ../../../LLaMA/LLaMA/30B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model

torchrun --nproc_per_node 8 --master_port=32384 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k-237 \
         --batch_size 32 \
         --model_name fair/fair-llama-65b \
         --ckpt_dir ../../../LLaMA/LLaMA/65B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model

torchrun --nproc_per_node 8 --master_port=32384 ../generate_llm_answers.py \
         --processed_path ../../processed_data/NELL \
         --batch_size 32 \
         --model_name fair/fair-llama-65b \
         --ckpt_dir ../../../LLaMA/LLaMA/65B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model

torchrun --nproc_per_node 8 --master_port=32384 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k \
         --batch_size 32 \
         --model_name fair/fair-llama-65b \
         --ckpt_dir ../../../LLaMA/LLaMA/65B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=32384 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k \
         --batch_size 32 \
         --model_name fair/fair-llama-30b \
         --ckpt_dir ../../../LLaMA/LLaMA/30B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model & \
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 --master_port=32385 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k-237 \
         --batch_size 32 \
         --model_name fair/fair-llama-13b \
         --ckpt_dir ../../../LLaMA/LLaMA/13B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model & \
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port=32386 ../generate_llm_answers.py \
         --processed_path ../../processed_data/NELL \
         --batch_size 32 \
         --model_name fair/fair-llama-13b \
         --ckpt_dir ../../../LLaMA/LLaMA/13B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port=32384 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k \
         --batch_size 32 \
         --model_name fair/fair-llama-13b \
         --ckpt_dir ../../../LLaMA/LLaMA/13B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model & \
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port=32385 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k-237 \
         --batch_size 32 \
         --model_name fair/fair-llama-7b \
         --ckpt_dir ../../../LLaMA/LLaMA/7B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model & \
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port=32386 ../generate_llm_answers.py \
         --processed_path ../../processed_data/NELL \
         --batch_size 32 \
         --model_name fair/fair-llama-7b \
         --ckpt_dir ../../../LLaMA/LLaMA/7B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model & \
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master_port=32387 ../generate_llm_answers.py \
         --processed_path ../../processed_data/FB15k \
         --batch_size 32 \
         --model_name fair/fair-llama-7b \
         --ckpt_dir ../../../LLaMA/LLaMA/7B/ \
         --tokenizer_path ../../../LLaMA/LLaMA/tokenizer.model

