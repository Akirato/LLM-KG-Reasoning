deepspeed --include localhost:0 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k-237 --batch_size 10 \
             --model_name decapoda-research/llama-7b-hf \
             --lora_weights tloen/alpaca-lora-7b \
             --deepspeed --deepspeed_config ds_config.json
