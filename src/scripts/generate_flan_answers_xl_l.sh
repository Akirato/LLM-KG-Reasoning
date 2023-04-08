deepspeed --include localhost:4,5,6,7 --master_port 61000 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k-237 --batch_size 14     \
             --model_name google/flan-t5-xl \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:0,1 --master_port 61001 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k-237 --batch_size 84     \
             --model_name google/flan-t5-large \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:2,3 --master_port 61002 ../generate_llm_answers.py \
             --processed_path ../../processed_data/NELL --batch_size 84     \
             --model_name google/flan-t5-large \
             --deepspeed --deepspeed_config ds_config.json && \
deepspeed --include localhost:0,1,2,3 --master_port 61003 ../generate_llm_answers.py \
             --processed_path ../../processed_data/NELL --batch_size 14     \
             --model_name google/flan-t5-xl \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:4,5,6,7 --master_port 61004 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k --batch_size 14     \
             --model_name google/flan-t5-xl \
             --deepspeed --deepspeed_config ds_config.json && \
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 61005 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k --batch_size 84     \
             --model_name google/flan-t5-large \
             --deepspeed --deepspeed_config ds_config.json 
