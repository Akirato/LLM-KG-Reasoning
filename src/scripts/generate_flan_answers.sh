deepspeed --include localhost:0,1,2,3 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k-237 --batch_size 10 \
             --model_name google/flan-t5-xxl \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:4,5,6,7 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k-237 --batch_size 10 \
             --model_name google/flan-t5-xl \
             --deepspeed --deepspeed_config ds_config.json

deepspeed --include localhost:0,1,2,3 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k-237 --batch_size 10 \
             --model_name google/flan-t5-large \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:4,5,6,7 ../generate_llm_answers.py \
             --processed_path ../../processed_data/NELL --batch_size 10 \
             --model_name google/flan-t5-xxl \
             --deepspeed --deepspeed_config ds_config.json

deepspeed --include localhost:0,1,2,3 ../generate_llm_answers.py \
             --processed_path ../../processed_data/NELL --batch_size 10 \
             --model_name google/flan-t5-xl \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:4,5,6,7 ../generate_llm_answers.py \
             --processed_path ../../processed_data/NELL --batch_size 10 \
             --model_name google/flan-t5-large \
             --deepspeed --deepspeed_config ds_config.json

deepspeed --include localhost:0,1,2,3 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k --batch_size 10 \
             --model_name google/flan-t5-xxl \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:4,5,6 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k --batch_size 10 \
             --model_name google/flan-t5-xl \
             --deepspeed --deepspeed_config ds_config.json & \
deepspeed --include localhost:7 ../generate_llm_answers.py \
             --processed_path ../../processed_data/FB15k --batch_size 10 \
             --model_name google/flan-t5-large \
             --deepspeed --deepspeed_config ds_config.json 