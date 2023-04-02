python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k-237/answers \
                            --prediction_path ../../processed_data/FB15k-237/google/flan-t5-large \
                            --log_score_path ../../processed_data/FB15k-237/scores/google/flan-t5-large

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k-237/answers \
                            --prediction_path ../../processed_data/FB15k-237/google/flan-t5-xl \
                            --log_score_path ../../processed_data/FB15k-237/scores/scores/google/flan-t5-xl

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k-237/answers/ \
                            --prediction_path ../../processed_data/FB15k-237/google/flan-t5-xxl \
                            --log_score_path ../../processed_data/FB15k-237/scores/google/flan-t5-xxl

python ../compute_scores.py --ground_truth_path ../../processed_data/NELL/answers/ \
                            --prediction_path ../../processed_data/NELL/google/flan-t5-large \
                            --log_score_path ../../processed_data/NELL/scores/google/flan-t5-large

python ../compute_scores.py --ground_truth_path ../../processed_data/NELL/answers/ \
                            --prediction_path ../../processed_data/NELL/google/flan-t5-xl \
                            --log_score_path ../../processed_data/NELL/scores/google/flan-t5-xl

python ../compute_scores.py --ground_truth_path ../../processed_data/NELL/answers \
                            --prediction_path ../../processed_data/NELL/google/flan-t5-xxl \
                            --log_score_path ../../processed_data/NELL/scores/google/flan-t5-xxl

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k/answers/ \
                            --prediction_path ../../processed_data/FB15k/google/flan-t5-large \
                            --log_score_path ../../processed_data/FB15k/scores/google/flan-t5-large

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k/answers/ \
                            --prediction_path ../../processed_data/FB15k/google/flan-t5-xl \
                            --log_score_path ../../processed_data/FB15k/scores/google/flan-t5-xl

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k/answers/ \
                            --prediction_path ../../processed_data/FB15k/google/flan-t5-xxl \
                            --log_score_path ../../processed_data/FB15k/scores/google/flan-t5-xxl
