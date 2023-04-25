python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k-237/answers \
                            --prediction_path ../../processed_data/FB15k-237/step_predictions/google/flan-t5-large \
                            --log_score_path ../../processed_data/FB15k-237/scores/step_predictions/google/flan-t5-large

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k-237/answers \
                            --prediction_path ../../processed_data/FB15k-237/step_predictions/google/flan-t5-xl \
                            --log_score_path ../../processed_data/FB15k-237/scores/scores/step_predictions/google/flan-t5-xl

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k-237/answers/ \
                            --prediction_path ../../processed_data/FB15k-237/step_predictions/google/flan-t5-xxl \
                            --log_score_path ../../processed_data/FB15k-237/scores/step_predictions/google/flan-t5-xxl

python ../compute_scores.py --ground_truth_path ../../processed_data/NELL/answers/ \
                            --prediction_path ../../processed_data/NELL/step_predictions/google/flan-t5-large \
                            --log_score_path ../../processed_data/NELL/scores/step_predictions/google/flan-t5-large

python ../compute_scores.py --ground_truth_path ../../processed_data/NELL/answers/ \
                            --prediction_path ../../processed_data/NELL/step_predictions/google/flan-t5-xl \
                            --log_score_path ../../processed_data/NELL/scores/step_predictions/google/flan-t5-xl

python ../compute_scores.py --ground_truth_path ../../processed_data/NELL/answers \
                            --prediction_path ../../processed_data/NELL/step_predictions/google/flan-t5-xxl \
                            --log_score_path ../../processed_data/NELL/scores/step_predictions/google/flan-t5-xxl

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k/answers/ \
                            --prediction_path ../../processed_data/FB15k/step_predictions/google/flan-t5-large \
                            --log_score_path ../../processed_data/FB15k/scores/step_predictions/google/flan-t5-large

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k/answers/ \
                            --prediction_path ../../processed_data/FB15k/step_predictions/google/flan-t5-xl \
                            --log_score_path ../../processed_data/FB15k/scores/step_predictions/google/flan-t5-xl

python ../compute_scores.py --ground_truth_path ../../processed_data/FB15k/answers/ \
                            --prediction_path ../../processed_data/FB15k/step_predictions/google/flan-t5-xxl \
                            --log_score_path ../../processed_data/FB15k/scores/step_predictions/google/flan-t5-xxl
