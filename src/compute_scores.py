import os
import re
import sys
from tqdm import tqdm
from global_config import QUERY_STRUCTS
import numpy as np
import argparse

def clean_string(string):
    clean_str = re.sub(r"[^0-9,]","",string)
    return clean_str

import numpy as np

def compute_mrr_score(ground_truth, predictions):
    if len(ground_truth) == len(predictions) == 0:
        return 1
    reciprocal_ranks = []
    for i, prediction in enumerate(predictions):
        if prediction in ground_truth:
            reciprocal_rank = 1 / (i + 1)
            reciprocal_ranks.append(reciprocal_rank)
    if len(reciprocal_ranks) == 0: return 0
    mrr = sum(reciprocal_ranks)/len(reciprocal_ranks)
    return mrr


def compute_ndcg_score(ground_truth, predictions, k=5):
    if len(ground_truth) == len(predictions) == 0:
        return 1
    relevance_scores = []
    length = min(len(ground_truth),len(predictions))
    k = min(length,k)
    ground_truth = ground_truth[:k]
    predictions = predictions[:k]
    for i in range(k):
        prediction = predictions[i]
        relevance_score = 1 if prediction in ground_truth else 0
        relevance_scores.append(relevance_score)
    dcg_k = np.sum(relevance_scores / np.log2(np.arange(2, k+2)))
    sorted_ground_truth = sorted(ground_truth, reverse=True)
    idcg_k = np.sum([1 if sample in ground_truth else 0 for sample in sorted_ground_truth[:k]] / np.log2(np.arange(2, k+2)))
    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0
    return ndcg_k

def compute_hits_score(ground_truth, predictions, k=1):
    if len(ground_truth) == len(predictions) == 0:
        return 1
    hits = len(set(predictions[:k]).intersection(set(ground_truth)))
    l = len(predictions[:k])
    if l == 0: l = 1
    return hits/l

def main(ground_truth_path, prediction_path, log_score_path):
    log_score_filename = os.path.join(f"{log_score_path}","score.txt")
    if os.path.exists(log_score_filename):
        os.remove(log_score_filename)
    for qtype, _ in QUERY_STRUCTS.items():
        idx = 0 
        gt_filename = os.path.join(f"{ground_truth_path}",f"{qtype}_{idx}_answer.txt")
        pred_filename = os.path.join(f"{prediction_path}",f"{qtype}_{idx}_predicted_answer.txt")
        if not os.path.exists(f"{log_score_path}"):
            os.makedirs(f"{log_score_path}")
        scores = {"hits@1":0,"hits@3":0,"hits@10":0,
                  "ndcg@1":0,"ndcg@3":0,"ndcg@10":0,
                  "mrr":0}
        pbar = tqdm(total=None)
        while os.path.exists(gt_filename) and os.path.exists(pred_filename):
            with open(gt_filename) as gt_f:
                cleaned_gt = clean_string(gt_f.read()).split(",")
                gt = [int(x) for x in cleaned_gt if x.isdigit()]

            with open(pred_filename) as pred_f:
                cleaned_pred = clean_string(pred_f.read()).split(",")
                pred = [int(x) for x in cleaned_pred if x.isdigit()]
            gt = list(dict.fromkeys(gt))
            pred = list(dict.fromkeys(pred))
            scores["hits@1"] += compute_hits_score(gt, pred, k=1)
            scores["hits@3"] += compute_hits_score(gt, pred, k=3)
            scores["hits@10"] += compute_hits_score(gt, pred, k=10)
            scores["ndcg@1"] += compute_ndcg_score(gt, pred, k=1)
            scores["ndcg@3"] += compute_ndcg_score(gt, pred, k=3)
            scores["ndcg@10"] += compute_ndcg_score(gt, pred, k=10)
            scores["mrr"] += compute_mrr_score(gt, pred)
            idx += 1
            gt_filename = os.path.join(f"{ground_truth_path}",f"{qtype}_{idx}_answer.txt")
            pred_filename = os.path.join(f"{prediction_path}",f"{qtype}_{idx}_predicted_answer.txt")
            pbar.update(1)
        pbar.close()
        with open(log_score_filename, "a") as score_file:
            print(qtype, file=score_file)
            print("HITS@1:",scores["hits@1"]/(idx-1), file=score_file)
            print("HITS@3:",scores["hits@3"]/(idx-1), file=score_file)
            print("HITS@10:",scores["hits@10"]/(idx-1), file=score_file)
            print("NDCG@1:",scores["ndcg@1"]/(idx-1), file=score_file)
            print("NDCG@3:",scores["ndcg@3"]/(idx-1), file=score_file)
            print("NDCG@10:",scores["ndcg@10"]/(idx-1), file=score_file)
            print("MRR:",scores["mrr"]/(idx-1), file=score_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_path', type=str, required=True, help="Path to ground truth data.")
    parser.add_argument('--prediction_path', type=str, required=True, help="Path to the prediction files.")
    parser.add_argument('--log_score_path', type=str, required=True, help="Path to log scores")
    args = parser.parse_args()
    main(args.ground_truth_path, args.prediction_path, args.log_score_path)


            
