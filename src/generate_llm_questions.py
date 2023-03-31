import sys
import os
import csv
from tqdm import tqdm
import logging
import pickle as pkl
import argparse
from premise_generator import PremiseGenerator
from prompt_generator import LogicalPromptGenerator
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)

def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = value.union(dict_1[key])
   return dict_3

def process_logical_queries(logical_query, 
                            qtype, idx, output_path):
    e1, r1, e2, r2, e3, r3 = logic_processor.parse_logical_query(logical_query, qtype)
    entity_set = list(filter(lambda x: x!=None, [e1, e2, e3]))
    relation_set = list(filter(lambda x: x!=None, [r1, r2, r3]))
    answer = answers[logical_query]
    premise = premise_generator.generate_premise(entity_set, relation_set, qtype)
    question = logic_processor.generate_prompt(logical_query, qtype)
    premise_question = "\n".join([premise,question])
    answer_text = ", ".join(map(str,answer))
    question_file_path = os.path.join(f"{output_path}","questions",f"{qtype}_{idx}_question.txt")
    answer_file_path = os.path.join(f"{output_path}","answers",f"{qtype}_{idx}_answer.txt")
    with open(question_file_path,"w") as q_f:
        print(premise_question,file=q_f)
    with open(answer_file_path,"w") as a_f:
        print(answer_text,file=a_f)

#Premise Generation
def main(data_path, output_path):
    logging.info("Loading stats as sanity check")
    with open(os.path.join(f"{data_path}","stats.txt"), 'r') as f:
        print(f.read())
    
    logging.info("Loading path of entities and relations")
    entities_path = os.path.join(f"{data_path}","id2ent.pkl")
    relations_path = os.path.join(f"{data_path}","id2rel.pkl")
    
    logging.info("Merging train, test and valid query dictionaries")
    train_queries_path = os.path.join(f"{data_path}","train-queries.pkl")
    test_queries_path = os.path.join(f"{data_path}","test-queries.pkl")
    valid_queries_path = os.path.join(f"{data_path}","valid-queries.pkl") 
    with open(train_queries_path,"rb") as trainq_file:
        train_queries = pkl.load(trainq_file)
    with open(test_queries_path,"rb") as testq_file:
        test_queries = pkl.load(testq_file)
    with open(valid_queries_path,"rb") as validq_file:
        valid_queries = pkl.load(validq_file)
    temp_queries = mergeDictionary(train_queries, test_queries)
    queries = mergeDictionary(temp_queries,valid_queries)
    del(train_queries, test_queries, valid_queries, temp_queries)

    global answers
    logging.info("Merging train, test and valid answer dictionaries")
    train_answers_path = os.path.join(f"{data_path}","train-answers.pkl")
    test_easy_answers_path = os.path.join(f"{data_path}","test-easy-answers.pkl")
    valid_easy_answers_path = os.path.join(f"{data_path}","valid-easy-answers.pkl")
    test_hard_answers_path = os.path.join(f"{data_path}","test-hard-answers.pkl")
    valid_hard_answers_path = os.path.join(f"{data_path}","valid-hard-answers.pkl")
    with open(train_answers_path,"rb") as traina_file:
        train_a = pkl.load(traina_file)
    with open(test_easy_answers_path,"rb") as testea_file:
        test_easy_a = pkl.load(testea_file)
    with open(valid_easy_answers_path,"rb") as validea_file:
        valid_easy_a = pkl.load(validea_file)
    with open(test_hard_answers_path,"rb") as testha_file:
        test_hard_a = pkl.load(testha_file)
    with open(valid_hard_answers_path,"rb") as validha_file:
        valid_hard_a = pkl.load(validha_file)
    test_answers = mergeDictionary(test_easy_a, test_hard_a)
    valid_answers = mergeDictionary(valid_easy_a, valid_hard_a)
    temp_answers = mergeDictionary(train_a,valid_answers)
    answers = mergeDictionary(temp_answers, test_answers)
    del(train_a, test_easy_a, valid_easy_a, test_hard_a, valid_hard_a, test_answers, valid_answers, temp_answers)

    logging.info("Loading KG triplets")
    entity_triplets, relation_triplets = {}, {}
    triplet_files = [os.path.join(f"{data_path}","train.txt"), 
                     os.path.join(f"{data_path}","valid.txt"), 
                     os.path.join(f"{data_path}","test.txt")]
    for triplet_file in triplet_files:
        with open(triplet_file,"r") as kg_data_file:
            kg_tsv_file = csv.reader(kg_data_file, delimiter="\t")
            for line in kg_tsv_file:
                e1, r, e2 = map(int,line)
                triplet = (e1, r, e2)
                if e1 in entity_triplets: entity_triplets[e1].add(triplet)
                else: entity_triplets[e1] = set([triplet])
                if e2 in entity_triplets: entity_triplets[e2].add(triplet)
                else: entity_triplets[e2] = set([triplet])
                if r in relation_triplets: relation_triplets[r].add(triplet)
                else: relation_triplets[r] = set([triplet])
    if not os.path.exists(f"{output_path}"):
        os.makedirs(f"{output_path}")
    with open(os.path.join(f"{output_path}","entity_triplets.pkl"),"wb") as entity_triplets_file:
        pkl.dump(entity_triplets, entity_triplets_file)
    with open(os.path.join(f"{output_path}","relation_triplets.pkl"),"wb") as relation_triplets_file:
        pkl.dump(relation_triplets, relation_triplets_file)

    global logic_processor
    global premise_generator
    logic_processor = LogicalPromptGenerator(entities_path=entities_path,relations_path=relations_path)
    premise_generator = PremiseGenerator(entities_path=entities_path,
                                        relations_path=relations_path,
                                        entity_triplets_path=os.path.join(f"{output_path}","entity_triplets.pkl"),
                                        relation_triplets_path=os.path.join(f"{output_path}","relation_triplets.pkl"))

    logging.info("Generating premises and questions")
    question_path = os.path.join(f"{output_path}","questions")
    answer_path = os.path.join(f"{output_path}","answers")
    if not os.path.exists(question_path):
        os.makedirs(question_path)
    if not os.path.exists(answer_path):
        os.makedirs(answer_path)
    for qtype, qpattern in logic_processor.query_structs.items():
        logging.info(f"Generating question and answers for {qtype} queries in {output_path}")
        logical_queries = queries[qpattern]
        llq = len(logical_queries)
        args = zip(logical_queries,
                   [qtype]*llq, range(llq),
                   [output_path]*llq)
        with mp.Pool(mp.cpu_count()) as pool:
            pool.starmap(process_logical_queries, tqdm(args, total=llq))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to raw data.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output the processed files.")
    args = parser.parse_args()
    main(args.data_path, args.output_path)