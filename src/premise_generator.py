import pickle as pkl
from collections import defaultdict
from global_config import QUERY_STRUCTS

class PremiseGenerator:
    def __init__(self, entities_path, relations_path, 
                entity_triplets_path,
                relation_triplets_path):
        self.premise_tag = "Given the following (h,r,t) triplets where entity h is related to entity t by relation r;\n"
        self.premise_end_tag = "\n"
        self.entities = pkl.load(open(entities_path,"rb"))
        self.relations = pkl.load(open(relations_path,"rb"))
        self.entity_triplets = pkl.load(open(entity_triplets_path,"rb"))
        self.relation_triplets = pkl.load(open(relation_triplets_path,"rb"))
        self.query_structs = QUERY_STRUCTS

    def get_premise(self, entity_set, relation_set):
        kg_triplets = defaultdict(set)
        for entity in entity_set:
            for triplet in self.entity_triplets[entity]:
                h, r, t = triplet
                kg_triplets[(h,r)].add(t)
        for relation in relation_set:
            for triplet in self.relation_triplets[relation]:
                h, r, t = triplet
                kg_triplets[(h,r)].add(t)
        return kg_triplets

    def filter_premise_1p(self, kg_triplets, entities, relations):
        e, r = entities[0], relations[0]
        tails = kg_triplets.get((e,r),set([]))
        filtered_set = set([])
        for tail in tails:
            filtered_set.add((e,r,tail))
        return filtered_set

    def filter_premise_2p(self, kg_triplets, entities, relations):
        e = entities[0]
        r1, r2 = relations
        entity_set = kg_triplets.get((e,r1),set([]))
        filtered_set = set([])
        for entity in entity_set:
            tails = kg_triplets[(entity,r2)]
            if len(tails) != 0:
                filtered_set.add((e,r1,entity))
            for tail in tails:
                filtered_set.add((entity,r2,tail))
        return filtered_set

    def filter_premise_3p(self, kg_triplets, entities, relations):
        e = entities[0]
        r1, r2, r3 = relations
        entity_set1 = kg_triplets[(e,r1)]
        filtered_set = set([])
        for entity1 in entity_set1:
            entity_set2 = kg_triplets[(entity1,r2)]
            for entity in entity_set2:
                tails = kg_triplets[(entity,r3)]
                if len(tails) != 0:
                    filtered_set.add((e, r1, entity1))
                    filtered_set.add((entity1, r2, entity))
                for tail in tails:
                    filtered_set.add((entity,r3,tail))
        return filtered_set
    
    def filter_premise_2i(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = kg_triplets[(e2,r2)]
        tails = entity_set1.intersection(entity_set2)
        for tail in tails:
            filtered_set.add((e1,r1,tail))
            filtered_set.add((e2,r2,tail))
        return filtered_set

    def filter_premise_3i(self, kg_triplets, entities, relations):
        e1, e2, e3 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = kg_triplets[(e2,r2)]
        entity_set3 = kg_triplets[(e3,r3)]
        tails = entity_set1.intersection(entity_set2).intersection(entity_set3)
        for tail in tails:
            filtered_set.add((e1,r1,tail))
            filtered_set.add((e2,r2,tail))
            filtered_set.add((e3,r3,tail))
        return filtered_set

    def filter_premise_2in(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = set([])
        neg_tail_map = {}
        for key in kg_triplets:
            if (key[0]==e2) and (key[1]!=r2):
                for tail in kg_triplets[key]: 
                    neg_tail_map[tail] = key[1]
                entity_set2 = entity_set2.union(kg_triplets[key])
        tails = entity_set1.intersection(entity_set2)
        for tail in tails:
            filtered_set.add((e1,r1,tail))
            filtered_set.add((e2,neg_tail_map[tail],tail))
        return filtered_set

    def filter_premise_3in(self, kg_triplets, entities, relations):
        e1, e2, e3 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = kg_triplets[(e2,r2)]
        entity_set3 = set([])
        neg_tail_map = {}
        for key in kg_triplets:
            if (key[0]==e3) and (key[1]!=r3):
                for tail in kg_triplets[key]: 
                    neg_tail_map[tail] = key[1]
                entity_set3 = entity_set3.union(kg_triplets[key])
        tails = entity_set1.intersection(entity_set2).intersection(entity_set3)
        for tail in tails:
            filtered_set.add((e1,r1,tail))
            filtered_set.add((e2,r2,tail))
            filtered_set.add((e3,neg_tail_map[tail],tail))
        return filtered_set

    def filter_premise_inp(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = set([])
        neg_tail_map = {}
        for key in kg_triplets:
            if (key[0]==e2) and (key[1]!=r2):
                for tail in kg_triplets[key]: 
                    neg_tail_map[tail] = key[1]
                entity_set2 = entity_set2.union(kg_triplets[key])
        entity_set3 = entity_set1.intersection(entity_set2)
        for entity in entity_set3:
            tails = kg_triplets[(entity,r3)]
            if len(tails) != 0:
                filtered_set.add((e1,r1,entity))
                filtered_set.add((e2,neg_tail_map[entity],entity))
            for tail in tails:
                filtered_set.add((entity,r3,tail))
        return filtered_set

    def filter_premise_pin(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = set([])
        entity_path_map = {}
        for entity in entity_set1:
            entity_set2 = entity_set2.union(kg_triplets[(entity,r2)])
            for triplet in kg_triplets[(entity, r2)]: 
                entity_path_map[triplet] = entity
        entity_set3 = set([])
        neg_tail_map = {}
        for key in kg_triplets:
            if (key[0]==e2) and (key[1]!=r3):
                for tail in kg_triplets[key]: 
                    neg_tail_map[tail] = key[1]
                entity_set3 = entity_set3.union(kg_triplets[key])
        tails = entity_set2.intersection(entity_set3)
        for tail in tails:
            filtered_set.add((e1,r1,entity_path_map[tail]))
            filtered_set.add((entity_path_map[tail],r2,tail))
            filtered_set.add((e2,neg_tail_map[tail],tail))
        return filtered_set
        
    def filter_premise_pni(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = set([])
        entity_path_map = {}
        neg_tail_map = {}
        for entity in entity_set1:
            for key in kg_triplets:
                if (key[0]==entity) and (key[1]!=r2):
                    for tail in kg_triplets[key]: 
                        neg_tail_map[tail] = key[1]
                        entity_path_map[tail] = entity
                    entity_set2 = entity_set2.union(kg_triplets[key])
                
        entity_set3 = kg_triplets[(e2,r3)]
        tails = entity_set2.intersection(entity_set3)
        for tail in tails:
            filtered_set.add((e1,r1,entity_path_map[tail]))
            filtered_set.add((entity_path_map[tail],neg_tail_map[tail],tail))
            filtered_set.add((e2,r3,tail))
        return filtered_set

    def filter_premise_ip(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_path_map = {}
        entity_set1 = kg_triplets[(e1,r1)]
        entity_set2 = kg_triplets[(e2,r2)]
        entity_set3 = entity_set1.intersection(entity_set2)
        tails = set([])
        for entity in entity_set3:
            tails = tails.union(kg_triplets[(entity,r3)])
            for tail in kg_triplets[(entity,r3)]:
                entity_path_map[tail] = entity
        for tail in tails:
            filtered_set.add((e1,r1,entity_path_map[tail]))
            filtered_set.add((e2,r2,entity_path_map[tail]))
            filtered_set.add((entity_path_map[tail],r3,tail))
        return filtered_set
    
    def filter_premise_pi(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_set2 = set([])
        entity_path_map = {}
        entity_set1 = kg_triplets[(e1,r1)]
        for entity in entity_set1:
            entity_set2 = entity_set2.union(kg_triplets[(entity,r2)])
            for es2 in entity_set2:
                entity_path_map[es2] = entity
        entity_set3 = kg_triplets[(e2,r3)]
        tails = entity_set2.intersection(entity_set3)
        for tail in tails:
            filtered_set.add((e1,r1,entity_path_map[tail]))
            filtered_set.add((entity_path_map[tail],r2,tail))
            filtered_set.add((e2,r3,tail))
        return filtered_set
    
    def filter_premise_2u(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        for es1 in entity_set1:
            filtered_set.add((e1,r1,es1))
        entity_set2 = kg_triplets[(e2,r2)]
        for es2 in entity_set2:
            filtered_set.add((e2,r2,es2))
        return filtered_set

    def filter_premise_up(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_path_map1, entity_path_map2 = {},{}
        entity_set1 = kg_triplets[(e1,r1)]
        for es1 in entity_set1:
            entity_path_map1[es1] = e1
        entity_set2 = kg_triplets[(e2,r2)]
        for es2 in entity_set2:
            entity_path_map2[es2] = e2
        entity_set3 = entity_set1.union(entity_set2)
        for entity in entity_set3:
            tails = kg_triplets[(entity,r3)]
        for tail in tails:
            if tail in entity_path_map1:
                filtered_set.add((e1,r1,entity_path_map1[tail]))
                filtered_set.add((entity_path_map1[tail],r3,tail))
            if tail in entity_path_map2:
                filtered_set.add((e2,r2,entity_path_map2[tail]))
                filtered_set.add((entity_path_map2[tail],r3,tail))
        return filtered_set

    def filter_premise_nin(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2 = relations
        filtered_set = set([])
        entity_set1 = kg_triplets[(e1,r1)]
        for es1 in entity_set1:
            filtered_set.add((e1,r1,es1))
        entity_set2 = kg_triplets[(e2,r2)]
        for es2 in entity_set2:
            filtered_set.add((e2,r2,es2))
        return filtered_set
    

    def filter_premise_nipn(self, kg_triplets, entities, relations):
        e1, e2 = entities
        r1, r2, r3 = relations
        filtered_set = set([])
        entity_path_map1, entity_path_map2 = {},{}
        entity_set1 = kg_triplets[(e1,r1)]
        for es1 in entity_set1:
            entity_path_map1[es1] = e1
        entity_set2 = kg_triplets[(e2,r2)]
        for es2 in entity_set2:
            entity_path_map2[es2] = e2
        entity_set3 = entity_set1.union(entity_set2)
        for entity in entity_set3:
            tails = kg_triplets[(entity,r3)]
        for tail in tails:
            if tail in entity_path_map1:
                filtered_set.add((e1,r1,entity_path_map1[tail]))
                filtered_set.add((entity_path_map1[tail],r3,tail))
            if tail in entity_path_map2:
                filtered_set.add((e2,r2,entity_path_map2[tail]))
                filtered_set.add((entity_path_map2[tail],r3,tail))
        return filtered_set
        
    def filter_premise(self, kg_triplets, entities, relations, query_type):
        assert (query_type in self.query_structs),f"Only the following {list(self.query_structs.keys())} are supported."
        if query_type=="1p": return self.filter_premise_1p(kg_triplets, entities, relations)
        if query_type=="2p": return self.filter_premise_2p(kg_triplets, entities, relations)
        if query_type=="3p": return self.filter_premise_3p(kg_triplets, entities, relations)
        if query_type=="2i": return self.filter_premise_2i(kg_triplets, entities, relations)
        if query_type=="3i": return self.filter_premise_3i(kg_triplets, entities, relations)
        if query_type=="2in": return self.filter_premise_2in(kg_triplets, entities, relations)
        if query_type=="3in": return self.filter_premise_3in(kg_triplets, entities, relations)
        if query_type=="inp": return self.filter_premise_inp(kg_triplets, entities, relations)
        if query_type=="pin": return self.filter_premise_pin(kg_triplets, entities, relations)
        if query_type=="pni": return self.filter_premise_pni(kg_triplets, entities, relations)
        if query_type=="ip": return self.filter_premise_ip(kg_triplets, entities, relations)
        if query_type=="pi": return self.filter_premise_pi(kg_triplets, entities, relations)
        if query_type=="2u": return self.filter_premise_2u(kg_triplets, entities, relations)
        if query_type=="up": return self.filter_premise_up(kg_triplets, entities, relations)
        if query_type=="nin": return self.filter_premise_nin(kg_triplets, entities, relations)
        if query_type=="nipn": return self.filter_premise_nipn(kg_triplets, entities, relations)

    def generate_premise(self, entity_set, relation_set, query_type):
        kg_triplets = self.get_premise(entity_set, relation_set)
        filtered_set = self.filter_premise(kg_triplets,entity_set,relation_set, query_type)
        texts = []
        # for relation, pairs in agg_triplets.items():
        #     texts.append(f'The entity pairs {",".join(pairs)} are connected by relation {relation}.')
        for triplet in filtered_set:
            texts.append(str((triplet)).strip().replace(" ",""))
        return self.premise_tag+",".join(texts)+"\n"+self.premise_end_tag
