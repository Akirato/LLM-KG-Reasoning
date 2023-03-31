import pickle as pkl
from global_config import QUERY_STRUCTS

class LogicalPromptGenerator:
    def __init__(self, entities_path, relations_path):
        self.question_tag = "Answer the question:\n"
        self.explain_tag = "\nReturn only the answer entities separated by commas with no other text."
        self.query_structs = QUERY_STRUCTS
        with open(entities_path,"rb") as entity_file:
            self.entities = pkl.load(entity_file)
        with open(relations_path,"rb") as relation_file:
            self.relations = pkl.load(open(relations_path,"rb"))
        self.reverse_query_structs = {v: k for k, v in self.query_structs.items()}

    def generate_premise(self, relation_triplets):
        triplets = []
        for relation_triplet in relation_triplets:
            head, relation, tail = relation_triplet
            triplets.append(f"""Entity "{head}" is related to entity "{tail}" 
                                by the relation "{relation}."\n""")
        return "".join(triplets)

    def parse_logical_query(self, logical_query, query_type):
        assert (query_type in self.query_structs),f"Only the following {list(self.query_structs.keys())} are supported."
        e1 = r1 = e2 = r2= e3 = r3 = None
        if query_type=="1p": (e1, (r1,)) = logical_query
        if query_type=="2p": (e1, (r1, r2)) = logical_query
        if query_type=="3p": (e1, (r1, r2, r3)) = logical_query
        if query_type=="2i": ((e1, (r1,)), (e2, (r2,))) = logical_query
        if query_type=="3i": ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = logical_query
        if query_type=="2in": ((e1, (r1,)), (e2, (r2, n))) = logical_query
        if query_type=="3in": ((e1, (r1,)), (e2, (r2,)), (e3, (r3, n))) = logical_query
        if query_type=="inp": (((e1, (r1,)), (e2, (r2, n))), (r3,)) = logical_query
        if query_type=="pin": ((e1, (r1, r2)), (e2, (r3, n))) = logical_query
        if query_type=="pni": ((e1, (r1, r2, n)), (e2, (r3,))) = logical_query
        if query_type=="ip": (((e1, (r1,)), (e2, (r2,))), (r3,)) = logical_query
        if query_type=="pi": ((e1, (r1, r2)), (e2, (r3,))) = logical_query
        if query_type=="2u": ((e1, (r1,)), (e2, (r2,)), (u,)) = logical_query
        if query_type=="up": (((e1, (r1,)), (e2, (r2,)), (u,)), (r3,)) = logical_query
        if query_type=="nin": (((e1, (r1, n)), (e2, (r2, n))), (n,)) = logical_query
        if query_type=="nipn": (((e1, (r1, n)), (e2, (r2, n))), (n, r3)) = logical_query
        return [e1, r1, e2, r2, e3, r3]
        
    def generate_question_1p(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="1p")
        entity = e1
        relation = r1
        return f"Which entities are connected to {entity} by relation {relation}?"

    def generate_question_2p(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="2p")
        entity = e1
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity} by relation {relation1}. \
                Then, what are the entities connected to E by relation {relation2}? \
                """

    def generate_question_3p(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="3p")
        entity = e1
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity} by relation {relation1} and the set of entities F \
                is connected to entities in E by relation {relation2}. \
                Then, what are the entities connected to F by relation {relation3}? \
                """

    def generate_question_2i(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="2i")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1} and the set of entities F \
                is connected to entity {entity2} by relation {relation2}. \
                Then, what are the entities in the intersection of set E and F, i.e., \
                entities present in both F and G? \
                """
    
    def generate_question_3i(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="3i")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1}, the set of entities F \
                is connected to entity {entity2} by relation {relation2} and the \
                set of entities G is connected to entity {entity3} by relation {relation3}. \
                Then, what are the entities in the intersection of set E, F and G, i.e., \
                entities present in all E, F and G?"""

    def generate_question_2in(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="2in")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1} and F is the set of entities \
                connected to entity {entity2} by any relation other than relation {relation2}. \
                Then, what are the entities in the intersection of set E and F, i.e., \
                entities present in both F and G?"""

    def generate_question_3in(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="3in")
        entity1 = e1
        entity2 = e2
        entity3 = e3
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1}, F is the set of entities \
                connected to entity {entity2} by relation {relation2}, and F is the \
                set of entities connected to entity {entity3} by any relation other \
                than relation {relation3}. \
                Then, what are the entities in the intersection of set E and F, i.e., \
                entities present in both F and G? \
                """
    
    def generate_question_inp(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="inp")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1}, and F is the set of entities \
                connected to entity {entity2} by any relation other than relation {relation2}. \
                Then, what are the entities that are connected to the entities in the \
                intersection of set E and F by relation {relation3}? \
                """    

    def generate_question_pin(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="pin")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1}, F is the set of entities \
                connected to entities in E by relation {relation2}, and G is the \
                set of entities connected to entity {entity2} by any relation other \
                than relation {relation3}. \
                Then, what are the entities in the intersection of set F and G, i.e., \
                entities present in both F and G? \
                """    

    def generate_question_pni(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="pni")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1}, F is the set of entities \
                connected to entities in E by any relation other than {relation2}, \
                and G is the set of entities connected to entity {entity2} by relation {relation3}. \
                Then, what are the entities in the intersection of set F and G, i.e., \
                entities present in both F and G? \
                """   

    def generate_question_ip(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="ip")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1}, F is the set of entities \
                connected to entity {entity2} by relation {relation2}, and G is the \
                set of entities in the intersection of E and F. \
                Then, what are the entities connected to entities in set G by relation {relation3}?\
                """ 
    
    def generate_question_pi(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="pi")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1}, F is the set of entities \
                connected to entities in E by relation {relation2}, and G is the set of entities \
                connected to entity {entity2} by relation {relation3}. \
                Then, what are the entities in the intersection of set F and G, i.e., \
                entities present in both F and G? \
                """ 

    def generate_question_2u(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="2u")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1} and F is the set of entities \
                connected to entity {entity2} by relation {relation2}. \
                Then, what are the entities in the union of set F and G, i.e., \
                entities present in either F or G? \
                """ 
    
    def generate_question_up(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="up")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by relation {relation1} and F is the set of entities \
                connected to entity {entity2} by relation {relation2}. G is the \
                set of entities in the union of E and F. \
                Then, what are the entities connected to entities in G by relation {relation3}? \
                """ 

    def generate_question_nin(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="nin")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        return f"""Let us assume that the set of entities E is connected to \
                entity {entity1} by any relation other than {relation1}, F is the set of entities \
                connected to entity {entity2} by any relation other than {relation2} and G is the \
                set of entities in the intersection of E and F. \
                Then, what are the entities which are not in the set G? \
                """ 
    
    def generate_question_nipn(self,logical_query):
        e1, r1, e2, r2, e3, r3 = self.parse_logical_query(logical_query,query_type="nipn")
        entity1 = e1
        entity2 = e2
        relation1 = r1
        relation2 = r2 
        relation3 = r3
        return f"""Let us assume that the set of entities E is connected to entity {entity1} \
                by any relation other than relation {relation1} and \
                F is the set of entities connected to entity {entity2} by any relation \
                other than {relation2}. G is the set of entities in the intersection of E and F. \
                Then, what are the entities connected to entities in G by any relation other than {relation3}? \
                """ 

    def generate_question(self, logical_query, query_type):
        assert (query_type in self.query_structs),f"Only the following {list(self.query_structs.keys())} are supported."
        if query_type=="1p": return self.generate_question_1p(logical_query)
        if query_type=="2p": return self.generate_question_2p(logical_query)
        if query_type=="3p": return self.generate_question_3p(logical_query)
        if query_type=="2i": return self.generate_question_2i(logical_query)
        if query_type=="3i": return self.generate_question_3i(logical_query)
        if query_type=="2in": return self.generate_question_2in(logical_query)
        if query_type=="3in": return self.generate_question_3in(logical_query)
        if query_type=="inp": return self.generate_question_inp(logical_query)
        if query_type=="pin": return self.generate_question_pin(logical_query)
        if query_type=="pni": return self.generate_question_pni(logical_query)
        if query_type=="ip": return self.generate_question_ip(logical_query)
        if query_type=="pi": return self.generate_question_pi(logical_query)
        if query_type=="2u": return self.generate_question_2u(logical_query)
        if query_type=="up": return self.generate_question_up(logical_query)
        if query_type=="nin": return self.generate_question_nin(logical_query)
        if query_type=="nipn": return self.generate_question_nipn(logical_query)

    def generate_prompt(self, logical_query, query_type):
        question = self.generate_question(logical_query, query_type)
        return "".join([self.question_tag, question, self.explain_tag])
