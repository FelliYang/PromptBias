from ast import expr_context
import json
import os
from utils import load_file
from functools import cmp_to_key
def print_relations_meta():
    data =  load_file("/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LAMA_relations.jsonl")

    type_ref = {"1-1":0, "N-1":1, "N-M":2}

    def cmp(r1, r2):
        t1 = type_ref[r1["type"]]
        t2 = type_ref[r2["type"]]
        if(t1<t2):
            return -1
        elif(t1>t2):
            return 1    
        else:
            num1 = int(r1["relation"][1:])
            num2 = int(r2["relation"][1:])
            if(num1>num2):
                return 1
            elif(num1<num2):
                return -1
            else:
                return 0

    data.sort(key = cmp_to_key(cmp))

    with open("./outputs/relation.txt","w") as f:
        f.write("relation\ttype\t\t\ttempalte{:30}\tlabel\n".format(""))
        for r in data:
            f.write("{:4}\t\t{}\t\t{:45}\t{}\n".format(r["relation"],r["type"],r["template"], r["label"]))

def print_relation_data():
    """
    this func extract subject and object from relation jsonl, then write them into a txt for esaier reference.
    """
    data =  load_file("/home/jiao/code/prompt/OptiPrompt/relation_metainfo/LAMA_relations.jsonl")
    relations = [ d["relation"] for d in data]
    output_dir = "/home/jiao/code/prompt/OptiPrompt/data/lama_txt"
    input_dir = "/home/jiao/code/prompt/OptiPrompt/data/autoprompt_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for r in relations:
        data = []
        try:
            with open(os.path.join(input_dir, r, "train.jsonl"), "r") as f:
                for line in f.readlines():
                    data.append(json.loads(line))
            with open(os.path.join(output_dir, r), "w") as f:
                for item in data:
                    f.write("{:20}{}\n".format(item["sub_label"],item["obj_label"]))
        except:
            continue



# print_relation_data()


    