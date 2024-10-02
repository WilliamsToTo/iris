import os
import re
import json
import argparse
import itertools
import math
import torch
import ast
import time
import pandas as pd
import numpy as np

from itertools import product, islice
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from algs.retrieve.google_retrieve import deep_retrieve_by_google, deep_retrieve_by_google_academic, count_occurrence
from algs.extract.extraction import extract_value_of_variables, causal_claim_verification, abstract_new_variables
from algs.pc.call_pc_proposal import run_only_add_relations, run_pc_post_add_relations, run_pc_post_add_remove_relations
from causallearn.utils.GraphUtils import GraphUtils
from metrics import compute_metrics


def replace_symbols(strings):
    # Define a regular expression pattern for symbols to replace
    pattern = r"[\\*().-_/#''&]"

    # Use a list comprehension to replace symbols in each string
    replaced_strings = [re.sub(pattern, ' ', s) for s in strings]
    replaced_strings = [" ".join(s.split()) for s in replaced_strings]
    replaced_strings = [s.strip() for s in replaced_strings]
    replaced_strings = list(set(replaced_strings))

    return replaced_strings


def remove_var(strings, exist_var):
    # Use a list comprehension to filter out elements that exist in exist_var
    filtered_strings = [s for s in strings if s not in exist_var]
    # remove too long variables
    filtered_strings = [s for s in filtered_strings if len(s.split()) < 4]
    return filtered_strings


def get_args():
    parser = argparse.ArgumentParser(description='LLM4Causal')
    parser.add_argument('--dataset', type=str, default='cancer', help='dataset name')
    parser.add_argument('--llm', type=str, default="llama-3.1", help='llm name')
    args = parser.parse_args()
    return args

args = get_args()

if args.dataset == "cancer":
    graph = json.load(open("./data_woTable/cancer.json"))
elif args.dataset == "diabetes":
    graph = json.load(open("./data_woTable/diabetes.json"))
elif args.dataset == "obesity":
    graph = json.load(open("./data_woTable/obesity.json"))
elif args.dataset == "respiratory":
    graph = json.load(open("./data_woTable/respiratory.json"))
elif args.dataset == "adni":
    graph = json.load(open("./data_woTable/adni.json"))

nodes = graph["nodes"]
synonyms = graph["synonyms"]
edges = graph["edges"]

target_dir = "./data_woTable"

# load llm
if "llama-3.1" in args.llm:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained("path to model",
                                                 quantization_config=quantization_config,
                                              torch_dtype=torch.bfloat16, device_map="auto",)
    tokenizer = AutoTokenizer.from_pretrained("path to model")
else:
    model = args.llm
    tokenizer = args.llm

############## propose missing variables ########
retrieved_docs_file = f"{target_dir}/{args.dataset}_retrieved_docs.json"
if not os.path.exists(retrieved_docs_file):  # True
    # create retrieve query
    terms = []
    for var, syn in synonyms.items():
        terms.append([var] + syn)
    all_combinations = [(ele,) for sublist in terms for ele in sublist]

    # Generate full combinations of the length equal to the number of lists
    full_combinations = list(itertools.product(*terms))
    all_combinations += full_combinations
    for one_full in full_combinations:
        # Generate combinations for every length
        for r in range(len(one_full) + 1, 1, -1):
            combinations = list(itertools.combinations(one_full, r))
            all_combinations.extend(combinations)  # Store the combinations

    # Remove duplicates and ensure unique combinations
    unique_combinations = set(all_combinations)
    unique_combinations = sorted(unique_combinations, key=lambda x: -len(x))
    print(unique_combinations)
    print(f"the number of queries: {len(unique_combinations)}")

    # 2. retrieve docs by google search api
    docs = {}
    for retrieval_terms in tqdm(unique_combinations):
        query = ""
        for term in retrieval_terms:
            query += f"\"{term}\" "
        query = query.strip()
        docs = deep_retrieve_by_google(query, docs, pages_per_query=2 * len(retrieval_terms))
        if len(docs) > 1000:
            break

        print(f"Current number of retrieve documents: {len(docs)}")

    # 3. save retrieved results to avoid duplicate work
    json.dump(docs, open(retrieved_docs_file, "w+"), indent=4)

else:
    docs = json.load(open(retrieved_docs_file, "r"))

# Abstract new variables
new_variable_file = f'{target_dir}/{args.dataset}_{args.llm}_new_variables.json'
if not os.path.exists(new_variable_file):
    new_variables = []
    for url, doc in tqdm(docs.items()):
        new_var = abstract_new_variables(doc, nodes, model, tokenizer)
        if new_var is not None:
            new_variables.append(new_var)
    json.dump(new_variables, open(new_variable_file, "w+"), indent=4)
else:
    new_variables = json.load(open(new_variable_file, "r"))
new_variables = replace_symbols(new_variables)
new_variables = remove_var(new_variables, nodes.keys())

# Select variables - causal relation verification
new_var_relation_file = f"{target_dir}/{args.dataset}_{args.llm}_new_var_relation_veracity.json"
if not os.path.exists(new_var_relation_file):
    claim_veracity = {}
    for new_var in tqdm(new_variables):
        for node_name in nodes.keys():
            docs = {}
            docs = deep_retrieve_by_google_academic(f"\"{new_var}\" \"{node_name}\"", docs, pages_per_query=1)
            rs = [f"{new_var} causes {node_name}", f"{node_name} causes {new_var}"]
            for r in rs:
                for url, doc in docs.items():
                    veracity_perDocs = causal_claim_verification(r, docs, model, tokenizer)
                    claim_veracity[r] = veracity_perDocs
    json.dump(claim_veracity, open(new_var_relation_file, "w+"), indent=4)
else:
    claim_veracity = json.load(open(new_var_relation_file, "r"))

# select variables - statistical
def get_top_k_keys(d, k):
    # Sort the dictionary by value in descending order and extract the keys
    sorted_keys = sorted(d, key=d.get, reverse=True)

    # Return the top-k keys
    return sorted_keys[:k]

var_occur_file = f'{target_dir}/{args.dataset}_{args.llm}_new_variable_pmi.json'
if not os.path.exists(var_occur_file):
    print(new_variables, len(new_variables))

    new_variable_occurrences = {}
    for new_var in tqdm(new_variables):
        if len(new_var) > 0:
            new_var_query = f"\"{new_var}\""
            new_var_occur = count_occurrence(new_var_query)
            new_variable_occurrences[new_var] = new_var_occur
        else:
            new_variable_occurrences[new_var] = 0

    node_name_occurrences = {}
    for node_name in nodes.keys():
        node_name_query = f"\"{node_name}\""
        node_name_occur = count_occurrence(node_name_query)
        node_name_occurrences[node_name] = node_name_occur

    new_variables_occur_dict = {}
    for new_var,  new_var_occur in tqdm(new_variable_occurrences.items()):
        if len(new_var) > 0 and new_var_occur > 0:
            pmi = 0
            for node_name, node_name_occur in node_name_occurrences.items():
                co_occur_query = f"\"{new_var}\" \"{node_name}\""
                co_occur = count_occurrence(co_occur_query)
                print(new_var, new_var_occur, node_name, node_name_occur, co_occur)
                pmi += np.log(co_occur/(new_var_occur*node_name_occur))
            print(new_var, pmi)
            if pmi != 0:
                new_variables_occur_dict[new_var] = pmi
            time.sleep(10)
    json.dump(new_variables_occur_dict, open(var_occur_file, "w+"), indent=4)
else:
    new_variables_occur_dict = json.load(open(var_occur_file, "r"))

# final new variables
print(get_top_k_keys(new_variables_occur_dict, 5))
verified_relations = []
for relation, veracity in claim_veracity.items():
    if veracity.count("True") > 0.5*len(veracity):
        verified_relations.append(relation)
print(verified_relations)