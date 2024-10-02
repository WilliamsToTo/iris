import os
import json
import argparse
import itertools
import math
import torch
import ast
import pandas as pd
import numpy as np

from itertools import product, islice
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from algs.retrieve.google_retrieve import deep_retrieve_by_google, deep_retrieve_by_google_academic
from algs.extract.extraction import extract_value_of_variables, causal_claim_verification, abstract_new_variables
from algs.pc.call_pc_proposal import run_only_add_relations, run_pc_post_add_relations, run_pc_post_add_remove_relations
from algs.ges.ges import ges, ges_post_add_remove_relations
from algs.notears.notears import notears, notears_post_add_remove_relations

from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphClass import CausalGraph

from metrics import compute_metrics


def get_args():
    parser = argparse.ArgumentParser(description='LLM4Causal')
    parser.add_argument('--dataset', type=str, default='cancer', help='dataset name')
    parser.add_argument('--alg', type=str, default='pc', help='algorithm name')
    parser.add_argument('--llm', type=str, default="llama-3.1", help='llm path, gpt-3.5-turbo, gpt-4o-mini, gpt-4o')
    parser.add_argument('--lambda1', type=float, default=0.01, help='lambda for NOTEARS and DAGMA')
    parser.add_argument('--w_threshold', type=float, default=0.3, help='w_threshold for NOTEARS')
    parser.add_argument('--logdir', type=str, default='./logs', help='log directory')
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

#################### retrieve relevant documents by google search ##################
retrieved_docs_file = f"{target_dir}/{args.dataset}_retrieved_docs.json"
if not os.path.exists(retrieved_docs_file): #True
    # create retrieve query
    terms = []
    for var, syn in synonyms.items():
        terms.append([var]+syn)
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
    for retrieval_terms in tqdm(unique_combinations[10000:]):
        query = ""
        for term in retrieval_terms:
            query += f"\"{term}\" "
        query = query.strip()
        docs = deep_retrieve_by_google(query, docs, pages_per_query=2*len(retrieval_terms))
        if len(docs) > 1000:
            break

        print(f"Current number of retrieve documents: {len(docs)}")

    # 3. save retrieved results to avoid duplicate work
    json.dump(docs, open(retrieved_docs_file, "w+"), indent=4)

else:
    docs = json.load(open(retrieved_docs_file, "r"))


############## create table data ##################
# create table data
table_file = f'{target_dir}/{args.dataset}_{args.llm}_extracted_table_data.csv'
if not os.path.exists(table_file):
    samples = []
    for url, doc in tqdm(docs.items(), desc="create table"):
        exacted_values = extract_value_of_variables(doc, nodes, model, tokenizer)
        samples.append(exacted_values)

        df = pd.DataFrame(samples)
        df.to_csv(table_file, index=False)
else:
    df = pd.read_csv(table_file)

################## Find explicitly mentions of causal relations. #################
explicit_causal_relation_evidence_file = f"{target_dir}/{args.dataset}_explicit_causal_relation_evidence.json"
if not os.path.exists(explicit_causal_relation_evidence_file):
    original_var = synonyms.keys()
    var_pairs = list(itertools.combinations(original_var, 2))

    pair2docs = {}
    for pair in tqdm(var_pairs):
        docs = {}
        query = f"\"{pair[0]}\" \"{pair[1]}\""
        docs = deep_retrieve_by_google_academic(query, docs, pages_per_query=3)
        for syn0 in synonyms[pair[0]]:
            for syn1 in synonyms[pair[1]]:
                query = f"\"{syn0}\" \"{syn1}\""
                docs = deep_retrieve_by_google_academic(query, docs, pages_per_query=1)
        pair2docs[pair] = docs

    explicit_causal_relation_evidence_file = f"{target_dir}/{args.dataset}_explicit_causal_relation_evidence.json"
    str_keys_pair2docs = {str(k): v for k, v in pair2docs.items()}
    json.dump(str_keys_pair2docs, open(explicit_causal_relation_evidence_file, "w+"), indent=4)
else:
    str_keys_pair2docs = json.load(open(explicit_causal_relation_evidence_file, "r"))
    pair2docs = {ast.literal_eval(k): v for k, v in str_keys_pair2docs.items()}

################ verify causal relations using retrieved docs ##################
causal_verify_file = f"{target_dir}/{args.dataset}_{args.llm}_causal_relation_verification_results.json"
if not os.path.exists(causal_verify_file):
    claim_veracity = {}
    for pair, docs in tqdm(pair2docs.items()):
        claim1 = f"{pair[0]} causes {pair[1]}"
        claim2 = f"{pair[1]} causes {pair[0]}"
        claim1_veracity_perDocs = causal_claim_verification(claim1, docs, model, tokenizer)
        claim2_veracity_perDocs = causal_claim_verification(claim2, docs, model, tokenizer)
        claim_veracity[claim1] = claim1_veracity_perDocs
        claim_veracity[claim2] = claim2_veracity_perDocs
        json.dump(claim_veracity, open(causal_verify_file, "w+"), indent=4)
else:
    claim_veracity = json.load(open(causal_verify_file, "r"))

############## merge statistical and relation extraction results #################
verified_relations = []
remove_relations = []
for relation, veracity in claim_veracity.items():
    if veracity.count("True") > 0.5*len(veracity):
        verified_relations.append(relation)
    elif veracity.count("False") > 0.5*len(veracity):
        remove_relations.append(relation)
print("add relations: ", verified_relations)
print("remove relations: ", remove_relations)

def convert_values(x):
    if x is True:
        return 1
    elif x is False:
        return -1
    else:
        return 0


def set_true_edges_in_matrix(variables, edges):
    # Create a mapping of variable names to indices
    index_map = {name: idx for idx, name in enumerate(variables)}

    # Initialize the matrix with zeros
    n = len(variables)
    m = np.zeros((n, n), dtype=int)

    # Set the corresponding positions for the edges to 1
    for edge in edges:
        if edge[0] in index_map and edge[1] in index_map:
            src_idx = index_map[edge[0]]
            dst_idx = index_map[edge[1]]
            m[src_idx, dst_idx] = 1

    return m

if args.dataset in ["cancer", "diabetes", "obesity", "respiratory", "adni"]:
    df = df.applymap(convert_values)
data = df.to_numpy()
var_names = df.columns

true_matrix = set_true_edges_in_matrix(var_names, edges)

if args.alg == "pc":
    predicted_adj_mat, predicted_dag = run_pc_post_add_remove_relations(data, var_names, verified_relations, remove_relations)
elif args.alg == "ges":
    # predicted_adj_mat, predicted_dag = ges(data, node_names=var_names)
    predicted_adj_mat, predicted_dag = ges_post_add_remove_relations(data, node_names=var_names, add_relations=verified_relations, remove_relations=remove_relations)
elif args.alg == "notears":
    # predicted_adj_mat, predicted_dag = notears(data, lambda1=args.lambda1, loss_type='l2', w_threshold=args.w_threshold, node_names=var_names)
    predicted_adj_mat, predicted_dag = notears_post_add_remove_relations(data, lambda1=args.lambda1, loss_type='l2',
                                                                         w_threshold=args.w_threshold, node_names=var_names,
                                                                         add_relations=verified_relations,
                                                                         remove_relations=remove_relations)
print(predicted_adj_mat)
metrics = compute_metrics(true_matrix, predicted_adj_mat, var_names)
print(metrics)

if isinstance(predicted_dag, CausalGraph):
    pyd = GraphUtils.to_pydot(predicted_dag.G, labels=var_names)
elif isinstance(predicted_dag, GeneralGraph):
    pyd = GraphUtils.to_pydot(predicted_dag, labels=var_names)
pyd.write_png(f"{target_dir}/{args.dataset}_{args.alg}_{args.llm}_predict_dag.png")