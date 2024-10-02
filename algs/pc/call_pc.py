from .PC import pc as inner_pc
from ..llm.causalRelationVerify import selfCheck_majorVote, retrieveLLMVerfy, googleRetrieve, crossrefapiRetreve
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphClass import CausalGraph

import numpy as np
import torch
import json
import time
import hf_olmo
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

def dag_to_adj_mat(dag):
    # Transform causal learn format into bn format
    n = len(dag.G.graph)
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if dag.G.graph[i][j] == -1:
                adj_mat[i][j] = 1
    return adj_mat


def run_pc(data, node_names):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    predicted_dag = inner_pc(data, indep_test="fisherz", node_names=node_names, verbose=True)
    predicted_adj_mat = dag_to_adj_mat(predicted_dag)
    return predicted_adj_mat, predicted_dag

def run_prior_knowledge(data, llm_path, node_names, node_names_and_desc):
    """
        Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
        """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16, device_map="auto", )
    model.eval()

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    init_cg.G.graph = np.zeros((data.shape[1], data.shape[1]), np.dtype(int))
    # for i in range(data.shape[1]):
    #     for j in range(i + 1, data.shape[1]):
    #         self.G.remove_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                x_desc = node_names_and_desc[x_name].description
                y_desc = node_names_and_desc[y_name].description
                print(f"Does {x_desc} cause {y_desc}?")
                check_result = selfCheck_majorVote(x_desc, y_desc, model, tokenizer, num_return_sequences=10,
                                                   shot_num=3, threshold=0.5)
                if check_result == 'yes':
                    bk.add_required_by_node(nodes[x], nodes[y])
                    init_cg.G.add_directed_edge(nodes[x], nodes[y])
                elif check_result == 'no':
                    bk.add_forbidden_by_node(nodes[x], nodes[y])
                    # remove_edge = init_cg.G.get_edge(nodes[x], nodes[y])
                    # init_cg.G.remove_edge(remove_edge)

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")
    for edge in bk.forbidden_rules_specs:
        print(f"forbidden edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    # predicted_dag = inner_pc(data, indep_test="fisherz", node_names=node_names, background_knowledge=bk, verbose=True)
    predicted_adj_mat = dag_to_adj_mat(init_cg)

    return predicted_adj_mat, init_cg


def run_pc_prior_remove_knowledge(data, llm_path, node_names, node_names_and_desc):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16, device_map="auto", )
    model.eval()

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                x_desc = node_names_and_desc[x_name].description
                y_desc = node_names_and_desc[y_name].description
                print(f"Does {x_desc} cause {y_desc}?")
                check_result = selfCheck_majorVote(x_desc, y_desc, model, tokenizer, num_return_sequences=10, shot_num=3, threshold=0.5)
                if check_result == 'yes':
                    bk.add_required_by_node(nodes[x], nodes[y])
                elif check_result == 'no':
                    bk.add_forbidden_by_node(nodes[x], nodes[y])

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")
    for edge in bk.forbidden_rules_specs:
        print(f"forbidden edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    predicted_dag = inner_pc(data, indep_test="fisherz", node_names=node_names, background_knowledge=bk, verbose=True)
    predicted_adj_mat = dag_to_adj_mat(predicted_dag)

    return predicted_adj_mat, predicted_dag

def run_pc_post_add_knowledge(data, llm_path, node_names, node_names_and_desc):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16, device_map="auto", )
    model.eval()

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                x_desc = node_names_and_desc[x_name].description
                y_desc = node_names_and_desc[y_name].description
                print(f"Does {x_desc} cause {y_desc}?")
                check_result = selfCheck_majorVote(x_desc, y_desc, model, tokenizer, num_return_sequences=10, shot_num=3, threshold=0.5)
                if check_result == 'yes':
                    bk.add_required_by_node(nodes[x], nodes[y])
                elif check_result == 'no':
                    bk.add_forbidden_by_node(nodes[x], nodes[y])

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")
    for edge in bk.forbidden_rules_specs:
        print(f"forbidden edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    predicted_dag = inner_pc(data, indep_test="fisherz", node_names=node_names, verbose=True)

    for edge in bk.required_rules_specs:
        predicted_dag.G.add_directed_edge(edge[0], edge[1])

    predicted_adj_mat = dag_to_adj_mat(predicted_dag)

    return predicted_adj_mat, predicted_dag


def run_pc_prior_post_knowledge(data, llm_path, node_names, node_names_and_desc):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16, device_map="auto", )
    model.eval()

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                x_desc = node_names_and_desc[x_name].description
                y_desc = node_names_and_desc[y_name].description
                print(f"Does {x_desc} cause {y_desc}?")
                check_result = selfCheck_majorVote(x_desc, y_desc, model, tokenizer, num_return_sequences=10, shot_num=3, threshold=0.5)
                if check_result == 'yes':
                    bk.add_required_by_node(nodes[x], nodes[y])
                elif check_result == 'no':
                    bk.add_forbidden_by_node(nodes[x], nodes[y])

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")
    for edge in bk.forbidden_rules_specs:
        print(f"forbidden edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    predicted_dag = inner_pc(data, indep_test="fisherz", node_names=node_names, background_knowledge=bk, verbose=True)

    for edge in bk.required_rules_specs:
        predicted_dag.G.add_directed_edge(edge[0], edge[1])

    predicted_adj_mat = dag_to_adj_mat(predicted_dag)

    return predicted_adj_mat, predicted_dag


def run_googleSearch(data, node_names, node_names_and_desc, save_file):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    query2docs = {}
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                x_names = [node_names_and_desc[x_name].name] + node_names_and_desc[x_name].synonyms
                y_names = [node_names_and_desc[y_name].name] + node_names_and_desc[y_name].synonyms
                # print(f"Retrieve: {x_names} causes {y_names}")
                retrieved_docs = googleRetrieve(x_names, y_names)
                query2docs[f"{x_name} causes {y_name}"] = retrieved_docs
                json.dump(query2docs, open(save_file, "w+"), indent=4)
                time.sleep(60)
                print("wait 60 seconds")

    return query2docs


def run_crossrefapiSearch(data, node_names, node_names_and_desc):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    query2docs = {}
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                x_names = [node_names_and_desc[x_name].name] + node_names_and_desc[x_name].synonyms
                y_names = [node_names_and_desc[y_name].name] + node_names_and_desc[y_name].synonyms
                print(f"Retrieve: {x_names} causes {y_names}")
                retrieved_docs = crossrefapiRetreve(x_names, y_names)
                query2docs[f"{x_name} causes {y_name}"] = retrieved_docs

    return query2docs


def run_pc_prior_remove_knowledge_googleSearch(data, llm_path, node_names, node_names_and_desc, claim2docs):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16, device_map="auto", )
    model.eval()

    # process docs
    claim2snippets = {}
    for relation, items in claim2docs.items():
        claim2snippets[relation] = []
        for item in items:
            snippet = item["snippet"]
            claim2snippets[relation] += [snippet]

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()

    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                x_desc = node_names_and_desc[x_name].description
                y_desc = node_names_and_desc[y_name].description
                print(f"Claim: {x_desc} causes {y_desc}")
                check_result = retrieveLLMVerfy(x_name, y_name, x_desc, y_desc, claim2snippets, model, tokenizer, num_return_sequences=10, threshold=0.5)
                print(f"Claim: {x_desc} causes {y_desc}; Verification: {check_result}")
                if check_result == 'yes':
                    bk.add_required_by_node(nodes[x], nodes[y])
                elif check_result == 'no':
                    bk.add_forbidden_by_node(nodes[x], nodes[y])

    print("forbidden edges: ", bk.forbidden_rules_specs)
    print("required edges: ", bk.required_rules_specs)

    predicted_dag = inner_pc(data, indep_test="fisherz", node_names=node_names, background_knowledge=bk, verbose=True)
    predicted_adj_mat = dag_to_adj_mat(predicted_dag)
    return predicted_adj_mat, predicted_dag

def run_pc_prior_remove_knowledge_crossrefapiSearch(data, llm_path, node_names, node_names_and_desc, claim2docs):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    model = AutoModelForCausalLM.from_pretrained(llm_path, quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16, device_map="auto", )
    model.eval()

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_symbol = nodes[x].get_name()
                y_symbol = nodes[y].get_name()
                x_name = node_names_and_desc[x_symbol].name
                y_name = node_names_and_desc[y_symbol].name
                x_desc = node_names_and_desc[x_symbol].description
                y_desc = node_names_and_desc[y_symbol].description
                print(f"Claim: {x_desc} causes {y_desc}")
                check_result = retrieveLLMVerfy(x_name, y_name, x_desc, y_desc, claim2docs, model, tokenizer, num_return_sequences=10, threshold=0.5)

                if check_result == 'yes':
                    bk.add_required_by_node(nodes[x], nodes[y])
                # elif check_result == 'no':
                #     bk.add_forbidden_by_node(nodes[x], nodes[y])

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")
    # for edge in bk.forbidden_rules_specs:
    #     print(f"forbidden edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    predicted_dag = inner_pc(data, indep_test="fisherz", node_names=node_names, background_knowledge=bk, verbose=True)
    predicted_adj_mat = dag_to_adj_mat(predicted_dag)
    return predicted_adj_mat, predicted_dag