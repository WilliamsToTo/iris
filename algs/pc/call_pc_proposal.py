import numpy as np
from .PC import pc as inner_pc
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphClass import CausalGraph

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

def run_only_add_relations(data, node_names, relations):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """

    init_cg = CausalGraph(data.shape[1], node_names) #TODO: remove all edges
    nodes = init_cg.G.get_nodes()
    init_cg.G.graph[:] = 0
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                if f"{x_name} causes {y_name}" in relations:
                    bk.add_required_by_node(nodes[x], nodes[y])

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    for edge in bk.required_rules_specs:
        add_edge = init_cg.G.get_edge(edge[0], edge[1])
        if add_edge is not None:
            init_cg.G.remove_edge(Edge(edge[0], edge[1], Endpoint.ARROW, Endpoint.TAIL))
            init_cg.G.add_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
        else:
            init_cg.G.add_directed_edge(edge[0], edge[1])

    predicted_adj_mat = dag_to_adj_mat(init_cg)

    return predicted_adj_mat, init_cg

def run_pc_post_add_relations(data, node_names, relations):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                if f"{x_name} causes {y_name}" in relations:
                    bk.add_required_by_node(nodes[x], nodes[y])

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    predicted_dag = inner_pc(data, indep_test="mv_fisherz", node_names=node_names, verbose=True, mvpc=True)

    for edge in bk.required_rules_specs:
        add_edge = predicted_dag.G.get_edge(edge[0], edge[1])
        if add_edge is not None:
            if add_edge.endpoint1 == Endpoint.ARROW and add_edge.endpoint2 == Endpoint.TAIL:
                predicted_dag.G.remove_edge(Edge(edge[0], edge[1], Endpoint.ARROW, Endpoint.TAIL))
                predicted_dag.G.add_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
            else:
                predicted_dag.G.add_directed_edge(edge[0], edge[1])
        else:
            predicted_dag.G.add_directed_edge(edge[0], edge[1])

    predicted_adj_mat = dag_to_adj_mat(predicted_dag)

    return predicted_adj_mat, predicted_dag

def run_pc_post_add_remove_relations(data, node_names, add_relations, remove_relations):
    """
    Wrapper for PC algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """

    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()
    for x in tqdm(range(len(nodes))):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                if f"{x_name} causes {y_name}" in add_relations:
                    bk.add_required_by_node(nodes[x], nodes[y])
                elif f"{x_name} causes {y_name}" in remove_relations:
                    bk.add_forbidden_by_node(nodes[x], nodes[y])

    for edge in bk.required_rules_specs:
        print(f"required edges: {edge[0].get_name()} -> {edge[1].get_name()}")
    for edge in bk.forbidden_rules_specs:
        print(f"forbidden edges: {edge[0].get_name()} -> {edge[1].get_name()}")

    predicted_dag = inner_pc(data, indep_test="mv_fisherz", node_names=node_names, verbose=True, mvpc=True)

    for edge in bk.required_rules_specs:
        add_edge = predicted_dag.G.get_edge(edge[0], edge[1])
        if add_edge is not None:
            print(f"{edge[0].get_name()} -> {edge[1].get_name()}")
            print(add_edge.node1, add_edge.node2, add_edge.endpoint1, add_edge.endpoint2)
            if add_edge.endpoint1 == Endpoint.TAIL and add_edge.endpoint2 == Endpoint.TAIL:
                predicted_dag.G.add_directed_edge(edge[0], edge[1])
            else:
                predicted_dag.G.remove_edge(add_edge)
                predicted_dag.G.add_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
        else:
            predicted_dag.G.add_directed_edge(edge[0], edge[1])
    for edge in bk.forbidden_rules_specs:
        remove_edge = predicted_dag.G.get_edge(edge[0], edge[1])
        if remove_edge is not None:
            if remove_edge.endpoint1 == Endpoint.TAIL and remove_edge.endpoint2 == Endpoint.TAIL:
                predicted_dag.G.remove_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
                predicted_dag.G.add_directed_edge(edge[1], edge[0])
            elif remove_edge.endpoint1 == Endpoint.TAIL and remove_edge.endpoint2 == Endpoint.ARROW:
                predicted_dag.G.remove_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.ARROW))

    predicted_adj_mat = dag_to_adj_mat(predicted_dag)

    return predicted_adj_mat, predicted_dag