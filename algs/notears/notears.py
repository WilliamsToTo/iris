import numpy as np
from tqdm import tqdm
from .linear import notears_linear
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def dag_to_adj_mat(dag):
    # Transform causal learn format into bn format
    n = len(dag.G.graph)
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if dag.G.graph[i][j] == -1:
                adj_mat[i][j] = 1
    return adj_mat

def notears(data, lambda1, loss_type, node_names, w_threshold=0.3):
    """Wrapper for NOTEARS."""
    predicted_dag = notears_linear(data, lambda1, loss_type, w_threshold=w_threshold)
    predicted_adj_mat = (predicted_dag != 0).astype(int)
    # create dag
    predicted_cg = CausalGraph(data.shape[1], node_names)
    nodes = predicted_cg.G.get_nodes()
    predicted_cg.G.graph[:] = 0.

    n = len(predicted_adj_mat)
    for i in range(n):
        for j in range(n):
            if predicted_adj_mat[i, j] == 1:
                predicted_cg.G.add_directed_edge(nodes[i], nodes[j])
    predicted_adj_mat = dag_to_adj_mat(predicted_cg)

    return predicted_adj_mat, predicted_cg

def notears_post_add_remove_relations(data, lambda1, loss_type, node_names, add_relations, remove_relations, w_threshold=0.3):
    """Wrapper for NOTEARS."""

    # create background
    init_cg = CausalGraph(data.shape[1], node_names)
    nodes = init_cg.G.get_nodes()
    bk = BackgroundKnowledge()
    for x in range(len(nodes)):
        for y in range(len(nodes)):
            if x != y:
                x_name = nodes[x].get_name()
                y_name = nodes[y].get_name()
                if f"{x_name} causes {y_name}" in add_relations:
                    bk.add_required_by_node(nodes[x], nodes[y])
                elif f"{x_name} causes {y_name}" in remove_relations:
                    bk.add_forbidden_by_node(nodes[x], nodes[y])

    # run notears
    predicted_dag = notears_linear(data, lambda1, loss_type, w_threshold=w_threshold)
    predicted_adj_mat = (predicted_dag != 0).astype(int)
    # create dag
    predicted_cg = CausalGraph(data.shape[1], node_names)
    nodes = predicted_cg.G.get_nodes()
    predicted_cg.G.graph[:] = 0.

    n = len(predicted_adj_mat)
    for i in range(n):
        for j in range(n):
            if predicted_adj_mat[i, j] == 1:
                predicted_cg.G.add_directed_edge(nodes[i], nodes[j])
    # predicted_adj_mat = dag_to_adj_mat(predicted_cg)

    for edge in bk.required_rules_specs:
        add_edge = predicted_cg.G.get_edge(edge[0], edge[1])
        if add_edge is not None:
            print(add_edge.node1, add_edge.node2, add_edge.endpoint1, add_edge.endpoint2)
            if add_edge.endpoint1 == Endpoint.ARROW and add_edge.endpoint2 == Endpoint.TAIL:
                predicted_cg.G.remove_edge(Edge(edge[0], edge[1], Endpoint.ARROW, Endpoint.TAIL))
                predicted_cg.G.add_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
            else:
                predicted_cg.G.add_directed_edge(edge[0], edge[1])
        else:
            predicted_cg.G.add_directed_edge(edge[0], edge[1])
    for edge in bk.forbidden_rules_specs:
        remove_edge = predicted_cg.G.get_edge(edge[0], edge[1])
        if remove_edge is not None:
            if remove_edge.endpoint1 == Endpoint.TAIL and remove_edge.endpoint2 == Endpoint.TAIL:
                predicted_cg.G.remove_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
                predicted_cg.G.add_directed_edge(edge[1], edge[0])
            elif remove_edge.endpoint1 == Endpoint.TAIL and remove_edge.endpoint2 == Endpoint.ARROW:
                predicted_cg.G.remove_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.ARROW))

    predicted_adj_mat = dag_to_adj_mat(predicted_cg)

    return predicted_adj_mat, predicted_cg