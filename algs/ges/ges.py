import numpy as np
from tqdm import tqdm
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ScoreBased.GES import ges as inner_ges


def dag_to_adj_mat(dag):
    # Transform causal learn format into bn format
    n = len(dag.graph)
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if dag.graph[i][j] == -1:
                adj_mat[i][j] = 1
    return adj_mat


def ges(data, score_func='local_score_BIC', maxP=None, parameters=None, node_names=None):
    """
    Wrapper for GES algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    predicted_dag = inner_ges(data, score_func=score_func, maxP=maxP, parameters=parameters, node_names=node_names)['G']
    predicted_adj_mat = dag_to_adj_mat(predicted_dag)
    return predicted_adj_mat, predicted_dag


def ges_post_add_remove_relations(data, score_func='local_score_BIC', maxP=None, parameters=None, node_names=None, add_relations=None, remove_relations=None):
    """
    Wrapper for GES algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
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

    # run ges
    predicted_dag = inner_ges(data, score_func=score_func, maxP=maxP, parameters=parameters, node_names=node_names)['G']

    for edge in bk.required_rules_specs:
        add_edge = predicted_dag.get_edge(edge[0], edge[1])
        if add_edge is not None:
            print(add_edge.node1, add_edge.node2, add_edge.endpoint1, add_edge.endpoint2)
            if add_edge.endpoint1 == Endpoint.ARROW and add_edge.endpoint2 == Endpoint.TAIL:
                predicted_dag.remove_edge(Edge(edge[0], edge[1], Endpoint.ARROW, Endpoint.TAIL))
                predicted_dag.add_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
            else:
                predicted_dag.add_directed_edge(edge[0], edge[1])
        else:
            predicted_dag.add_directed_edge(edge[0], edge[1])
    for edge in bk.forbidden_rules_specs:
        remove_edge = predicted_dag.get_edge(edge[0], edge[1])
        if remove_edge is not None:
            if remove_edge.endpoint1 == Endpoint.TAIL and remove_edge.endpoint2 == Endpoint.TAIL:
                predicted_dag.remove_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.TAIL))
                predicted_dag.add_directed_edge(edge[1], edge[0])
            elif remove_edge.endpoint1 == Endpoint.TAIL and remove_edge.endpoint2 == Endpoint.ARROW:
                predicted_dag.remove_edge(Edge(edge[0], edge[1], Endpoint.TAIL, Endpoint.ARROW))

    predicted_adj_mat = dag_to_adj_mat(predicted_dag)

    return predicted_adj_mat, predicted_dag