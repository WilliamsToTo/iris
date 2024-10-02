import dagma.utils as utils
from cdt.metrics import SID
from utils import adj_mat_to_edge_list, is_dag

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, confusion_matrix

def accuracy(B_true, B_est):
  '''
  B_true and B_est are edge lists
  '''
  B_true = set(B_true)
  B_est = set(B_est)
  return len(B_true.intersection(B_est)) / len(B_true.union(B_est))

def precision(B_true, B_est):
  '''
  B_true and B_est are edge lists
  '''
  B_true = set(B_true)
  B_est = set(B_est)
  if len(B_est) == 0:
    return 0.0
  else:
    return len(B_true.intersection(B_est)) / len(B_est)

def recall(B_true, B_est):
  '''
  B_true and B_est are edge lists
  '''
  B_true = set(B_true)
  B_est = set(B_est)
  return len(B_true.intersection(B_est)) / len(B_true)

def F_score(B_true, B_est):
  '''
  B_true and B_est are edge lists
  '''
  p = precision(B_true, B_est)
  r = recall(B_true, B_est)
  if p + r == 0:
    return "p + r = 0"
  return 2 * p * r / (p + r)


def normalized_hamming_distance(prediction, target, total_nodes):
  '''
  prediction and target are edge lists
  calculate the normalized hamming distance

  For a graph with m nodes, the distance is given by ∑m i,j=1 1 m2 1Gij 6=G′ ij , 
  the number of edges that are present in one graph but not the other, 
  divided by the total number of all possible edges.
  '''
  prediction = set(prediction)
  target = set(target)
  if total_nodes is None:
    total_nodes = set()
    for i,j in target:
      total_nodes.add(i)
      total_nodes.add(j)

  no_overlap = len(prediction.union(target)) - len(prediction.intersection(target))
  nhd = no_overlap / (len(total_nodes) ** 2)
  reference_nhd = min((len(prediction) + len(target)), (len(total_nodes) ** 2))/ (len(total_nodes) ** 2)
  return nhd, reference_nhd, nhd / reference_nhd

def compute_metrics(B_true_matrix, B_est_matrix, nodes):
  '''
  B_true and B_est are adjacency matrices
  '''
  # print(B_true_matrix, B_est_matrix)
  is_dag_true = is_dag(B_true_matrix)
  is_dag_est = is_dag(B_est_matrix)

  B_true = adj_mat_to_edge_list(B_true_matrix)
  B_est = adj_mat_to_edge_list(B_est_matrix)
  #print(B_est)

  nhd = normalized_hamming_distance(B_est, B_true, nodes)
  try:
    sid = SID(B_true, B_est)
  except:
    sid = 'Not a DAG'
  return {
    'precision': precision_score(B_true_matrix.flatten(), B_est_matrix.flatten()),
    'recall': recall_score(B_true_matrix.flatten(), B_est_matrix.flatten()),
    'F_score': f1_score(B_true_matrix.flatten(), B_est_matrix.flatten()),
    'accuracy': accuracy_score(B_true_matrix.flatten(), B_est_matrix.flatten()),
    'tn, fp, fn, tp': confusion_matrix(B_true_matrix.flatten(), B_est_matrix.flatten()).ravel().tolist(),
    'Number of predicted edges': len(B_est),
    'Number of true edges': len(B_true),
    'SID': sid,
    'Is true graph a DAG?': is_dag_true,
    'Is estimated graph a DAG?': is_dag_est,
    'NHD': hamming_loss(B_true_matrix.flatten(), B_est_matrix.flatten()),
    'REFERENCE NHD': nhd[1],
    'RATIO': hamming_loss(B_true_matrix.flatten(), B_est_matrix.flatten())/nhd[1],
  }