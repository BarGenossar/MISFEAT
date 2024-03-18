import math
import torch
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Evaluation
import tqdm
from utils import *
from torch_geometric.nn import to_hetero
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train(lattice_graph, model, g_id, optimizer, criterion):
    lattice_graph.to(device)
    model.train()
    optimizer.zero_grad()
    out = model(lattice_graph.x_dict, lattice_graph.edge_index_dict)
    labels = lattice_graph[g_id].y
    predictions = out[g_id]
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(lattice_graph, model, g_id, at_k, comb_size, feature_num):
    lattice_graph.to(device)
    model.eval()
    with torch.no_grad():
        out = model(lattice_graph.x_dict, lattice_graph.edge_index_dict)
    labels = lattice_graph[g_id].y
    predictions = out[g_id]
    results = compute_eval_metrics(labels, predictions, at_k, comb_size, feature_num)
    print_results(results, at_k, comb_size)
    save_results(results, at_k, comb_size)
    return


def compute_eval_metrics(ground_truth, predictions, at_k, comb_size, feature_num):
    eval_metrics, eval_metric_func = get_eval_metric_func()
    comb_size_indices = get_comb_size_indices(len(predictions), comb_size, feature_num)
    sorted_gt_indices = get_sorted_indices(ground_truth, comb_size_indices)
    sorted_pred_indices = get_sorted_indices(predictions, comb_size_indices)
    results = dict()
    for metric in eval_metrics:
        results[metric] = dict()
        if metric not in eval_metric_func:
            raise ValueError(f"Invalid evaluation metric: {metric}")
        else:
            if type(at_k) is not list:
                at_k = [at_k]
            for k in at_k:
                results = eval_metric_func[metric](ground_truth, k, sorted_gt_indices, sorted_pred_indices, results)
    return results


def get_eval_metric_func():
    # TODO: Consider adding more evaluation metrics
    eval_metric_func = {
        'ndcg': compute_ndcg,
        'hits': compute_hits
    }
    return Evaluation.eval_metrics, eval_metric_func


def get_comb_size_indices(num_nodes, comb_size, feature_num):
    comb_size_indices = []
    for i in range(num_nodes):
        binary_vec = convert_decimal_to_binary(i+1, feature_num)
        if binary_vec.count('1') == comb_size:
            comb_size_indices.append(i)
    return comb_size_indices


def get_sorted_indices(score_tensor, comb_size_indices):
    sorted_indices = torch.argsort(score_tensor, descending=True)
    return [idx.item() for idx in sorted_indices if idx.item() in comb_size_indices]


def compute_dcg(ground_truth, sorted_indices, at_k):
    DCG = 0
    for i in range(1, at_k + 1):
        DCG += (math.pow(2, ground_truth[sorted_indices[i-1]].item()) - 1) / math.log2(i+1)
    return DCG


def compute_ndcg(ground_truth, k, sorted_gt_indices, sorted_pred_indices, results):
    IDCG = compute_dcg(ground_truth, sorted_gt_indices, k)
    DCG = compute_dcg(ground_truth, sorted_pred_indices, k)
    results['ndcg'][k] = round(DCG / IDCG, 4)
    return results

def compute_hits(ground_truth, k, sorted_gt_indices, sorted_pred_indices, results):
    hits = sum([1 for i in range(k) if sorted_pred_indices[i] in sorted_gt_indices[:k]])
    results['hits'][k] = round(hits / k, 4)
    return results

def print_results(results, at_k, comb_size):
    print(f'Evaluation results for comb_size={comb_size} and at_k={at_k}:\n')
    for metric in results:
        for k in at_k:
            print(f'{metric}@{k}: {results[metric][k]}')
        print(5*'-------------------')
    return


def save_results(results, at_k, comb_size):
    # TODO: Implement saving the results to a file
    return

if __name__ == "__main__":
    formula_idx = LatticeGeneration.formula_idx
    hyperparams_idx = LatticeGeneration.hyperparams_idx
    min_k = LatticeGeneration.min_k
    within_level_edges = LatticeGeneration.within_level_edges
    is_hetero = LatticeGeneration.is_hetero
    with_edge_attrs = LatticeGeneration.with_edge_attrs

    gnn_model = GNN.gnn_model
    hidden_channels = GNN.hidden_channels
    num_layers = GNN.num_layers
    p_dropout = GNN.p_dropout
    epochs = GNN.epochs

    at_k = Evaluation.at_k
    comb_size = Evaluation.comb_size



    dataset_path = f"GeneratedData/Formula{formula_idx}/Config{hyperparams_idx}/dataset.pkl"
    lattice_path = f"GeneratedData/Formula{formula_idx}/Config{hyperparams_idx}/dataset_hetero_graph.pt"
    feature_num = read_feature_num_from_txt(dataset_path)

    lattice_graph = torch.load(lattice_path)
    model = LatticeGNN(gnn_model, feature_num, hidden_channels, num_layers, p_dropout)
    model = to_hetero(model, lattice_graph.metadata())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    subgroups = lattice_graph.x_dict.keys()

    for subgroup in subgroups:
        print(f"\nTraining on subgroup {subgroup}...")
        for epoch in range(1, epochs + 1):
            loss_val = train(lattice_graph, model, subgroup, optimizer, criterion)
            print(f'Epoch: {epoch}, Loss: {round(loss_val, 4)}')
        print(5*'====================')
        print("Results:")
        test(lattice_graph, model, subgroup, at_k, comb_size, feature_num)
