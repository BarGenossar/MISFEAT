import os
import torch
from matplotlib import pyplot as plt
import pickle
import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Evaluation
from utils import *
from torch_geometric.nn import to_hetero
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print_results(results, at_k, comb_size, g_id)
    return results


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


def print_results(results, at_k, comb_size, subgroup):
    print(5 * '======================')
    print(f'Evaluation results for subgroup {subgroup} with comb_size={comb_size} and at_k={at_k}:')
    for metric in results:
        for k in at_k:
            print(f'{metric}@{k}: {results[metric][k]}')
        print(5*'-------------------')
    return


def save_results(test_results, dir_path, seed):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    results_path = dir_path + f'results_seed{seed}.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(test_results, f)
    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds_num', type=int, default=3)
    parser.add_argument('--formula', type=str, default=str(LatticeGeneration.formula_idx))
    parser.add_argument('--config', type=str, default=str(LatticeGeneration.hyperparams_idx))
    parser.add_argument('--model', type=str, default=GNN.gnn_model)
    parser.add_argument('--hidden_channels', type=int, default=GNN.hidden_channels)
    parser.add_argument('--num_layers', type=int, default=GNN.num_layers)
    parser.add_argument('--p_dropout', type=float, default=GNN.p_dropout)
    parser.add_argument('--epochs', type=int, default=GNN.epochs)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size', type=int, default=Evaluation.comb_size)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--display', type=bool, default=True)
    args = parser.parse_args()

    dataset_path, graph_path, dir_path = read_paths(args)
    feature_num = read_feature_num_from_txt(dataset_path)

    lattice_graph = torch.load(graph_path)
    for seed in range(1, args.seeds_num + 1):
        info_string = generate_info_string(args, seed)
        torch.manual_seed(seed)
        model = LatticeGNN(args.model, feature_num, args.hidden_channels, args.num_layers, args.p_dropout)
        model = to_hetero(model, lattice_graph.metadata())
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.MSELoss()
        subgroups = lattice_graph.x_dict.keys()
        loss_vals = {subgroup: [] for subgroup in subgroups}
        for subgroup in subgroups:
            print(f"\nTraining on subgroup {subgroup}...")
            for epoch in range(1, args.epochs + 1):
                loss_val = train(lattice_graph, model, subgroup, optimizer, criterion)
                loss_vals[subgroup].append(loss_val)
                if epoch == 1 or epoch % 5 == 0:
                    print(f'Epoch: {epoch}, Loss: {round(loss_val, 4)}')
            test_results = test(lattice_graph, model, subgroup, args.at_k, args.comb_size, feature_num)
        save_results(test_results, dir_path, seed)
