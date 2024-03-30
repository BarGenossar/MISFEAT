import math
import torch
from config import Evaluation
import os
import pickle
import numpy as np


def convert_binary_to_decimal(binary):
    return int(binary, 2)


def convert_decimal_to_binary(decimal, feature_num):
    binary = bin(decimal)[2:]
    return '0' * (feature_num - len(binary)) + binary


def convert_comb_to_binary(comb, feature_num):
    if type(comb) == str:
        # A single feature tuple is actually a string
        idx = int(comb.split('_')[1])
        return convert_decimal_to_binary(2 ** idx, feature_num)
    binary = '0' * feature_num
    for feature in comb:
        idx = int(feature.split('_')[1])
        binary = binary[:idx] + '1' + binary[idx + 1:]
    return binary[::-1]


def get_lattice_nodes_num(feature_num, min_subset_size, max_subset_size):
    return sum([math.comb(feature_num, i) for i in range(min_subset_size, max_subset_size + 1)])


def get_lattice_edges_num(feature_num, min_subset_size, max_subset_size, within_levels=True):
    edges_num = sum([math.comb(feature_num, i) * i for i in range(min_subset_size+1, max_subset_size+1)])
    if within_levels:
        edges_num += sum([math.comb(feature_num, i) * i * (feature_num - i) for i
                          in range(max(2, min_subset_size), max_subset_size)])
    return edges_num


def read_feature_num_from_txt(dataset_path):
    description_path = dataset_path.replace('dataset.pkl', 'description.txt')
    description = open(description_path, 'r').readlines()
    for line in description:
        if 'feature_num' in line:
            return int(line.split(':')[-1].strip())
    return None


def compute_eval_metrics(ground_truth, predictions, at_k, comb_size, feature_num):
    eval_metrics, eval_func = get_eval_metric_func()
    comb_size_indices = get_comb_size_indices(len(predictions), comb_size, feature_num)
    sorted_gt_indices = get_sorted_indices(ground_truth, comb_size_indices)
    sorted_pred_indices = get_sorted_indices(predictions, comb_size_indices)
    g_results = {metric: dict() for metric in eval_metrics}
    for metric in eval_metrics:
        for k in at_k:
            g_results = eval_func[metric](ground_truth, k, sorted_gt_indices, sorted_pred_indices, g_results)
    return g_results


def verify_at_k(at_k):
    if type(at_k) is not list:
        at_k = [at_k]
    return at_k


def get_eval_metric_func():
    # TODO: Consider adding more evaluation metrics
    eval_metric_func = {
        'ndcg': compute_ndcg,
        'hits': compute_hits,
        'MAE': compute_MAE
    }
    return Evaluation.eval_metrics, eval_metric_func


def get_sorted_indices(score_tensor, comb_size_indices):
    sorted_indices = torch.argsort(score_tensor, descending=True)
    return [idx.item() for idx in sorted_indices if idx.item() in comb_size_indices]


def compute_dcg(ground_truth, sorted_indices, at_k):
    DCG = 0
    for i in range(1, min(at_k + 1), len(sorted_indices)):
        DCG += (math.pow(2, ground_truth[sorted_indices[i-1]].item()) - 1) / math.log2(i+1)
    return DCG


def compute_ndcg(ground_truth, k, sorted_gt_indices, sorted_pred_indices, results):
    IDCG = compute_dcg(ground_truth, sorted_gt_indices, k)
    DCG = compute_dcg(ground_truth, sorted_pred_indices, k)
    results['ndcg'][k] = round(DCG / IDCG, 4)
    return results


def compute_hits(ground_truth, k, sorted_gt_indices, sorted_pred_indices, results):
    hits = sum([1 for i in range(min(k, len(sorted_pred_indices))) if sorted_pred_indices[i] in sorted_gt_indices[:k]])
    results['hits'][k] = round(hits / k, 4)
    return results


def compute_MAE(ground_truth, k, sorted_gt_indices, sorted_pred_indices, results):
    # TODO: Implement Mean Absolute Error
    pass


def get_comb_size_indices(num_nodes, comb_size, feature_num):
    comb_size_indices = []
    for i in range(num_nodes):
        binary_vec = convert_decimal_to_binary(i+1, feature_num)
        if binary_vec.count('1') == comb_size:
            comb_size_indices.append(i)
    return comb_size_indices


def save_results(test_results, dir_path, comb_size, args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    results_path = dir_path + f'results_comb_size={comb_size}.pkl'
    final_test_results = comp_ave_results(test_results)
    with open(results_path, 'wb') as f:
        pickle.dump(final_test_results, f)
    save_hyperparams(dir_path, args)
    return


def comp_ave_results(results_dict):
    subgroup_list = list(results_dict[1].keys())
    eval_metrics = list(results_dict[1][subgroup_list[0]].keys())
    at_k = list(results_dict[1][subgroup_list[0]][eval_metrics[0]].keys())
    final_results = {subgroup: {metric: {k: round(np.mean([results_dict[seed][subgroup][metric][k]
                                                           for seed in results_dict]), 4) for k in at_k}
                                for metric in eval_metrics} for subgroup in subgroup_list}
    return final_results


def save_hyperparams(dir_path, args):
    hyperparams = (f"Model: {args.model}\nHidden channels: {args.hidden_channels}\n"
                   f"Number of layers: {args.num_layers}\nDropout: {args.p_dropout}\nlr: {args.lr}\n"
                   f"weight_decay: {args.weight_decay}\n")
    hyperparams_path = dir_path + 'hyperparams.txt'
    with open(hyperparams_path, 'w') as f:
        f.write(hyperparams)
    return


def print_results(results, at_k, comb_size, subgroup):
    print(5 * '======================')
    print(f'Evaluation results for subgroup {subgroup} with comb_size={comb_size} and at_k={at_k}:')
    for metric in results:
        for k in at_k:
            print(f'{metric}@{k}: {results[metric][k]}')
        print(5*'-------------------')
    return


def read_paths(args):
    dataset_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/dataset.pkl"
    graph_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/dataset_hetero_graph.pt"
    # hyperparams = (f"{args.model}_hidden{args.hidden_channels}_layers{args.num_layers}_dropout"
    #                f"{args.p_dropout}_lr{args.lr}_weight_decay{args.weight_decay}")
    dir_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/"
    return dataset_path, graph_path, dir_path


def generate_info_string(args, seed):
    info_string = f"""
    Training using a GNN of the model {args.model}
    ===================================
    Hyperparameters:
        Seed: {seed}
        Hidden channels: {args.hidden_channels}
        Number of layers: {args.num_layers}
        Dropout probability: {args.p_dropout}
        Learning rate: {args.lr}
        Weight decay: {args.weight_decay}
    -----------------------------------
    Evaluation:
        Metrics: {Evaluation.eval_metrics}
        @k: {args.at_k}
        Lattice level: {args.comb_size}
    ===================================
    """
    if args.display:
        print(info_string)
    return info_string
