import os
import math
import torch
import random
import pickle
import typing as t
import numpy as np
from config import Evaluation
from sklearn.metrics import ndcg_score


def Kendall_tau(rank_true: t.List[int], rank_pred: t.List[int]) -> int:
    r"""
    Compute Kendall tau score: count the number of inversions between two ranked lists
    Arguments:
        `rank_true`: ground truth rank
        `rank_pred`: predicted rank
        `descending`: rank in descending order, or vice versa
    Return:
        Kendall tau score (int)
    Example:
        truth = [A, B, C, D]
        pred  = [C, A, B, D]
        inversions: BC - CB, AC - CA --> output = 2
    """
    n = len(rank_true)

    pred_to_idx = {rank: idx for idx, rank in enumerate(rank_pred)}

    ## iterate each pair and count inversions
    inversions = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if pred_to_idx[rank_true[i]] > pred_to_idx[rank_true[j]]: inversions += 1
    return inversions


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


def get_min_level(min_m, num_layers):
    return max(1, min_m-num_layers)


def get_max_level(max_m, num_layers, feature_num):
    return min(feature_num, max_m+num_layers)


def get_lattice_nodes_num(feature_num, min_level, max_level):
    return sum([math.comb(feature_num, i) for i in range(min_level, max_level + 1)])


def get_restricted_graph_idxs_mapping(feature_num=None, min_m=None, max_m=None):
    binary_vecs = [convert_decimal_to_binary(i + 1, feature_num) for i in range(2 ** feature_num - 1)]
    rel_nid = [i for i, binary_vec in enumerate(binary_vecs) if min_m <= binary_vec.count('1') <= max_m]
    return {orig_nid: new_nid for new_nid, orig_nid in enumerate(rel_nid)}


def get_lattice_edges_num(feature_num, max_subset_size, within_levels=True):
    edges_num = sum([math.comb(feature_num, i) * i for i in range(2, max_subset_size+1)])
    if within_levels:
        edges_num += sum([math.comb(feature_num, i) * i * (feature_num - i) for i
                          in range(2, max_subset_size)])
    return edges_num


def read_feature_num_from_txt(dataset_path):
    description_path = dataset_path.replace('dataset.pkl', 'description.txt')
    description = open(description_path, 'r').readlines()
    for line in description:
        if 'feature_num' in line:
            return int(line.split(':')[-1].strip())
    return None


def compute_eval_metrics(ground_truth, preds, test_indices, at_k, comb_size, feature_num):
    eval_metrics, eval_func = get_eval_metric_func()
    comb_size_indices = get_comb_size_indices(test_indices, comb_size, feature_num)
    sorted_gt_indices = get_sorted_indices(ground_truth, comb_size_indices)
    sorted_pred_indices = get_sorted_indices(preds, comb_size_indices)
    g_results = {metric: dict() for metric in eval_metrics}
    for metric in eval_metrics:
        for k in at_k:
            g_results = eval_func[metric](ground_truth, preds, k, sorted_gt_indices, sorted_pred_indices, g_results)
    return g_results


# def verify_at_k(at_k):
#     if type(at_k) is not list:
#         at_k = [at_k]
#     return at_k


def get_eval_metric_func():
    # TODO: Consider adding more evaluation metrics
    eval_metric_func = {
        'NDCG': compute_ndcg,
        'PREC': compute_precision,
        'RMSE': compute_RMSE
    }
    return Evaluation.eval_metrics, eval_metric_func


def get_sorted_indices(score_tensor, comb_size_indices):
    sorted_indices = torch.argsort(score_tensor, descending=True)
    return [idx.item() for idx in sorted_indices if idx.item() in comb_size_indices]


#### Bar's version
# def compute_dcg(ground_truth, sorted_indices, at_k):
#     DCG = 0
#     for i in range(1, min(at_k + 1, len(sorted_indices))):
#         DCG += ground_truth[sorted_indices[i-1]].item() / math.log2(i+1)
#         # DCG += (math.pow(2, ground_truth[sorted_indices[i-1]].item()) - 1) / math.log2(i+1)
#     return DCG


# def compute_ndcg(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices, results):
#     IDCG = compute_dcg(ground_truth, sorted_gt_indices, k)
#     DCG = compute_dcg(ground_truth, sorted_pred_indices, k)
#     results['NDCG'][k] = round(DCG / IDCG, 4)
#     return results


#### sklearn version
# def compute_ndcg(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices, results):
#     relevance = [0] * len(sorted_gt_indices)
#     for i in range(k): relevance[i] = 1
#     relevance_pred = [1 if idx in sorted_gt_indices[:k] else 0 for idx in sorted_pred_indices]
#     results['NDCG'][k] = ndcg_score(np.asarray([relevance]), np.asarray([relevance_pred]))


### Bar's after Thinh's version
def compute_ndcg(ground_truth, predictions, k, sorted_list_idx_gt, sorted_list_idx_pred, results):
    cands_num = len(sorted_list_idx_gt)
    relevance = [0 for _ in range(len(ground_truth))]
    for i in range(cands_num):
        relevance[sorted_list_idx_gt[i]] = cands_num - i
    DCG, IDCG = 0, 0
    for i in range(k):
        IDCG += relevance[sorted_list_idx_gt[i]] / math.log(i + 2, 2)
        DCG += relevance[sorted_list_idx_pred[i]] / math.log(i + 2, 2)
    results['NDCG'][k] = round(DCG / IDCG, 4)
    return results

### Thinh's version
def compute_ndcg_thin(ground_truth, predictions, k, sorted_list_idx_true, sorted_list_idx_pred, results):
    relevance = [0] * len(ground_truth)
    for i in range(k): relevance[sorted_list_idx_true[i]] = k - i
    # for i in range(k): relevance[sorted_list_idx_true[i]] = 1
    DCG = 0.
    IDCG = 0.
    for i in range(k):
        IDCG += (k - i) / math.log(i + 2, 2)
        # IDCG += 1 / math.log(i + 2, 2)
        DCG += relevance[sorted_list_idx_pred[i]] / math.log(i + 2, 2)
    results['NDCG'][k] = round(DCG / IDCG, 4)
    return results


def compute_precision(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices, results):
    precision = len(set.intersection(set(sorted_gt_indices[:k]), set(sorted_pred_indices[:k])))
    results['PREC'][k] = round(precision / k, 4)
    return results


def compute_RMSE(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices, results):
    # Implement normalized MAE, such that the difference is divided by the maximum possible difference
    rmse = sum([(ground_truth[sorted_gt_indices[i]] -
                   predictions[sorted_gt_indices[i]])**2 for i in range(k)])
    results['RMSE'][k] = round(math.sqrt(rmse.item() / k), 4)
    return results


def get_comb_size_indices(node_ids, comb_size, feature_num):
    comb_size_indices = []
    for node_id in node_ids:
        binary_vec = convert_decimal_to_binary(node_id + 1, feature_num)
        if binary_vec.count('1') == comb_size:
            comb_size_indices.append(node_id)
    return comb_size_indices


def save_results(test_results, dir_path, args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for comb_size in args.comb_size_list:
        results_path = dir_path + (f'combSize={comb_size}_samplingRatio={args.sampling_ratio}_'
                                   f'missingRatio={args.missing_prob}_samplingMethod={args.sampling_method}_'
                                   f'edgeSamplingRatio={args.edge_sampling_ratio}_model={args.model}.pkl')
        final_test_results = comp_ave_results(test_results[comb_size])
        with open(results_path, 'wb') as f:
            pickle.dump(final_test_results, f)
    save_hyperparams(dir_path, args)
    return


def comp_ave_results(results_dict):
    for seed in results_dict.keys():
        idx = seed  # Just to get the first seed in case the seed is not 1
        break
    subgroup_list = list(results_dict[idx].keys())
    eval_metrics = list(results_dict[idx][subgroup_list[0]].keys())
    at_k = list(results_dict[idx][subgroup_list[0]][eval_metrics[0]].keys())
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


def print_results(results, at_k, comb_size, subgroup):
    print(5 * '======================')
    print(f'Evaluation results for subgroup {subgroup} with comb_size={comb_size} and at_k={at_k}:')
    for metric in results:
        for k in at_k:
            print(f'{metric}@{k}: {results[metric][k]}')
        print(5*'-------------------')


def read_paths(args):
    if args.data_name == 'synthetic':
        dir_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/"
    else:
        dir_path = f"RealWorldData/{args.data_name}/"
    graph_path = os.path.join(dir_path, f'dataset_hetero_graph_edgeSamplingRatio={args.edge_sampling_ratio}.pt')
    return graph_path, dir_path


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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
