import argparse
from config import LatticeGeneration, MissingDataConfig, Sampling, GNN, Evaluation
from utils import *
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_node_level_dict(lattice_graph, inverse_restrict_mapping):
    node_level_dict = {}
    for node_idx in range(len(inverse_restrict_mapping)):
        node_level_dict[node_idx] = convert_decimal_to_binary(inverse_restrict_mapping[node_idx]+1,
                                                              feature_num).count('1')
    return node_level_dict


def create_inter_level_edges_dict(lattice_graph, node_level_dict, min_level, max_level):
    parent_child_edges_dict = defaultdict(list)
    intra_lattice_edges = lattice_graph.edge_index_dict[('g0', 'g0TOg0', 'g0')].t().tolist()
    for idx, (parent_node, child_node) in enumerate(intra_lattice_edges):
        parent_level = node_level_dict[parent_node]
        child_level = node_level_dict[child_node]
        if (min_level <= parent_level <= max_level and min_level <= child_level <= max_level and
                parent_level == child_level + 1):
            parent_child_edges_dict[parent_node].append(child_node)
    return parent_child_edges_dict


def evaluate_upward_closure(parent_child_edges_dict, predictions, min_level, max_level):
    tmp_results_dict = defaultdict(list)
    for parent, children in parent_child_edges_dict.items():
        parent_level = convert_decimal_to_binary(parent+1, feature_num).count('1')
        if not min_level < parent_level <= max_level:
            continue
        tmp_results_dict[parent_level].extend([1 if predictions[parent] >= predictions[child] else 0
                                               for child in children])

    results_dict = {level: round(sum(tmp_results_dict[level]) / len(tmp_results_dict[level]), 4)
                    for level in tmp_results_dict.keys()}
    return results_dict


def print_accuracy_per_level(accuracy_dict):
    for level, accuracy in sorted(accuracy_dict.items()):
        print(f'Level: {level}, Accuracy: {accuracy}')


def get_average_accuracy_per_level(accuracy_dict):
    average_accuracy = {level: round(sum([accuracy_dict[g_id][level] for g_id in subgroups]) / len(subgroups), 4)
                        for level in accuracy_dict[subgroups[0]].keys()}
    print_accuracy_per_level(average_accuracy)
    return average_accuracy


def save_average_accuracy_per_level(average_accuracy_dict, average_accuracy_path):
    with open(average_accuracy_path, 'wb') as f:
        pickle.dump(average_accuracy_dict, f)
    return


def compute_model_average_accuracy(lattice_graph, subgroups, parent_child_edges_dict, model_path,
                                   data_name, min_level, max_level):
    lattice_graph.to(device)
    accuracy_dict = dict()
    for g_id in subgroups:
        model = torch.load(model_path + f'{g_id}.pt')
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(lattice_graph.x_dict, lattice_graph.edge_index_dict)
        # to cpu
        preds = out[g_id].to('cpu').tolist()
        accuracy_dict[g_id] = evaluate_upward_closure(parent_child_edges_dict, preds, min_level, max_level)
    average_accuracy_dict = get_average_accuracy_per_level(accuracy_dict)
    average_accuracy_path = f"RealWorldData/{data_name}/model_average_upward_closure_accuracy.pkl"
    save_average_accuracy_per_level(average_accuracy_dict, average_accuracy_path)
    return


def compute_baseline_average_accuracy(baseline_lattice, subgroups, parent_child_edges_dict, baseline,
                                      data_name, min_level, max_level):
    accuracy_dict = dict()
    for g_id in subgroups:
        preds = baseline_lattice[g_id].y.to('cpu').tolist()
        accuracy_dict[g_id] = evaluate_upward_closure(parent_child_edges_dict, preds, min_level, max_level)
    average_accuracy_dict = get_average_accuracy_per_level(accuracy_dict)
    average_accuracy_path = f"RealWorldData/{data_name}/{baseline}_average_upward_closure_accuracy.pkl"
    save_average_accuracy_per_level(average_accuracy_dict, average_accuracy_path)
    return


if __name__ == "__main__":
    # Currently adjusted only for real-world data
    parser = argparse.ArgumentParser()
    parser.add_argument('--formula', type=str, default=str(LatticeGeneration.formula_idx))
    parser.add_argument('--config', type=str, default=str(LatticeGeneration.hyperparams_idx))
    parser.add_argument('--num_layers', type=int, default=GNN.num_layers)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size_list', type=int, default=Evaluation.comb_size_list)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--min_m', type=int, default=LatticeGeneration.min_m, help='min size of feature combinations')
    parser.add_argument('--max_m', type=int, default=LatticeGeneration.max_m, help='max size of feature combinations')
    parser.add_argument('--edge_sampling_ratio', type=float, default=LatticeGeneration.edge_sampling_ratio)
    parser.add_argument('--load_mapping_dict', type=bool, default=False)
    parser.add_argument('--data_name', type=str, default='loan', help='options:{synthetic, loan, startup, mobile}')
    parser.add_argument('--model_name', type=str, default="SAGE_seed1_")
    parser.add_argument('--baseline', type=str, default='mode', help='options:{KNN, mode}')

    args = parser.parse_args()

    data_name = args.data_name
    graph_path = f"RealWorldData/{data_name}/dataset_hetero_graph_edgeSamplingRatio={args.edge_sampling_ratio}.pt"
    lattice_graph = torch.load(graph_path)
    feature_num = lattice_graph['g0'].x.shape[1]
    subgroups = lattice_graph.node_types
    model_path = f"RealWorldData/{data_name}/{args.model_name}"
    min_level = get_min_level(args.min_m, args.num_layers)
    max_level = get_max_level(args.max_m, args.num_layers, feature_num)

    restricted_mapping = get_restricted_graph_idxs_mapping(feature_num, min_level, max_level)
    inverse_restrict_mapping = {v: k for k, v in restricted_mapping.items()}
    node_level_dict = get_node_level_dict(lattice_graph, inverse_restrict_mapping)
    parent_child_edges_dict = create_inter_level_edges_dict(lattice_graph, node_level_dict, min_level, max_level)
    compute_model_average_accuracy(lattice_graph, subgroups, parent_child_edges_dict, model_path,
                                   data_name, min_level, max_level)
    baseline_lattice = torch.load(f"RealWorldData/{data_name}/dataset{args.baseline}_lattice.pt")
    compute_baseline_average_accuracy(baseline_lattice, subgroups, parent_child_edges_dict, args.baseline, data_name,
                                      min_level, max_level)

