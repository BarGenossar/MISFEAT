import math
from config import Evaluation


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


def read_paths(args):
    dataset_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/dataset.pkl"
    graph_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/dataset_hetero_graph.pt"
    hyperparams = (f"{args.model}_hidden{args.hidden_channels}_layers{args.num_layers}_dropout"
                   f"{args.p_dropout}_lr{args.lr}_weight_decay{args.weight_decay}")
    dir_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/{hyperparams}/"
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
