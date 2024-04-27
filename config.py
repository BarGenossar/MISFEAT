
class LatticeGeneration:
    formula_idx = 2
    hyperparams_idx = 1
    min_m = 3
    max_m = 5
    within_level_edges = False
    is_hetero = False
    with_edge_attrs = False
    num_workers = 5  # Number of workers for multiprocessing
    edge_sampling_ratio = 0.5


class MissingDataConfig:
    missing_prob = 0.1
    max_missing_ratio = 0.6
    missing_rate_dict = {'relevant': 0.2, 'correlated': 0.2, 'redundant': 0.2, 'noisy': 0.2}


class Sampling:
    method = 'randwalk'
    sampling_ratio = 1.0
    validation_ratio = 0.2


class GNN:
    gnn_model = 'SAGE'  # Can't use GAT for heterogeneous graphs. Use only SAGE for now
    hidden_channels = 64
    num_layers = 2
    p_dropout = 0
    epochs = 300
    epochs_stable_val = 5


class MLP:
    hidden_channels = 64
    num_layers = 2
    p_dropout = 0
    epochs = 300
    epochs_stable_val = 5
    MLP_results_dir = 'MLP_results/'


class Evaluation:
    at_k = [3, 5, 10]
    comb_size_list = [3, 4, 5]
    eval_metrics = ['NDCG', 'PREC']
    formula_idx_list = [0, 1, 2, 3, 4]
    config_idx_list = [1, 3]
