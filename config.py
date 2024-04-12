
class LatticeGeneration:
    formula_idx = 2
    hyperparams_idx = 2
    min_k = 1
    within_level_edges = False
    is_hetero = False
    with_edge_attrs = False
    num_workers = 5  # Number of workers for multiprocessing


class MissingDataConfig:
    general_missing_prob = 0.1
    missing_rate_dict = {'relevant': 0.2, 'correlated': 0.2, 'redundant': 0.2, 'noisy': 0.2}


class Sampling:
    method= 'random'
    sampling_ratio = 0.75
    validation_ratio = 0.2


class GNN:
    gnn_model = 'SAGE'  # Can't use GAT for heterogeneous graphs. Use only SAGE for now
    hidden_channels = 64
    num_layers = 2
    p_dropout = 0
    epochs = 500
    epochs_stable_val = 5


class Evaluation:
    at_k = [3, 5, 10]
    comb_size_list = [3, 4, 5]
    eval_metrics = ['NDCG', 'PREC', 'RMSE']
