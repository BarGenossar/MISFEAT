
class LatticeGeneration:
    formula_idx = 1
    hyperparams_idx = 1
    min_k = 1
    within_level_edges = False
    is_hetero = False
    with_edge_attrs = False


class MissingDataConfig:
    general_missing_prob = 0.2
    missing_rate_dict = {'relevant': 0.2, 'correlated': 0.2, 'redundant': 0.2, 'noisy': 0.2}


class GNN:
    gnn_model = 'SAGE'  # Can't use GAT for heterogeneous graphs. Use only SAGE for now
    hidden_channels = 64
    num_layers = 2
    p_dropout = 0
    epochs = 300


class Evaluation:
    at_k = [3, 5, 10]
    comb_size = 4
    eval_metrics = ['ndcg', 'hits']

