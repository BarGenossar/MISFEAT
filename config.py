
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
    sampling_ratio = 0.5


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


# TODO: Important: Add multiprocessing to lattice generator to compute the mutual information.
# TODO: Think about how to create formulas / play with parameters so that there aren’t many groups that have high mutual
#  information. Obviously in notebook.
# TODO: Check if mutual information can be computed on the GPU.
# TODO: Think about how we can add noise for different subgroups in the synthetic data.
