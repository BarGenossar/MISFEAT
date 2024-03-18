
class LatticeGeneration:
    formula_idx = 1
    hyperparams_idx = 1
    min_k = 1
    within_level_edges = False
    is_hetero = False
    with_edge_attrs = False


class GNN:
    gnn_model = 'SAGE'  # Can't use GAT for heterogeneous graphs. Use only SAGE for now
    hidden_channels = 64
    num_layers = 2
    p_dropout = 0
    epochs = 200


class Evaluation:
    at_k = [1, 3, 5, 10]
    comb_size = 4
    eval_metrics = ['ndcg', 'hits']

