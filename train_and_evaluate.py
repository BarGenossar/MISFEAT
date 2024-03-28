
import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN
from missing_data_masking import MissingDataMasking
from utils import *
from torch_geometric.nn import to_hetero
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(lattice_graph, train_indices, model, g_id, optimizer, criterion):
    lattice_graph.to(device)
    model.train()
    optimizer.zero_grad()
    out = model(lattice_graph.x_dict, lattice_graph.edge_index_dict)
    labels = lattice_graph[g_id].y[train_indices]
    predictions = out[g_id][train_indices]
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
    tmp_results_dict = compute_eval_metrics(labels, predictions, at_k, comb_size, feature_num)
    print_results(tmp_results_dict, at_k, comb_size, g_id)
    return tmp_results_dict


def initialize_model_and_optimizer(args):
    model = LatticeGNN(args.model, feature_num, args.hidden_channels, seed, args.num_layers, args.p_dropout)
    model = to_hetero(model, lattice_graph.metadata())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


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

    config_idx = args.config
    seeds_num = args.seeds_num
    comb_size = args.comb_size
    epochs = args.epochs
    at_k = verify_at_k(args.at_k)

    dataset_path, graph_path, dir_path = read_paths(args)
    feature_num = read_feature_num_from_txt(dataset_path)

    lattice_graph = torch.load(graph_path)
    subgroups = lattice_graph.x_dict.keys()
    missing_indices_dict = MissingDataMasking(feature_num, subgroups, config_idx).missing_indices_dict
    results_dict = {seed: {subgroup: dict() for subgroup in subgroups} for seed in range(1, seeds_num + 1)}
    for seed in range(1, seeds_num + 1):
        info_string = generate_info_string(args, seed)
        torch.manual_seed(seed)
        criterion = torch.nn.MSELoss()
        loss_vals = {subgroup: [] for subgroup in subgroups}
        for subgroup in subgroups:
            print(f"\nTraining on subgroup {subgroup}...")
            model, optimizer = initialize_model_and_optimizer(args)
            train_indices = [idx for idx in range(lattice_graph[subgroup].num_nodes) if idx not in
                             missing_indices_dict[subgroup]['all']]
            for epoch in range(1, epochs + 1):
                loss_val = train(lattice_graph, train_indices, model, subgroup, optimizer, criterion)
                loss_vals[subgroup].append(loss_val)
                if epoch == 1 or epoch % 5 == 0:
                    continue
                    # print(f'Epoch: {epoch}, Loss: {round(loss_val, 4)}')
            results_dict[seed][subgroup] = test(lattice_graph, model, subgroup, at_k, comb_size, feature_num)
    save_results(results_dict, dir_path, comb_size, args)
