
import argparse
import pandas as pd
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling
from missing_data_masking import MissingDataMasking
from random_graph_generator import GraphSampling
from sampler import NodeSampler
from lattice_graph_generator import FeatureLatticeGraph
from utils import *
from torch_geometric.nn import to_hetero
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data, train_indices, model, g_id, optimizer, criterion):
    data.to(device)
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    labels = data[g_id].y[train_indices]
    predictions = out[g_id][train_indices]
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(data, test_indices, model, g_id, at_k, comb_size, feature_num):
    data.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    labels = data[g_id].y
    predictions = out[g_id]
    tmp_results_dict = compute_eval_metrics(labels, predictions, at_k, comb_size, feature_num)
    print_results(tmp_results_dict, at_k, comb_size, g_id)
    return tmp_results_dict


def initialize_model_and_optimizer(args):
    model = LatticeGNN(args.model, feature_num, args.hidden_channels, seed, args.num_layers, args.p_dropout)
    model = to_hetero(model, data.metadata())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds_num', type=int, default=1)
    parser.add_argument('--formula', type=str, default=str(LatticeGeneration.formula_idx))
    parser.add_argument('--config', type=str, default=str(LatticeGeneration.hyperparams_idx))
    parser.add_argument('--model', type=str, default=GNN.gnn_model)
    parser.add_argument('--hidden_channels', type=int, default=GNN.hidden_channels)
    parser.add_argument('--num_layers', type=int, default=GNN.num_layers)
    parser.add_argument('--p_dropout', type=float, default=GNN.p_dropout)
    parser.add_argument('--epochs', type=int, default=GNN.epochs)
    parser.add_argument('--sampling_ratio', type=float, default=0.33)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size', type=int, default=Evaluation.comb_size)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--data_path', default='GeneratedData/Formula2/Config2/dataset.pkl', type=str, help='Path to the dataset including the .pkl prefix')
    parser.add_argument('--workers', default=8, type=int, help="number of parallel workers")
    args = parser.parse_args()

    config_idx = args.config
    seeds_num = args.seeds_num
    comb_size = args.comb_size
    epochs = args.epochs
    manual = args.manual_md
    at_k = verify_at_k(args.at_k)

    # if args.data_path:
    #     dataset_path = read_paths(args, args.data_path)
    # else:
    # dataset_path, graph_path, dir_path = read_paths(args)
    # feature_num = read_feature_num_from_txt(dataset_path)
    
    ## load dataset
    df = pd.read_pickle(args.data_path)
    print(df)
    exit()
    base_features = [feat for feat in list(df.columns) if 'f_' in feat]
    feature_num = len(base_features)
    subgroups = [f'g{gid}' for gid in range(df.subgroup.nunique())]
    results_dict = {seed: {subgroup: dict() for subgroup in subgroups} for seed in range(1, seeds_num + 1)}

    graph_path = "GeneratedData/Formula2/Config2/dataset_hetero_graph.pt"
    data = torch.load(graph_path)


    # graph = GraphSampling(
    #     df,
    #     missing_indices_dict,
    #     args.sampling_ratio,
    #     'uniform',
    #     min_k=1,
    #     num_workers=args.workers,
    #     edge_threshold=5,
    # )



    for seed in range(1, seeds_num + 1):
        info_string = generate_info_string(args, seed)
        # torch.manual_seed(seed)
        set_seed(seed)

        ## sample missing features, train_indices
        missing_indices_dict = MissingDataMasking(feature_num, subgroups, config_idx, manual).missing_indices_dict
        sampler = NodeSampler(subgroups, feature_num, missing_indices_dict, args.sampling_ratio, sampling_method='uniform')
        sampled_indices = sampler.get_selected_samples()
        ## QUESTION: for the test_indices, should we also include the rest of the training nodes? (the ones that were not sampled for training)

        ## init model
        criterion = torch.nn.MSELoss()
        loss_vals = {subgroup: [] for subgroup in subgroups}
        for subgroup in subgroups:
            print(f"\nTraining on subgroup {subgroup}...")
            model, optimizer = initialize_model_and_optimizer(args)
            # train_indices = [idx for idx in range(data[subgroup].num_nodes) if idx not in
            #                     missing_indices_dict[subgroup]['all']]
            train_indices = sampled_indices[subgroup]
            test_indices = missing_indices_dict[subgroup]['all']
            for epoch in range(1, epochs + 1):
                loss_val = train(data, train_indices, model, subgroup, optimizer, criterion)
                loss_vals[subgroup].append(loss_val)
                if epoch == 1 or epoch % 5 == 0:
                    continue
                    # print(f'Epoch: {epoch}, Loss: {round(loss_val, 4)}')
            results_dict[seed][subgroup] = test(data, test_indices, model, subgroup, at_k, comb_size, feature_num)
    # save_results(results_dict, dir_path, comb_size, args)
