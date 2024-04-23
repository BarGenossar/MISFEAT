import os
import time
import json
import torch
import argparse
import numpy as np
from sampler import NodeSampler
from GNN_models import LatticeGNN
from torch_geometric.nn import to_hetero
from missing_data_masking import MissingDataMasking
from config import LatticeGeneration, GNN, Sampling, Evaluation, MissingDataConfig, MLP
from sklearn.model_selection import train_test_split
from utils import compute_eval_metrics, print_results, get_comb_size_indices, set_seed, convert_decimal_to_binary
import warnings
warnings.filterwarnings("ignore")
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_input_vectors(indices, feature_num):
    x_input = []
    for node_id in indices:
        binary_vec = convert_decimal_to_binary(node_id + 1, feature_num)
        binary_vec = [int(digit) for digit in binary_vec]
        x_input.append(binary_vec)
    return torch.tensor(x_input, dtype=torch.float32).to(device)

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p_dropout):
        super(MLPModel, self).__init__()
        self.relu = torch.nn.ReLU()
        self.p_dropout = p_dropout
        self.num_layers = num_layers
        self._set_layers(input_size, hidden_size)
        self.out = Linear(hidden_size, 1)

    def _set_layers(self, input_size, hidden_size):
        for layer_idx in range(1, self.num_layers + 1):
            if layer_idx == 1:
                setattr(self, f'fc{layer_idx}', Linear(input_size, hidden_size))
            else:
                setattr(self, f'fc{layer_idx}', Linear(hidden_size, hidden_size))
        return

    def forward(self, x):
        for layer_idx in range(1, self.num_layers + 1):
            fc = getattr(self, f'fc{layer_idx}')
            x = self.relu(fc(x))
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        output = self.out(x).squeeze()
        return output



class PipelineManager:
    def __init__(self, args, lattice_graph):
        self.args = args
        self.multiplier = 200    # best (so far):  loan 400,  mobile 50-100 , startup 50
        self.at_k = [args.at_k] if not isinstance(args.at_k, list) else args.at_k
        self.dir_path = args.dir_path
        self.lattice_graph = lattice_graph
        self.subgroups = lattice_graph.x_dict.keys()
        self.feature_num = int(np.log2(len(self.lattice_graph['g0']['x']) + 1))
        self.missing_indices_dict = self._get_missing_data_dict()
        self.missing_features_dict = {subgroup: list(self.missing_indices_dict[subgroup].keys())[:-1] for subgroup in self.subgroups}
        self.test_indices = self._get_test_indices()
        self.train_idxs_dict, self.valid_idxs_dict = self._train_validation_split()


    def _get_missing_data_dict(self):
        return MissingDataMasking(
            self.feature_num,
            self.subgroups,
            self.args.manual_md,
        ).missing_indices_dict


    def _init_model_optim(self):
        model = MLPModel(self.feature_num, self.args.hidden_channels, self.args.num_layers, self.args.p_dropout)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer


    def _train_validation_split(self):
        train_idxs_dict, valid_idxs_dict = dict(), dict()
        sampler = NodeSampler(
            self.subgroups,
            self.feature_num,
            self.missing_indices_dict,
            self.args.sampling_ratio,
            self.args.sampling_method,
        ) 
        for g_id in self.subgroups:
            if self.args.valid_ratio == 0:
                train_idxs_dict[g_id] = sampler.selected_samples[g_id]
                valid_idxs_dict[g_id] = self.test_indices[g_id]
            else:
                train_idxs_dict[g_id], valid_idxs_dict[g_id] = train_test_split(sampler.selected_samples[g_id],
                                                                                test_size=self.args.valid_ratio)
        return train_idxs_dict, valid_idxs_dict


    def _get_test_indices(self):
        return {subgroup: self.missing_indices_dict[subgroup]['all'] for subgroup in self.subgroups}


    def test_subgroup(self, subgroup, comb_size, show_results=False):
        test_indices = get_comb_size_indices(self.test_indices[subgroup], comb_size, self.feature_num)
        if len(test_indices) == 0: return ['this subgroup does not have missing feature']
        x_test = get_input_vectors(test_indices, pipeline_obj.feature_num)
        lattice_graph = pipeline_obj.lattice_graph
        lattice_graph.to(device)
        model = torch.load(f"{self.args.dir_path}/{'MLP_model'}_seed{seed}_{subgroup}.pt")
        model.to(device)
        model.eval()
        with torch.no_grad():
            preds = model(x_test)
        labels = lattice_graph[subgroup].y[test_indices]

        tmp_results_dict = compute_eval_metrics(labels, preds, self.at_k, comb_size, self.feature_num)
        if show_results:
            print_results(tmp_results_dict, self.at_k, comb_size, subgroup)
        return tmp_results_dict


    def _run_training_epoch(self, train_indices, x_train, model, subgroup, optimizer, criterion, lattice_graph):
        model.train()
        optimizer.zero_grad()
        preds = model(x_train)
        labels = lattice_graph[subgroup].y[train_indices]
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()


    def get_validation_loss(self, lattice_graph, validation_indices, x_valid, model, subgroup, criterion):
        model.eval()
        model.to(device)
        with torch.no_grad():
            preds = model(x_valid)
        labels = lattice_graph[subgroup].y[validation_indices]
        loss = criterion(preds, labels)
        return loss.item()


    def run_over_validation(self, lattice_graph, validation_indices, x_valid, model, subgroup, criterion,
                        best_val, no_impr_counter, seed, dir_path):
        loss_validation = self.get_validation_loss(lattice_graph, validation_indices, x_valid, model, subgroup, criterion)
        if loss_validation < best_val:
            best_val = loss_validation
            no_impr_counter = 0
            torch.save(model, f"{dir_path}/{'MLP_model'}_seed{seed}_{subgroup}.pt")
        else:
            no_impr_counter += 1
        return best_val, no_impr_counter



    def train_model(self, seed):
        torch.manual_seed(seed)
        criterion = torch.nn.MSELoss()
        lattice_graph = pipeline_obj.lattice_graph
        lattice_graph.to(device)
        dir_path = pipeline_obj.dir_path
        feature_num = pipeline_obj.feature_num
        input_size = feature_num
        for subgroup in subgroups:
            print(f"\nTraining on subgroup {subgroup}...")
            model = MLPModel(input_size, self.args.hidden_channels, self.args.num_layers, self.args.p_dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            train_indices = pipeline_obj.train_idxs_dict[subgroup]
            valid_indices = pipeline_obj.valid_idxs_dict[subgroup]
            x_train = get_input_vectors(train_indices, feature_num)
            x_valid = get_input_vectors(valid_indices, feature_num)
            no_impr_counter = 0
            epochs_stable_val = MLP.epochs_stable_val
            best_val = float('inf')
            for epoch in range(args.epochs):
                if no_impr_counter == epochs_stable_val:
                    break
                loss_value = self._run_training_epoch(train_indices, x_train, model, subgroup, optimizer, criterion, lattice_graph)
                if epoch == 1 or epoch % 5 == 0:
                    print(f'Epoch: {epoch}, Loss: {round(loss_value, 4)}')
                    if not args.save_model:
                        continue
                    best_val, no_impr_counter = self.run_over_validation(lattice_graph, valid_indices, x_valid, model, subgroup,
                                                                    criterion, best_val, no_impr_counter, seed, dir_path)

def average_results(results_dict):
    num_seeds = len(results_dict[3])
    seeds = list(results_dict[3].keys())

    subgroups = results_dict[3][seeds[0]].keys()
    aggr = {comb_size: {subgroup: {'NDCG': {3: 0., 5: 0., 10: 0.}, 'PREC': {3: 0., 5: 0., 10: 0.}} for subgroup in subgroups} for comb_size in [3, 4, 5]}
    for comb_size in aggr:
        for subgroup in subgroups:
            for seed in results_dict[comb_size]:
                for at_k in [3, 5, 10]:
                    # if isinstance(results_dict[comb_size][seed][subgroup], dict):
                    aggr[comb_size][subgroup]['NDCG'][at_k] += 1/num_seeds * results_dict[comb_size][seed][subgroup]['NDCG'][at_k]
                    aggr[comb_size][subgroup]['PREC'][at_k] += 1/num_seeds * results_dict[comb_size][seed][subgroup]['PREC'][at_k]

    for comb_size in aggr:
        print('comb size =', comb_size)
        for at_k in [3, 5, 10]:
            print(f"\tnDCG @ {at_k}")
            avg_subgroup = 0
            for subgroup in subgroups:
                avg_subgroup += aggr[comb_size][subgroup]['NDCG'][at_k]
                print(f"\t\tsubgroup: {subgroup} = {round(aggr[comb_size][subgroup]['NDCG'][at_k], 2)}")
            print(f"\t\taverage: {avg_subgroup/len(subgroups)}")

        for at_k in [3, 5, 10]:
            print(f"\tprecision @ {at_k}")
            avg_subgroup = 0
            for subgroup in subgroups:
                avg_subgroup += aggr[comb_size][subgroup]['PREC'][at_k]
                print(f"\t\tsubgroup: {subgroup} = {round(aggr[comb_size][subgroup]['PREC'][at_k], 2)}")
            print(f"\t\taverage: {avg_subgroup/len(subgroups)}")
    return aggr



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
    parser.add_argument('--sampling_ratio', type=float, default=Sampling.sampling_ratio)
    parser.add_argument('--sampling_method', type=str, default=Sampling.method)
    parser.add_argument('--valid_ratio', type=str, default=Sampling.validation_ratio)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size_list', type=int, default=Evaluation.comb_size_list)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--dir_path', type=str, default=None, help='path to the directory file')
    args = parser.parse_args()

    ## load input data
    lattice_graph = torch.load(f'{args.dir_path}/dataset_hetero_graph.pt')
    subgroups = lattice_graph.x_dict.keys()

    ## store results
    seeds = range(32, 32 + args.seeds_num)
    # seeds = [1, 2, 3]
    results_dict = {comb_size: {seed: {subgroup: dict() for subgroup in subgroups}
                                for seed in seeds} for comb_size in args.comb_size_list}
    for seed in seeds:
        print(f"Seed: {seed}\n=============================")
        set_seed(seed)

        pipeline_obj = PipelineManager(args, lattice_graph)
        pipeline_obj.train_model(seed)

        for comb_size in args.comb_size_list:
            results_dict[comb_size][seed] = {g_id: pipeline_obj.test_subgroup(g_id, comb_size) for g_id in subgroups}


    avg_res = average_results(results_dict)


    