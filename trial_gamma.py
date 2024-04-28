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
from config import LatticeGeneration, GNN, Sampling, Evaluation, MissingDataConfig
from sklearn.model_selection import train_test_split
from utils import compute_eval_metrics, print_results, get_comb_size_indices, set_seed, convert_decimal_to_comb
from torchviz import make_dot
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')


class PipelineManager:
    def __init__(self, args, lattice_graph):
        self.args = args
        self.gamma = torch.autograd.Variable(torch.tensor([2000.0]), requires_grad=True).to(device)
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
        model = LatticeGNN(self.args.model, self.feature_num, self.args.hidden_channels, self.args.num_layers, self.args.p_dropout)
        model = to_hetero(model, self.lattice_graph.metadata()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
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
        self.lattice_graph.to(device)
        model = torch.load(f"{self.dir_path}/{self.args.model}_seed{seed}_ratio{self.args.sampling_ratio}_missing{MissingDataConfig.general_missing_prob}.pt")
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        mi_true = self.lattice_graph[subgroup].y[test_indices] * self.gamma
        mi_pred = out[subgroup][test_indices]

        ### print the true/pred combinations
        ###############################
        rank_true = np.argsort(mi_true.cpu().tolist())[::-1]
        rank_pred = np.argsort(mi_pred.cpu().tolist())[::-1]
        if comb_size == 5 :
            print(subgroup)
            for r in range(5):
                true_comb = convert_decimal_to_comb(test_indices[rank_true[r]]+1, self.feature_num)
                pred_comb = convert_decimal_to_comb(test_indices[rank_pred[r]]+1, self.feature_num)
                print(f"top {r+1}: true: {true_comb}, pred: {pred_comb}")
        ###############################

        tmp_results_dict = compute_eval_metrics(mi_true, mi_pred, self.at_k, comb_size, self.feature_num)
        if show_results:
            print_results(tmp_results_dict, self.at_k, comb_size, subgroup)
        return tmp_results_dict


    def _run_training_epoch(self, model, optimizer, criterion, update_gamma=False):
        model.train()
        optimizer.zero_grad()
        out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        # num_missing = [1 / (1+len(feats)) for feats in self.missing_features_dict.values() ]
        grads = {}
        grad_gamma = 0.
        epoch_loss = 0
        for gid, subgroup in enumerate(self.subgroups):
            train_indices = self.train_idxs_dict[subgroup]
            mi_true = self.lattice_graph[subgroup].y[train_indices] * self.gamma
            mi_pred = out[subgroup][train_indices]

            ## accumulate gradients across subgroups
            if update_gamma:
                loss = torch.sum((mi_true - mi_pred) ** 2)/len(mi_true)
                grads = torch.autograd.grad(loss, self.gamma, torch.ones_like(loss))
                grad_gamma += grads[0]
            else:
                loss = criterion(mi_pred, mi_true)
                loss.backward(retain_graph=True)

                for n, p in model.named_parameters():
                    if p.grad is not None:
                        try:
                            grads[n] += p.grad# * num_missing[gid]/sum(num_missing)
                        except KeyError:
                            grads[n] = p.grad# * num_missing[gid]/sum(num_missing)
            epoch_loss += loss.item()

        if update_gamma:
            self.gamma = self.gamma - self.args.lr * grad_gamma
        else:
            for n, p in model.named_parameters():
                if grads[n] is not None:
                    p.data -= self.args.lr * grads[n]
        return epoch_loss / len(self.subgroups)


    def _get_val_pred(self, val_indices, model, subgroup):
        model.eval()
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        mi_true = self.lattice_graph[subgroup].y[val_indices] * self.gamma
        mi_pred = out[subgroup][val_indices]
        # for a, b in zip(mi_pred, mi_true):
        #     print(subgroup, round(a.item(), 5), round(b.item(), 5))
        # print(mi_pred, mi_true)
        return mi_true.tolist(), mi_pred.tolist()


    def _run_over_validation(self, model, criterion, best_mse, no_impr_counter, seed):
        all_mi_true = []
        all_mi_pred = []
        for subgroup in self.subgroups:
            val_indices = self.valid_idxs_dict[subgroup]
            mi_true, mi_pred = self._get_val_pred(val_indices, model, subgroup)
            all_mi_true.extend(mi_true)
            all_mi_pred.extend(mi_pred)
            mse = criterion(torch.tensor(all_mi_pred), torch.tensor(all_mi_true))
            if mse < best_mse:
                best_mse = mse
                no_impr_counter = 0
                torch.save(model, f"{self.dir_path}/{self.args.model}_seed{seed}_ratio{self.args.sampling_ratio}_missing{MissingDataConfig.general_missing_prob}.pt")
        else:
            no_impr_counter += 1
        return mse, best_mse, no_impr_counter



    def train_model(self, seed):
        criterion = torch.nn.MSELoss()
        self.lattice_graph.to(device)

        model, optimizer = self._init_model_optim()

        no_impr_counter = 0
        best_mse = float('inf')
        update_gamma = False
        for epoch in range(self.args.epochs):
            if no_impr_counter == GNN.epochs_stable_val:
                break
            update_gamma = True if (epoch + 1) % 5 == 0 else False
            train_loss = self._run_training_epoch(model, optimizer, criterion, update_gamma)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                mse, best_mse, no_impr_counter = self._run_over_validation(model, criterion, best_mse, no_impr_counter, seed)
                print(f"Epoch: {epoch+1}, train loss = {train_loss:.4f}, val loss: {mse:.4f}, best val loss: {best_mse:.4f}")
    


def average_results(results_dict):
    num_seeds = len(results_dict[5])
    seeds = list(results_dict[5].keys())

    subgroups = results_dict[5][seeds[0]].keys()
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
                # print(f"\t\tsubgroup: {subgroup} = {round(aggr[comb_size][subgroup]['NDCG'][at_k], 2)}")
            print(f"\t\taverage: {avg_subgroup/len(subgroups)}")

        for at_k in [3, 5, 10]:
            print(f"\tprecision @ {at_k}")
            avg_subgroup = 0
            for subgroup in subgroups:
                avg_subgroup += aggr[comb_size][subgroup]['PREC'][at_k]
                # print(f"\t\tsubgroup: {subgroup} = {round(aggr[comb_size][subgroup]['PREC'][at_k], 2)}")
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
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--missing_prob', type=float, default=MissingDataConfig.general_missing_prob, help='missing probability of a base feature')
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--dir_path', type=str, default='RealWorldData/loan-trial', help='path to the directory file')
    args = parser.parse_args()

    ## load input data
    lattice_graph = torch.load(f'{args.dir_path}/dataset_hetero_graph.pt')
    subgroups = lattice_graph.x_dict.keys()

    ## store results
    # seeds = range(32, 32 + args.seeds_num)
    seeds = [1, 2, 3]
    results_dict = {comb_size: {seed: {subgroup: dict() for subgroup in subgroups}
                                for seed in seeds} for comb_size in args.comb_size_list}
    for seed in seeds:
        print(f"Seed: {seed}\n=============================")
        set_seed(seed)

        pipeline_obj = PipelineManager(args, lattice_graph)
        # start_time of training
        pipeline_obj.train_model(seed)
        # end_time of training
        ## number of features (self.feature_num), number of records

        for comb_size in args.comb_size_list:
            results_dict[comb_size][seed] = {g_id: pipeline_obj.test_subgroup(g_id, comb_size) for g_id in subgroups}

        ## save {subgroup: [list of missing features]}
        with open(f'{args.dir_path}/missing_seed{seed}_{args.missing_prob}.json', 'w') as f:
            json.dump(pipeline_obj.missing_features_dict, f)


    avg_res = average_results(results_dict)

    with open(f'{args.dir_path}/results_dict_gnn_grads_ratio{args.sampling_ratio}_missing{MissingDataConfig.general_missing_prob}.json', 'w') as f:
        json.dump(avg_res, f)

    
    # python train.py --dir_path=RealWorldData/startup --epochs=500 --sampling_ratio=1.0
    ### change sampling ratio to 0.25, 0.5, 0.75


    