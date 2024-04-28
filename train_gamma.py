import torch
import pickle
import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling, MissingDataConfig
from missing_data_masking import MissingDataMasking
from sampler import NodeSampler
from utils import *
from torch_geometric.nn import to_hetero
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pipeline_obj(args, seed):
    pipeline_obj = PipelineManager(args, seed)
    return pipeline_obj


def get_dir_path(args):
    if args.data_name == 'synthetic':
        return f"GeneratedData/Formula{args.formula}/Config{args.config}/"
    else:
        return f"RealWorldData/{args.data_name}/"


class PipelineManager:
    def __init__(self, args, seed, missing_indices_dict=None):
        self.args = args
        self.seed = seed
        self.gamma = torch.autograd.Variable(torch.tensor([args.gamma]), requires_grad=True).to(device)
        self.epochs = args.epochs
        self.at_k = args.at_k if isinstance(args.at_k, list) else [args.at_k]
        self.graph_path, self.dir_path = read_paths(args)
        self.lattice_graph, self.subgroups = self._load_graph_information()
        self.feature_num = self.lattice_graph['g0'].x.shape[1]
        self.min_level = get_min_level(args.min_m, args.num_layers)
        self.max_level = get_max_level(args.max_m, args.num_layers, self.feature_num)
        self.restricted_graph_idxs_mapping = get_restricted_graph_idxs_mapping(self.feature_num, self.min_level,
                                                                               self.max_level)
        self.missing_indices_dict = self._get_missing_data_dict(missing_indices_dict)
        self.non_missing_dict = {subgroup: [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                            self.missing_indices_dict[subgroup]['all']] for subgroup in self.subgroups}
        self.train_idxs_dict, self.valid_idxs_dict = self._train_validation_split()
        self.test_idxs_dict = self._get_test_indices()

    def _load_graph_information(self):
        lattice_graph = torch.load(self.graph_path)
        subgroups = lattice_graph.node_types
        return lattice_graph, subgroups

    def _get_missing_data_dict(self, missing_indices_dict):
        if missing_indices_dict is not None:
            return missing_indices_dict
        else:
            missing_indices_dict = MissingDataMasking(self.feature_num, self.subgroups, self.seed,
                                                      self.args.missing_prob, self.restricted_graph_idxs_mapping,
                                                      self.args.manual_md).missing_indices_dict
            with open(f"{self.dir_path}missing_data_indices_seed{self.seed}.pkl", 'wb') as f:
                pickle.dump(missing_indices_dict, f)
            return missing_indices_dict

    def _init_model_optim(self):
        model = LatticeGNN(self.args.model, self.feature_num, self.args.hidden_channels, self.args.num_layers, self.args.p_dropout)
        model = to_hetero(model, self.lattice_graph.metadata())
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return model, optimizer

    def test_subgroup(self, subgroup, comb_size, show_results=True):
        test_indices = self.test_idxs_dict[subgroup]
        self.lattice_graph.to(device)
        model = torch.load(f"{self.dir_path}/{self.args.model}_seed{seed}_ratio{self.args.sampling_ratio}_missing{args.missing_prob}.pt")   
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[subgroup].y * self.gamma
        preds = out[subgroup]
        tmp_results_dict = compute_eval_metrics(labels, preds, test_indices, self.at_k, comb_size, self.feature_num)
        if show_results:
            print_results(tmp_results_dict, self.at_k, comb_size, subgroup)
        return tmp_results_dict

    def _train_validation_split(self):
        train_idxs_dict, valid_idxs_dict = dict(), dict()
        sampler = NodeSampler(
            self.seed,
            self.min_level,
            self.max_level,
            self.feature_num,
            self.non_missing_dict,
            self.missing_indices_dict,
            self.restricted_graph_idxs_mapping,
            self.args.sampling_ratio,
            self.args.sampling_method,
        )
        train_idxs_dict = sampler.train_indices_dict
        valid_idxs_dict = sampler.val_indices_dict
        return train_idxs_dict, valid_idxs_dict
        # for g_id in self.subgroups:
        #     if self.args.valid_ratio == 0:
        #         train_idxs_dict[g_id] = sampler.selected_samples[g_id]
        #         valid_idxs_dict[g_id] = self.test_idxs_dict[g_id]
        #     else:
        #         train_idxs_dict[g_id], valid_idxs_dict[g_id] = train_test_split(sampler.selected_samples[g_id],
        #                                                                         test_size=self.args.valid_ratio)
        # return train_idxs_dict, valid_idxs_dict

    def _get_test_indices(self):
        test_idxs_dict = dict()
        for subgroup in self.subgroups:
            test_idxs_dict[subgroup] = [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                        self.train_idxs_dict[subgroup] and idx not in self.valid_idxs_dict[subgroup]]
        return test_idxs_dict

    # def _get_test_indices(self):
    #     return {subgroup: self.missing_indices_dict[subgroup]['all'] for subgroup in self.subgroups}

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
                torch.save(model, f"{self.dir_path}/{self.args.model}_seed{seed}_ratio{self.args.sampling_ratio}_missing{args.missing_prob}.pt")
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
        for epoch in range(1, self.args.epochs + 1):
            if no_impr_counter == GNN.epochs_stable_val:
                break
            # update_gamma = True if epoch % 5 == 0 else False
            train_loss = self._run_training_epoch(model, optimizer, criterion, update_gamma)
            if epoch == 0 or epoch % 5 == 0:
                mse, best_mse, no_impr_counter = self._run_over_validation(model, criterion, best_mse, no_impr_counter, seed)
                print(f"Epoch: {epoch}, train loss = {train_loss:.4f}, val loss: {mse:.4f}, best val loss: {best_mse:.4f}")


    def model_not_found(self, seed):
        for subgroup in self.subgroups:
            path = f"{self.dir_path}{self.args.model}_seed{seed}_{subgroup}.pt"
            if not os.path.exists(path):
                return True
        return False


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
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--min_m', type=int, default=LatticeGeneration.min_m, help='min size of feature combinations')
    parser.add_argument('--max_m', type=int, default=LatticeGeneration.max_m, help='max size of feature combinations')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--data_name', type=str, default='synthetic', help='options:{synthetic, loan, startup, mobile}')
    parser.add_argument('--missing_prob', type=float, default=MissingDataConfig.missing_prob)
    parser.add_argument('--edge_sampling_ratio', type=float, default=LatticeGeneration.edge_sampling_ratio)
    parser.add_argument('--gamma', type=float, default=10.0, help="MI multiplier")
    
    args = parser.parse_args()

    dir_path = get_dir_path(args)
    pipeline_obj = get_pipeline_obj(args, 0)
    subgroups = pipeline_obj.lattice_graph.x_dict.keys()
    results_dict = {comb_size: {seed: {subgroup: dict() for subgroup in subgroups}
                                for seed in range(1, args.seeds_num + 1)} for comb_size in args.comb_size_list}
    missing_dict = {}
    for seed in range(1, args.seeds_num + 1):
        set_seed(seed)
        pipeline_obj = get_pipeline_obj(args, seed)
        subgroups = pipeline_obj.lattice_graph.x_dict.keys()
        # if not args.load_model or pipeline_obj.model_not_found(seed):
        print(f"Seed: {seed}\n=============================")
        pipeline_obj.train_model(seed)
        for comb_size in args.comb_size_list:
            results_dict[comb_size][seed] = {g_id: pipeline_obj.test_subgroup(g_id, comb_size) for g_id in subgroups}
    save_results(results_dict, pipeline_obj.dir_path, args, f'gamma={args.gamma}')

