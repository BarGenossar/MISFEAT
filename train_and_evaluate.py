import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling
from missing_data_masking import MissingDataMasking
from sampler import NodeSampler
from utils import *
from torch_geometric.nn import to_hetero
from sklearn.model_selection import train_test_split
import warnings
import json
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pipeline_obj(args, dir_path):
    pipeline_obj = None
    if args.load_model:
        try:
            with open(f"{dir_path}missing_data_indices.pkl", 'rb') as f:
                missing_indices_dict = pickle.load(f)
                pipeline_obj = PipelineManager(args, missing_indices_dict)
        except FileNotFoundError:
            pass
    if pipeline_obj is None:
        pipeline_obj = PipelineManager(args)
    return pipeline_obj


def get_dir_path(args):
    if args.dir_path is None:
        return f"GeneratedData/Formula{args.formula}/Config{args.config}/"
    else:
        return args.dir_path


class PipelineManager:
    def __init__(self, args, missing_indices_dict=None):
        self.args = args
        self.config_idx = int(args.config)
        self.seeds_num = args.seeds_num
        self.epochs = args.epochs
        self.at_k = verify_at_k(args.at_k)
        self.graph_path = read_paths(args)
        self.dir_path = args.dir_path
        self.lattice_graph, self.subgroups = self._load_graph_information()
        self.feature_num = int(np.log2(len(self.lattice_graph['g0']['x']) + 1))
        self.missing_indices_dict = self._get_missing_data_dict(missing_indices_dict)
        self.non_missing_dict = {subgroup: [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                            self.missing_indices_dict[subgroup]['all']] for subgroup in self.subgroups}
        self.train_idxs_dict, self.valid_idxs_dict = self._train_validation_split()
        self.test_indices = self._get_test_indices()

    def _load_graph_information(self):
        lattice_graph = torch.load(self.graph_path)
        subgroups = lattice_graph.x_dict.keys()
        return lattice_graph, subgroups

    def _get_missing_data_dict(self, missing_indices_dict):
        if missing_indices_dict is not None:
            return missing_indices_dict
        else:
            missing_indices_dict = MissingDataMasking(self.feature_num, self.subgroups, self.config_idx,
                                                        self.args.manual_md).missing_indices_dict
            self.missing_json = { subgroup: [feat for feat in missing_indices_dict[subgroup].keys() if 'f_' in feat] for subgroup in missing_indices_dict.keys() }

            with open(f"{self.dir_path}missing_data_indices.pkl", 'wb') as f:
                pickle.dump(missing_indices_dict, f)

            return missing_indices_dict

    def _init_model_optim(self):
        model = LatticeGNN(self.args.model, self.feature_num, self.args.hidden_channels, self.args.num_layers,
                           self.args.p_dropout)
        model = to_hetero(model, self.lattice_graph.metadata())
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return model, optimizer

    def test_subgroup(self, subgroup, comb_size, show_results=True):
        test_indices = self.test_indices[subgroup]
        self.lattice_graph.to(device)
        model = torch.load(f"{self.dir_path}{self.args.model}_seed{seed}_ratio{self.args.sampling_ratio}_{subgroup}.pt")
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[subgroup].y[test_indices]
        predictions = out[subgroup][test_indices]
        tmp_results_dict = compute_eval_metrics(labels, predictions, self.at_k, comb_size, self.feature_num)
        if show_results:
            print_results(tmp_results_dict, self.at_k, comb_size, subgroup)
        return tmp_results_dict

    def _train_validation_split(self):
        train_idxs_dict, valid_idxs_dict = dict(), dict()
        for g_id in self.subgroups:
            sampler = NodeSampler(
                self.subgroups,
                self.feature_num,
                self.missing_indices_dict,
                self.args.sampling_ratio,
                self.args.sampling_method,
            ) 
            if self.args.valid_ratio == 0:
                train_idxs_dict[g_id] = sampler.selected_samples
                valid_idxs_dict[g_id] = sampler.selected_samples
            else:
                train_idxs_dict[g_id], valid_idxs_dict[g_id] = train_test_split(sampler.selected_samples,
                                                                                test_size=self.args.valid_ratio,
                                                                                random_state=self.config_idx)
        return train_idxs_dict, valid_idxs_dict

    def _get_test_indices(self):
        test_idxs_dict = dict()
        for subgroup in self.subgroups:
            test_idxs_dict[subgroup] = [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                        self.train_idxs_dict[subgroup] and idx not in self.valid_idxs_dict[subgroup]]
            # test_idx_dict[subgroup] = 
        return test_idxs_dict

    def _run_training_epoch(self, train_indices, model, subgroup, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[subgroup].y[train_indices]
        predictions = out[subgroup][train_indices]
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _get_validation_loss(self, validation_indices, model, subgroup, criterion):
        model.eval()
        model.to(device)
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[subgroup].y[validation_indices]
        predictions = out[subgroup][validation_indices]
        loss = criterion(predictions, labels)
        return loss.item()

    def _run_over_validation(self, validation_indices, model, subgroup, criterion, best_val, no_impr_counter, seed):
        loss_validation = self._get_validation_loss(validation_indices, model, subgroup, criterion)
        if loss_validation < best_val:
            best_val = loss_validation
            no_impr_counter = 0
            torch.save(model, f"{self.dir_path}{self.args.model}_seed{seed}_ratio{self.args.sampling_ratio}_{subgroup}.pt")
        else:
            no_impr_counter += 1
        return best_val, no_impr_counter

    def train_model(self, seed):
        with open(f'{self.dir_path}/missing_seed{seed}.json', 'w') as f:
            json.dump(self.missing_json, f)

        criterion = torch.nn.MSELoss()
        self.lattice_graph.to(device)
        for subgroup in self.subgroups:
            print(f"\nTraining on subgroup {subgroup}...")
            model, optimizer = self._init_model_optim()
            train_indices, validation_indices = self.train_idxs_dict[subgroup], self.valid_idxs_dict[subgroup]
            no_impr_counter = 0
            epochs_stable_val = GNN.epochs_stable_val
            best_val = float('inf')
            for epoch in range(1, self.epochs + 1):
                if no_impr_counter == epochs_stable_val:
                    break
                loss_value = self._run_training_epoch(train_indices, model, subgroup, optimizer, criterion)
                if epoch == 1 or epoch % 5 == 0:
                    print(f'Epoch: {epoch}, Loss: {round(loss_value, 4)}')
                    if not self.args.save_model:
                        continue
                    best_val, no_impr_counter = self._run_over_validation(validation_indices, model, subgroup,
                                                                          criterion, best_val, no_impr_counter, seed)
        return

    def model_not_found(self, seed):
        for subgroup in self.subgroups:
            path = f"{self.dir_path}{self.args.model}_seed{seed}_ratio{self.args.sampling_ratio}_{subgroup}.pt"
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--dir_path', type=str, default=None, help='path to the directory file')
    args = parser.parse_args()


    DEBUG = False


    if DEBUG:
        args_seeds_num = 1
        args.epochs = 5
        args.display = True
        args.save_model = True

    seeds_num = args.seeds_num
    dir_path = get_dir_path(args)
    pipeline_obj = get_pipeline_obj(args, dir_path)
    subgroups = pipeline_obj.lattice_graph.x_dict.keys()
    results_dict = {comb_size: {seed: {subgroup: dict() for subgroup in subgroups}
                                for seed in range(1, seeds_num + 1)} for comb_size in args.comb_size_list}

    for seed in range(1, seeds_num + 1):
        set_seed(seed)
        if not args.load_model or pipeline_obj.model_not_found(seed):
            print(f"Seed: {seed}\n=============================")
            pipeline_obj.train_model(seed)
        for comb_size in args.comb_size_list:
            results_dict[comb_size][seed] = {g_id: pipeline_obj.test_subgroup(g_id, comb_size) for g_id in subgroups}

    if not DEBUG:
        save_results(results_dict, dir_path, args.comb_size_list, args)