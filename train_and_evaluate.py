
import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling
from missing_data_masking import MissingDataMasking
from sampler import NodeSampler
from utils import *
from torch_geometric.nn import to_hetero
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train(lattice_graph, train_indices, model, g_id, optimizer, criterion):
#     lattice_graph.to(device)
#     model.train()
#     optimizer.zero_grad()
#     out = model(lattice_graph.x_dict, lattice_graph.edge_index_dict)
#     labels = lattice_graph[g_id].y[train_indices]
#     predictions = out[g_id][train_indices]
#     loss = criterion(predictions, labels)
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#
#
# def test(lattice_graph, test_indices, model, g_id, at_k, comb_size, feature_num, show_results=True):
#     lattice_graph.to(device)
#     model.eval()
#     with torch.no_grad():
#         out = model(lattice_graph.x_dict, lattice_graph.edge_index_dict)
#     labels = lattice_graph[g_id].y[test_indices]
#     predictions = out[g_id][test_indices]
#     tmp_results_dict = compute_eval_metrics(labels, predictions, at_k, comb_size, feature_num)
#     if show_results:
#         print_results(tmp_results_dict, at_k, comb_size, g_id)
#     return tmp_results_dict
#
#
# def initialize_model_and_optimizer(args):
#     model = LatticeGNN(args.model, feature_num, args.hidden_channels, seed, args.num_layers, args.p_dropout)
#     model = to_hetero(model, lattice_graph.metadata())
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     return model, optimizer


class Trainer:
    def __init__(self, args):
        self.args = args
        self.config_idx = args.config
        self.seeds_num = args.seeds_num
        self.comb_size = args.comb_size
        self.epochs = args.epochs
        self.at_k = verify_at_k(args.at_k)
        self.dataset_path, self.graph_path, self.dir_path = read_paths(args)
        self.feature_num = read_feature_num_from_txt(self.dataset_path)
        self.lattice_graph, self.subgroups, self.missing_indices_dict = self.load_graph_information()
        self.test_indices = {subgroup: self.missing_indices_dict[subgroup]['all'] for subgroup in self.subgroups}

    def load_graph_information(self):
        lattice_graph = torch.load(self.graph_path)
        subgroups = lattice_graph.x_dict.keys()
        missing_indices_dict = MissingDataMasking(self.feature_num, subgroups, self.config_idx,
                                                  self.args.manual_md).missing_indices_dict
        return lattice_graph, subgroups, missing_indices_dict

    def init_model_optim(self, seed):
        model = LatticeGNN(self.args.model, self.feature_num, self.args.hidden_channels, seed, self.args.num_layers,
                           self.args.p_dropout)
        model = to_hetero(model, self.lattice_graph.metadata())
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return model, optimizer

    def train(self, train_indices, model, g_id, optimizer, criterion):
        self.lattice_graph.to(device)
        model.train()
        optimizer.zero_grad()
        out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[g_id].y[train_indices]
        predictions = out[g_id][train_indices]
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test(self, test_indices, model, g_id, show_results=True):
        self.lattice_graph.to(device)
        model.eval()
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[g_id].y[test_indices]
        predictions = out[g_id][test_indices]
        tmp_results_dict = compute_eval_metrics(labels, predictions, self.at_k, self.comb_size, self.feature_num)
        if show_results:
            print_results(tmp_results_dict, self.at_k, self.comb_size, g_id)
        return tmp_results_dict

    def save_results(self, results_dict):
        save_results(results_dict, self.dir_path, self.comb_size, self.args)

    def train_validation_split(self, non_missing):
        if self.args.sampling_ratio == 1.0:
            train_indices = non_missing
            validation_indices = []
        else:
            sampler = NodeSampler(self.subgroups, self.config_idx, non_missing, self.args.sampling_ratio,
                                  Sampling.method)
            train_indices = sampler.selected_samples
            validation_indices = [idx for idx in non_missing if idx not in train_indices]
        return train_indices, validation_indices

    def evaluate(self, model):
        pass

    def execute(self):
        try:  # try loading pretrained models
            if not self.args.pretrained:
                raise AttributeError
            results_dict = {}
            for seed in range(1, self.seeds_num + 1):
                results_dict[seed] = {}
                for subgroup in self.subgroups:
                    path = f"{self.dir_path}{self.args.model}_seed{seed}_{subgroup}.pt"
                    model = torch.load(path)
                    model.to(device)
                    model.eval()
                    if self.test_indices[subgroup]:
                        results_dict[seed][subgroup] = self.test(self.test_indices[subgroup], model, subgroup)
                    else:
                        results_dict[seed][subgroup] = {metric: {k: 0 for k in self.at_k} for metric in
                                                        Evaluation.eval_metrics}
            self.save_results(results_dict)

        except (AttributeError, FileNotFoundError) as e:
            results_dict = {seed: {subgroup: dict() for subgroup in self.subgroups}
                            for seed in range(1, self.seeds_num + 1)}
            all_non_missing = {subgroup: [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                          self.missing_indices_dict[subgroup]['all']] for subgroup in self.subgroups}
            for seed in range(1, self.seeds_num + 1):
                torch.manual_seed(seed)
                criterion = torch.nn.MSELoss()
                loss_vals = {subgroup: [] for subgroup in self.subgroups}

                for subgroup in self.subgroups:
                    print(f"\nTraining on subgroup {subgroup}...")

                    model, optimizer = self.init_model_optim(seed)

                    train_indices, validation_indices = self.train_validation_split(all_non_missing[subgroup])
                    count_validation_static = 0
                    previous_validation = -1  # Evaluation metric can't be negative
                    best_validation = -1

                    for epoch in range(1, self.epochs + 1):
                        if count_validation_static == GNN.epochs_stable_val:  # epochs with no change to the evaluation on the validation
                            break
                        loss_val = self.train(train_indices, model, subgroup, optimizer, criterion)
                        loss_vals[subgroup].append(loss_val)
                        if validation_indices:
                            output = self.test(validation_indices, model, subgroup, False)
                            single_output = output[Evaluation.eval_metrics[0]][self.at_k[-1]]  # First evaluation
                            # metric at highest k
                            if single_output <= previous_validation:
                                count_validation_static += 1
                            else:
                                count_validation_static = 0
                            if self.args.save_model and single_output > best_validation:
                                best_validation = single_output
                                torch.save(model, f"{self.dir_path}{self.args.model}_seed{seed}_{subgroup}.pt")
                            previous_validation = single_output

                        if epoch == 1 or epoch % 5 == 0:
                            continue
                            # print(f'Epoch: {epoch}, Loss: {round(loss_val, 4)}')
                    if self.test_indices[subgroup]:
                        results_dict[seed][subgroup] = self.test(self.test_indices[subgroup], model, subgroup)
                    else:
                        results_dict[seed][subgroup] = {metric: {k: 0 for k in self.at_k} for metric in
                                                        Evaluation.eval_metrics}
                    if not validation_indices and self.args.save_model:
                        torch.save(model, f"{self.dir_path}{self.args.model}_seed{seed}_{subgroup}.pt")
            self.save_results(results_dict)


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
    parser.add_argument('--sampling_ratio', type=float, default=0.5)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size', type=int, default=Evaluation.comb_size)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    _args = parser.parse_args()

    trainer = Trainer(_args)
    trainer.execute()
