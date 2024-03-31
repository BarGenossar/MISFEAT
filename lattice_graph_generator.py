import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
from itertools import combinations
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mutual_info_score
from utils import *
from config import LatticeGeneration
import argparse
import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureLatticeGraph:
    def __init__(self, dataset_path, feature_num, min_k=1, with_edge_attrs=False):
        self.min_k = min_k
        self.max_k = feature_num  # Consider to modify it
        self.feature_num = feature_num
        self.with_edge_attrs = with_edge_attrs
        self.dataset = self._read_dataset(dataset_path)
        self.subgroups_num = self.dataset['subgroup'].nunique()
        self.mappings_dict = self._create_mappings_dict()
        self.graph = self._create_multiple_feature_lattice()
        self.save(dataset_path)

    @staticmethod
    def _read_dataset(dataset_path):
        return pd.read_pickle(dataset_path)

    def _create_mappings_dict(self):
        """
        Create a dictionary that maps feature combinations to the following:
        1. The mutual information score between the feature combination and the target variable
        2. The binary vector representation of the feature combination
        3. The node ID of the feature combination
        """
        print(f"Generating the mappings dictionary...\n =====================================\n")
        dataframe = self.dataset.astype(str)
        mappings_dict = defaultdict(dict)
        y_series = dataframe['y'].copy()
        for g_id in range(self.subgroups_num):
            print(f"Generating the mappings dictionary for subgroup {g_id}:\n")
            mappings_dict, prev_tmp_dict = self._initialize_tmp_dict(g_id, mappings_dict, dataframe, y_series)
            for comb_size in range(self.min_k + 1, self.max_k + 1):
                mappings_dict, prev_tmp_dict = self._create_comb_size_mappings_dict(g_id, mappings_dict, comb_size,
                                                                                    dataframe, y_series, prev_tmp_dict)

        return mappings_dict

    def _create_comb_size_mappings_dict(self, g_id, mappings_dict, comb_size, dataframe, y_series, prev_tmp_dict):
        new_tmp_dict = dict()
        mappings_dict[g_id][comb_size] = defaultdict(dict)
        feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, comb_size))
        for comb in tqdm.tqdm(feature_set_combs):
            tmp_series, new_tmp_dict = self._create_feature_set_col(dataframe, comb, prev_tmp_dict, new_tmp_dict)
            mappings_dict = self._update_mappings_dict(g_id, mappings_dict, comb_size, comb, tmp_series, y_series)
        prev_tmp_dict = new_tmp_dict.copy()
        print(mappings_dict)
        return mappings_dict, prev_tmp_dict

    def _update_mappings_dict(self, g_id, mappings_dict, comb_size, comb, tmp_series, y_series):
        binary_vec = convert_comb_to_binary(comb, self.feature_num)
        score = mutual_info_score(tmp_series, y_series)
        mappings_dict[g_id][comb_size][comb]['score'] = round(score, 4)
        mappings_dict[g_id][comb_size][comb]['binary_vector'] = binary_vec
        mappings_dict[g_id][comb_size][comb]['node_id'] = convert_binary_to_decimal(binary_vec) - 1
        return dict(mappings_dict)

    def _initialize_tmp_dict(self, g_id, mappings_dict, dataframe, y_series):
        prev_tmp_dict = dict()
        feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, self.min_k))
        mappings_dict[g_id][self.min_k] = defaultdict(dict)
        for comb in feature_set_combs:
            tmp_series = dataframe[comb[0]]
            for i in range(1, len(comb)):
                tmp_series = tmp_series + dataframe[comb[i]]
            prev_tmp_dict[comb] = tmp_series.copy()
            mappings_dict = self._update_mappings_dict(g_id, mappings_dict, self.min_k, comb, tmp_series, y_series)
        return mappings_dict, prev_tmp_dict

    @staticmethod
    def _create_feature_set_col(dataframe, feature_set, prev_tmp_dict, new_tmp_dict):
        tmp_series = prev_tmp_dict[feature_set[:-1]] + dataframe[feature_set[-1]]
        new_tmp_dict[feature_set] = tmp_series.copy()
        return tmp_series, new_tmp_dict

    def _create_multiple_feature_lattice(self):
        print(f"\nCreating the feature lattice...")
        data = HeteroData()
        data = self._get_node_features_and_labels(data)
        data = self._get_edge_index(data)
        data = self._get_edge_attrs(data)
        return data

    def _get_node_features_and_labels(self, data):
        lattice_nodes_num = get_lattice_nodes_num(self.feature_num, self.min_k, self.max_k)
        for g_id in range(self.subgroups_num):
            x_tensor = torch.zeros(lattice_nodes_num, self.feature_num, dtype=torch.float)
            y_tensor = torch.zeros(lattice_nodes_num, dtype=torch.float)
            for comb_size in range(self.min_k, self.max_k + 1):
                for comb in self.mappings_dict[g_id][comb_size].keys():
                    node_id = self.mappings_dict[g_id][comb_size][comb]['node_id']
                    x_tensor[node_id] = torch.tensor([int(digit) for digit in
                                                      self.mappings_dict[g_id][comb_size][comb]['binary_vector']])
                    y_tensor[node_id] = self.mappings_dict[g_id][comb_size][comb]['score']
            data[f"g{g_id}"].x = x_tensor
            data[f"g{g_id}"].y = y_tensor
        return data

    def _get_edge_index(self, data):
        # TODO: Optimize this function. The current implementation is not efficient.
        # edges_num = get_lattice_edges_num(self.feature_num, self.min_k, self.max_k)
        data = self._get_intra_lattice_edges(data)
        data = self._get_inter_lattice_edges(data)
        return data

    def _get_intra_lattice_edges(self, data):
        data = self._get_inter_level_edges(data)
        data = self._get_intra_level_edges(data)
        return data

    def _get_inter_level_edges(self, data):
        for g_id in range(self.subgroups_num):
            edge_index = []
            for comb_size in range(self.min_k, self.max_k):
                for comb in self.mappings_dict[g_id][comb_size]:
                    node_id = self.mappings_dict[g_id][comb_size][comb]['node_id']
                    for next_comb in self.mappings_dict[g_id][comb_size + 1]:
                        if set(comb).issubset(set(next_comb)):
                            next_node_id = self.mappings_dict[g_id][comb_size + 1][next_comb]['node_id']
                            edge_index.append([node_id, next_node_id])
                            edge_index.append([next_node_id, node_id])
            edge_name = self._get_edge_name(g_id, g_id)
            data[f"g{g_id}", edge_name, f"g{g_id}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    @staticmethod
    def _get_edge_name(g_id1, g_id2):
        g_str1 = ''.join(('g', str(g_id1)))
        g_str2 = ''.join(('g', str(g_id2)))
        return ''.join((g_str1, 'TO', g_str2))


    def _get_intra_level_edges(self, data):
        for g_id in range(self.subgroups_num):
            edge_set = set()
            edge_index = []
            for comb_size in range(max(2, self.min_k), self.max_k + 1):
                for comb in self.mappings_dict[g_id][comb_size]:
                    node_id = self.mappings_dict[g_id][comb_size][comb]['node_id']
                    for next_comb in self.mappings_dict[g_id][comb_size]:
                        # Check if the overlapping between the two combinations is at size comb_size - 1
                        if len(set(comb).intersection(set(next_comb))) == comb_size - 1:
                            if (next_comb, comb) not in edge_set and (comb, next_comb) not in edge_set:
                                edge_set.add((comb, next_comb))
                                next_node_id = self.mappings_dict[g_id][comb_size][next_comb]['node_id']
                                edge_index.append([node_id, next_node_id])
                                edge_index.append([next_node_id, node_id])
            edge_name = self._get_edge_name(g_id, g_id)
            data[f"g{g_id}", edge_name, f"g{g_id}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    def _get_inter_lattice_edges(self, data):
        lattice_nodes_num = get_lattice_nodes_num(self.feature_num, self.min_k, self.max_k)
        edge_index = [[node_id, node_id] for node_id in range(lattice_nodes_num)]
        for g_id1 in range(self.subgroups_num):
            for g_id2 in range(g_id1 + 1, self.subgroups_num):
                edge_name1 = self._get_edge_name(g_id1, g_id2)
                edge_name2 = self._get_edge_name(g_id2, g_id1)
                data[f"g{g_id1}", edge_name1, f"g{g_id2}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                data[f"g{g_id2}", edge_name2, f"g{g_id1}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    def _get_edge_attrs(self, data):
        if self.with_edge_attrs:
            # TODO: Implement this method
            pass
        else:
            return data

    def save(self, dataset_path):
        graph_path = dataset_path.replace('.pkl', '_hetero_graph.pt')
        torch.save(self.graph, graph_path)
        print(f"The lattice graph was saved at {graph_path}")
        return

    # def __len__(self):
    #     return 1
    #
    # def __getitem__(self, idx):
    #     return self.graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Lattice Graph Generation')
    parser.add_argument('--formula', type=str, default=LatticeGeneration.formula_idx, help='index of the formula')
    parser.add_argument('--config', type=str, default=LatticeGeneration.hyperparams_idx, help='index of configuration')
    parser.add_argument('--min_k', type=int, default=LatticeGeneration.min_k, help='min size of feature combinations')
    parser.add_argument('--within_level', type=bool, default=LatticeGeneration.within_level_edges,
                        help='add edges within the same level')
    # parser.add_argument('--hetero', type=bool, default=LatticeGeneration.is_hetero, help='create heterogeneous graph')
    parser.add_argument('--edge_attrs', type=bool, default=LatticeGeneration.with_edge_attrs,
                        help='add attributes to the edges')
    args = parser.parse_args()

    dataset_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/dataset.pkl"
    feature_num = read_feature_num_from_txt(dataset_path)
    lattice = FeatureLatticeGraph(dataset_path, feature_num)
