import time

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
import multiprocessing
warnings.filterwarnings('ignore')


class FeatureLatticeGraph:
    def __init__(self, dataset_path, feature_num, min_k=1, with_edge_attrs=False):
        self.min_k = min_k
        self.max_k = feature_num  # Consider to modify it
        self.feature_num = feature_num
        self.with_edge_attrs = with_edge_attrs
        self.dataset = self._read_dataset(dataset_path)
        self.dataframe = self.dataset.astype(str)
        self.prev_sub_series = {}
        self.mappings_dict = defaultdict(dict)
        self.subgroups_num = self.dataset['subgroup'].nunique()
        self.y_series = self.dataframe['y'].copy()
        start = time.time()
        self._create_mappings_dict()
        print(f"Time to create mappings dictionary: {time.time() - start} seconds")
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
        feature_set_combs = {}  # All combinations for all relevant combination sizes
        for comb_size in range(self.min_k, self.max_k + 1):
            feature_set_combs[comb_size] = list(combinations(self.dataframe.drop(['y', 'subgroup'], axis=1).columns,
                                                             comb_size))
        for g_id in range(self.subgroups_num):
            self.subgroup = g_id
            print(f"Generating the mappings dictionary for subgroup {g_id}:\n")
            self._initialize_mapping(feature_set_combs[self.min_k])
            for comb_size in range(self.min_k + 1, self.max_k + 1):
                self._create_comb_size_mappings_dict(comb_size, feature_set_combs[comb_size])

    # def compute_value(self, comb):
    #     tmp_series = self._create_feature_set_col(self.subgroup, comb)
    #     return self.create_mapping(comb, tmp_series)

    def _create_comb_size_mappings_dict(self, comb_size, feature_combs):
        with multiprocessing.Pool() as pool:
            mappings = pool.map(self._create_feature_set_col, feature_combs)

        prev_sub_series, mappings = zip(*mappings)
        self.mappings_dict[self.subgroup][comb_size] = dict(zip(feature_combs, mappings))
        self.prev_sub_series = dict(zip(feature_combs, prev_sub_series))

    def create_mapping(self, comb, tmp_series):
        binary_vec = convert_comb_to_binary(comb, self.feature_num)
        return {
            'score': round(mutual_info_score(tmp_series, self.y_series), 4),
            'binary_vector': binary_vec,
            'node_id': convert_binary_to_decimal(binary_vec) - 1
        }

    def _get_sub_series(self, comb):
        return self.prev_sub_series[comb[:-1]] + self.dataframe[comb[-1]]

    def _initialize_mapping(self, feature_combs):
        """
        Does not use multiprocessing, assuming the first combinations are small
        """
        if len(feature_combs) > 0:
            self.mappings_dict[self.subgroup][len(feature_combs[0])] = {}
        for comb in feature_combs:
            sub_df = self.dataframe[list(comb)]
            tmp_series = sub_df.apply(lambda x: ''.join(x), axis=1)
            self.prev_sub_series[comb] = tmp_series
            self.mappings_dict[self.subgroup][len(comb)][comb] = self.create_mapping(comb, tmp_series)

    def _create_feature_set_col(self, feature_set):
        tmp_series = self._get_sub_series(feature_set)
        return tmp_series, self.create_mapping(feature_set, tmp_series)

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
