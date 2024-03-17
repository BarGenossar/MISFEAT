import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from itertools import combinations
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mutual_info_score
from utils import *
from config import LatticeGeneration
import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureLatticeGraph:
    def __init__(self, dataset_path, feature_num, min_k=1,
                 within_level_edges=True, is_hetero=False, with_edge_attrs=False):
        self.min_k = min_k
        self.max_k = feature_num  # Consider to modify it
        self.feature_num = feature_num
        self.within_level_edges = within_level_edges
        self.is_hetero = is_hetero
        self.with_edge_attrs = with_edge_attrs
        self.dataset = self._read_dataset(dataset_path)
        self.mappings_dict = self._create_mappings_dict()
        self.graph = self._create_feature_lattice()
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
        print(f"Generating the mappings dictionary...")
        dataframe = self.dataset.astype(str)
        mappings_dict = defaultdict(dict)
        y_series = dataframe['y'].copy()
        mappings_dict, prev_tmp_dict = self._initialize_tmp_dict(mappings_dict, dataframe, y_series)
        for comb_size in range(self.min_k + 1, self.max_k + 1):
            new_tmp_dict = dict()
            mappings_dict[comb_size] = defaultdict(dict)
            feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, comb_size))
            for comb in tqdm.tqdm(feature_set_combs):
                tmp_series, new_tmp_dict = self._create_feature_set_col(dataframe, comb, prev_tmp_dict, new_tmp_dict)
                mappings_dict = self._update_mappings_dict(mappings_dict, comb_size, comb, tmp_series, y_series)
            prev_tmp_dict = new_tmp_dict.copy()
        return mappings_dict
    
    def _update_mappings_dict(self, mappings_dict, comb_size, comb, tmp_series, y_series):
        binary_vec = convert_comb_to_binary(comb, self.feature_num)
        score = mutual_info_score(tmp_series, y_series)
        mappings_dict[comb_size][comb]['score'] = round(score, 4)
        mappings_dict[comb_size][comb]['binary_vector'] = binary_vec
        #
        mappings_dict[comb_size][comb]['node_id'] = convert_binary_to_decimal(binary_vec) - 1
        return mappings_dict

    def _initialize_tmp_dict(self, mappings_dict, dataframe, y_series):
        prev_tmp_dict = dict()
        feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, self.min_k))
        mappings_dict[self.min_k] = defaultdict(dict)
        for comb in feature_set_combs:
            tmp_series = dataframe[comb[0]]
            for i in range(1, len(comb)):
                tmp_series = tmp_series + dataframe[comb[i]]
            prev_tmp_dict[comb] = tmp_series.copy()
            mappings_dict = self._update_mappings_dict(mappings_dict, self.min_k, comb, tmp_series, y_series)
        return mappings_dict, prev_tmp_dict

    @staticmethod
    def _create_feature_set_col(dataframe, feature_set, prev_tmp_dict, new_tmp_dict):
        tmp_series = prev_tmp_dict[feature_set[:-1]] + dataframe[feature_set[-1]]
        new_tmp_dict[feature_set] = tmp_series.copy()
        return tmp_series, new_tmp_dict

    def _create_feature_lattice(self):
        print(f"\nCreating the feature lattice...")
        if self.is_hetero:
            return self._create_hetero_lattice()
        else:
            return self._create_homogeneous_lattice()

    def _create_homogeneous_lattice(self):
        x, y = self._get_homogeneous_node_features_and_labels()
        edge_index = self._get_homogeneous_edge_index()
        edge_attrs = self._get_homogeneous_edge_attrs()
        #TODO: Add support for edge features
        # make the data object undireced
        return Data(x=x, edge_index=edge_index, y=y)

    def _get_homogeneous_node_features_and_labels(self):
        nodes_num = get_lattice_nodes_num(self.feature_num, self.min_k, self.max_k)
        x = torch.zeros(nodes_num, self.feature_num, dtype=torch.float)
        y = torch.zeros(nodes_num, dtype=torch.float)
        for comb_size in range(self.min_k, self.max_k + 1):
            for comb in self.mappings_dict[comb_size].keys():
                node_id = self.mappings_dict[comb_size][comb]['node_id']
                x[node_id] = torch.tensor([int(digit) for digit in
                                           self.mappings_dict[comb_size][comb]['binary_vector']])
                y[node_id] = self.mappings_dict[comb_size][comb]['score']
        return x, y

    def _get_homogeneous_edge_index(self):
        # TODO: Optimize this function. The current implementation is not efficient.
        # edges_num = get_lattice_edges_num(self.feature_num, self.min_k, self.max_k, self.within_level_edges)
        edge_index = self._get_homogeneous_inter_level_edges()
        edge_index = self._get_homogeneous_intra_level_edges(edge_index)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index

    def _get_homogeneous_inter_level_edges(self):
        edge_index = []
        for comb_size in range(self.min_k, self.max_k):
            for comb in self.mappings_dict[comb_size]:
                node_id = self.mappings_dict[comb_size][comb]['node_id']
                for next_comb in self.mappings_dict[comb_size + 1]:
                    if set(comb).issubset(set(next_comb)):
                        next_node_id = self.mappings_dict[comb_size + 1][next_comb]['node_id']
                        edge_index.append([node_id, next_node_id])
                        edge_index.append([next_node_id, node_id])
        return edge_index

    def _get_homogeneous_intra_level_edges(self, edge_index):
        edge_set = set()
        if self.within_level_edges:
            for comb_size in range(max(2, self.min_k), self.max_k + 1):
                for comb in self.mappings_dict[comb_size]:
                    node_id = self.mappings_dict[comb_size][comb]['node_id']
                    for next_comb in self.mappings_dict[comb_size]:
                        # Check if the overlapping between the two combinations is at size comb_size - 1
                        if len(set(comb).intersection(set(next_comb))) == comb_size - 1:
                            if (next_comb, comb) not in edge_set and (comb, next_comb) not in edge_set:
                                edge_set.add((comb, next_comb))
                                next_node_id = self.mappings_dict[comb_size][next_comb]['node_id']
                                edge_index.append([node_id, next_node_id])
                                edge_index.append([next_node_id, node_id])
        return edge_index

    def _get_homogeneous_edge_attrs(self):
        if self.with_edge_attrs:
            # TODO: Implement this method
            pass
        else:
            return None

    def _create_hetero_lattice(self):
        # TODO: Implement this method
        return None

    def save(self, dataset_path):
        lattice_path = dataset_path.replace('.pkl', '_lattice.pt')
        torch.save(self.graph, lattice_path)
        print(f"The lattice was saved at {lattice_path}")
        return

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph


if __name__ == "__main__":
    formula_idx = LatticeGeneration.formula_idx
    hyperparams_idx = LatticeGeneration.hyperparams_idx
    min_k = LatticeGeneration.min_k
    within_level_edges = LatticeGeneration.within_level_edges
    is_hetero = LatticeGeneration.is_hetero
    with_edge_attrs = LatticeGeneration.with_edge_attrs

    dataset_path = f"GeneratedData/Formula{formula_idx}/Config{hyperparams_idx}/dataset.pkl"
    feature_num = read_feature_num_from_txt(dataset_path)
    lattice = FeatureLatticeGraph(dataset_path, feature_num)





