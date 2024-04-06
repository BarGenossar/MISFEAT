import torch
import torch_geometric
import typing as t
import random
from torch_geometric.data import Data, HeteroData
from itertools import combinations
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mutual_info_score
from utils import *
import argparse
import tqdm
import multiprocessing as mp
from itertools import repeat
import numpy as np
from sampler import NodeSampler
import pickle
import warnings
warnings.filterwarnings('ignore')

DEBUG = True


class GraphSampling:
    def __init__(self, df, missing_indices_dict, sampling_ratio, sampling_method='uniform', min_k=1, num_workers=8, edge_threshold=5):
        self.min_k = min_k
        # self.with_edge_attrs = with_edge_attrs
        # self.dataset = self._read_dataset(dataset_path)
        self.base_features = [feat for feat in list(df.columns) if 'f_' in feat]
        self.feature_num = len(self.base_features)
        self.subgroups_num = df['subgroup'].nunique()
        
        if DEBUG:
            with open('MI_dict_synthetic_data.pkl', 'rb') as f:
                self.mappings_dict = pickle.load(f)
        else:
            self.mappings_dict = self._create_mappings_dict(df)
            with open('MI_dict_synthetic_data.pkl', 'wb') as f:
                pickle.dump(self.mappings_dict, f)

        self.subgroups = [f"g{gid}" for gid in range(self.subgroups_num)]
        self.sampler = NodeSampler(self.subgroups, self.feature_num, missing_indices_dict, sampling_ratio, sampling_method)

        self.data = HeteroData()

        self._get_random_nodes()
        self._get_inter_edges(num_workers, edge_threshold)
        self._get_xy()


    def _get_random_nodes(self) -> None:
        """
        using sampler to create random nodes, append the random nodes to HeteroData
        """
        strings, intra_edges = self.sampler.get_selected_samples()
        for subgroup in self.subgroups:
            self.data[subgroup].string = strings[subgroup]
            self.data[subgroup, 'TO', subgroup] = torch.tensor(intra_edges[subgroup], dtype=torch.long).t()


    def _string_to_score(self, s: str, g_id: int) -> float:
        """
        Example:
            s = '1101' -> ('f_0', 'f_2', 'f_3') -> MI score
        """
        comb = tuple( [f"f_{idx}" for idx, bit in enumerate(s[::-1]) if bit == '1'] )
        comb_size = len(comb)
        return self.mappings_dict[g_id][comb_size][comb]['score']
    

    def _string_to_tensor(self, s: str) -> t.List[int]:
        return [int(bit) for bit in s]


    def _get_xy(self) -> None:
        for subgroup in self.subgroups:
            g_id = int(subgroup.split('g')[-1])
            ## convert binary strings -> x tensors
            tensors = list(map(self._string_to_tensor, self.data[subgroup].string))
            self.data[subgroup].x = torch.from_numpy( np.array(tensors) ).float()
            ## convert binary strings -> mi scores
            scores = list(map(self._string_to_score, self.data[subgroup].string, repeat(g_id)))
            self.data[subgroup].y = torch.tensor(scores, dtype=torch.float32)

    
    def _get_inter_edges(self, num_workers, edge_threshold) -> None:
        for G1 in range(len(self.subgroups) - 1):
            strings_1 = self.data[f'g{G1}'].string
            for G2 in range(G1 + 1, len(self.subgroups)):
                strings_2 = self.data[f'g{G2}'].string

                indices_1, indices_2 = [], []
                for b1 in range(len(strings_1)):
                    for b2 in range(len(strings_2)):
                        indices_1.append(b1)
                        indices_2.append(b2)

                with mp.Pool(num_workers) as pool:
                    check_edge_exist = pool.map(self._mp_wrapper_inter, zip(repeat(strings_1), repeat(strings_2), indices_1, indices_2, repeat(edge_threshold)))

                edges_12 = []
                edges_21 = []
                connected = False
                for idx_1, idx_2, has_edge in zip(indices_1, indices_2, check_edge_exist):
                    if has_edge:
                        edges_12.append([idx_1, idx_2])
                        edges_21.append([idx_2, idx_1])
                        connected = True
                
                if not connected:
                    b1 = random.choice(range(len(strings_1)))
                    b2 = random.choice(range(len(strings_2)))
                    edges_12.append([b1, b2])
                    edges_21.append([b2, b1])

            self.data[f'g{G1}', 'TO', f'g{G2}'].edge_index = torch.tensor(edges_12, dtype=torch.long).t()
            self.data[f'g{G2}', 'TO', f'g{G1}'].edge_index = torch.tensor(edges_21, dtype=torch.long).t()

    @staticmethod
    def _check_inter_edge(strings_1: t.List[str], strings_2: t.List[str], idx_1: int, idx_2: int, edge_threshold: int = 5) -> bool:
        common_ones = 0
        for b1, b2 in zip(strings_1[idx_1], strings_2[idx_2]):
            if b1 == b2 == '1': common_ones += 1
        return common_ones >= edge_threshold

    def _mp_wrapper_inter(self, args):
        return self._check_inter_edge(*args)




#############################################################
    def _create_mappings_dict(self, dataframe):
        """
        Create a dictionary that maps feature combinations to the following:
        1. The mutual information score between the feature combination and the target variable
        2. The binary vector representation of the feature combination
        3. The node ID of the feature combination
        """
        print(f"Generating the mappings dictionary...\n =====================================\n")
        dataframe = dataframe.astype(str)
        mappings_dict = defaultdict(dict)
        y_series = dataframe['y'].copy()
        for g_id in range(self.subgroups_num):
            print(f"Generating the mappings dictionary for subgroup {g_id}:\n")
            mappings_dict, prev_tmp_dict = self._initialize_tmp_dict(g_id, mappings_dict, dataframe, y_series)
            for comb_size in range(self.min_k + 1, self.feature_num + 1):
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
        return mappings_dict, prev_tmp_dict

    def _update_mappings_dict(self, g_id, mappings_dict, comb_size, comb, tmp_series, y_series):
        binary_vec = convert_comb_to_binary(comb, self.feature_num)
        score = mutual_info_score(tmp_series, y_series)
        mappings_dict[g_id][comb_size][comb]['score'] = round(score, 4)
        mappings_dict[g_id][comb_size][comb]['binary_vector'] = binary_vec
        mappings_dict[g_id][comb_size][comb]['node_id'] = convert_binary_to_decimal(binary_vec) - 1
        return mappings_dict

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
        lattice_nodes_num = get_lattice_nodes_num(self.feature_num, self.min_k, self.feature_num)
        for g_id in range(self.subgroups_num):
            x_tensor = torch.zeros(lattice_nodes_num, self.feature_num, dtype=torch.float)
            y_tensor = torch.zeros(lattice_nodes_num, dtype=torch.float)
            for comb_size in range(self.min_k, self.feature_num + 1):
                for comb in self.mappings_dict[g_id][comb_size].keys():
                    node_id = self.mappings_dict[g_id][comb_size][comb]['node_id']
                    x_tensor[node_id] = torch.tensor([int(digit) for digit in
                                                      self.mappings_dict[g_id][comb_size][comb]['binary_vector']])
                    y_tensor[node_id] = self.mappings_dict[g_id][comb_size][comb]['score']
            data[f"g{g_id}"].x = x_tensor
            data[f"g{g_id}"].y = y_tensor
        return data

    def _get_edge_attrs(self, data):
        if self.with_edge_attrs:
            # TODO: Implement this method
            pass
        else:
            return data