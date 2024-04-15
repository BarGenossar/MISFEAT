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
import time
import multiprocessing
from functools import partial
warnings.filterwarnings('ignore')


class FeatureLatticeGraph:
    def __init__(self, dataset_path, with_edge_attrs=False):
        self.with_edge_attrs = with_edge_attrs
        self.dataset = self._read_dataset(dataset_path)
        self.feature_num = self.dataset.shape[1] - 2
        self.subgroups_num = self.dataset['subgroup'].nunique()
        self.cores_to_use = min(self.subgroups_num, int(multiprocessing.cpu_count() - 4))
        self.mappings_dict = self._create_mappings_dict()
        self.graph = self._create_multiple_feature_lattice()
        self.save(dataset_path)

    @staticmethod
    def _read_dataset(dataset_path):
        return pd.read_pickle(dataset_path)

    def _create_mappings_dict(self):
        print(f"Generating the mappings dictionary...\n =====================================\n")
        start = time.time()
        dataframe = self.dataset.astype(str)
        y_series = dataframe['y'].copy()
        with multiprocessing.Pool(processes=self.cores_to_use) as pool:
            mappings_list = list(pool.imap(partial(self._process_gid_mapping_dict, dataframe=dataframe,
                                                   y_series=y_series), range(self.subgroups_num)))
        end = time.time()
        print(f"Mapping dictionary generation time: {round(end - start, 4)} seconds\n ========================\n")
        return self._convert_mappings_list_to_dict(mappings_list)

    @staticmethod
    def _convert_mappings_list_to_dict(mappings_list):
        mappings_dict = dict()
        for mappings in mappings_list:
            mappings_dict.update(mappings)
        return mappings_dict

    def _process_gid_mapping_dict(self, g_id, dataframe, y_series):
        mappings_dict, prev_tmp_dict = self._initialize_tmp_dict(g_id, dataframe, y_series)
        for comb_size in range(2, self.feature_num + 1):
            mappings_dict, prev_tmp_dict = self._create_comb_size_mappings_dict(g_id, mappings_dict, comb_size,
                                                                                dataframe, y_series, prev_tmp_dict)
        return mappings_dict


    def _create_comb_size_mappings_dict(self, g_id, mappings_dict, comb_size, dataframe, y_series, prev_tmp_dict):
        mappings_dict[g_id][comb_size] = defaultdict(dict)
        feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, comb_size))
        rel_idxs = dataframe[dataframe['subgroup'] == str(g_id)].index
        y_series = y_series[rel_idxs]
        comb_property_list = []
        for comb in tqdm.tqdm(feature_set_combs):
            comb_property_list.append(self._process_comb(comb, rel_idxs, dataframe, prev_tmp_dict, y_series))
        mappings_dict, prev_tmp_dict = self._update_comb_dicts(g_id, mappings_dict, comb_size, comb_property_list)
        return mappings_dict, prev_tmp_dict

    def _process_comb(self, comb, rel_idxs, dataframe, prev_tmp_dict, y_series):
        if len(comb) > 1:
            tmp_series = prev_tmp_dict[comb[:-1]] + dataframe[comb[-1]][rel_idxs]
        else:
            tmp_series = prev_tmp_dict[comb].copy()
        score = round(mutual_info_score(tmp_series, y_series), 6)
        binary_vec = convert_comb_to_binary(comb, self.feature_num)
        node_id = convert_binary_to_decimal(binary_vec) - 1
        return {comb: {'score': score, 'binary_vector': binary_vec, 'node_id': node_id, 'tmp_series': tmp_series}}

    @staticmethod
    def _update_comb_dicts(g_id, mappings_dict, comb_size, comb_property_list):
        new_tmp_dict = dict()
        for comb_dict in comb_property_list:
            comb = list(comb_dict.keys())[0]
            mappings_dict[g_id][comb_size][comb]['score'] = comb_dict[comb]['score']
            mappings_dict[g_id][comb_size][comb]['binary_vector'] = comb_dict[comb]['binary_vector']
            mappings_dict[g_id][comb_size][comb]['node_id'] = comb_dict[comb]['node_id']
            new_tmp_dict[comb] = comb_dict[comb]['tmp_series']
        return mappings_dict, new_tmp_dict

    def _initialize_tmp_dict(self, g_id, dataframe, y_series):
        mappings_dict = {g_id: {1: defaultdict(dict)}}
        prev_tmp_dict = dict()
        feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, 1))
        rel_idxs = dataframe[dataframe['subgroup'] == str(g_id)].index
        y_series = y_series[rel_idxs]
        comb_property_list = []
        for comb in feature_set_combs:
            tmp_series = dataframe[comb[0]][rel_idxs]
            prev_tmp_dict[comb] = tmp_series.copy()
            comb_property_list.append(self._process_comb(comb, rel_idxs, dataframe, prev_tmp_dict, y_series))
        mappings_dict, _ = self._update_comb_dicts(g_id, mappings_dict, 1, comb_property_list)
        return mappings_dict, prev_tmp_dict

    def _create_multiple_feature_lattice(self):
        print(f"\nCreating the feature lattice...")
        data = HeteroData()
        data = self._get_node_features_and_labels(data)
        data = self._get_edge_index(data)
        data = self._get_edge_attrs(data)
        return data

    def _get_node_features_and_labels(self, data):
        print(f"Getting the node features and labels...\n ------------------\n")
        lattice_nodes_num = get_lattice_nodes_num(self.feature_num, self.feature_num)
        with multiprocessing.Pool(processes=self.cores_to_use) as pool:
           gid_nodes_dict_list = list(pool.imap(partial(self._get_gid_nodes, lattice_nodes_num=lattice_nodes_num),
                                                range(self.subgroups_num)))
        return self._convert_gid_nodes_to_data(gid_nodes_dict_list, data)

    def _get_gid_nodes(self, g_id, lattice_nodes_num):
        x_tensor = torch.zeros(lattice_nodes_num, self.feature_num, dtype=torch.float)
        y_tensor = torch.zeros(lattice_nodes_num, dtype=torch.float)
        for comb_size in range(1, self.feature_num + 1):
            for comb in tqdm.tqdm(self.mappings_dict[g_id][comb_size].keys()):
                node_id = self.mappings_dict[g_id][comb_size][comb]['node_id']
                x_tensor[node_id] = torch.tensor([int(digit) for digit in
                                                  self.mappings_dict[g_id][comb_size][comb]['binary_vector']])
                y_tensor[node_id] = self.mappings_dict[g_id][comb_size][comb]['score']
        return {f"g{g_id}": {"x": x_tensor, "y": y_tensor}}

    @staticmethod
    def _convert_gid_nodes_to_data(gid_nodes_dict_list, data):
        for gid_nodes_dict in gid_nodes_dict_list:
            gid = list(gid_nodes_dict.keys())[0]
            data[gid].x = gid_nodes_dict[gid]["x"]
            data[gid].y = gid_nodes_dict[gid]["y"]
        return data


    def _get_edge_index(self, data):
        # edges_num = get_lattice_edges_num(self.feature_num, self.feature_num)
        data = self._get_intra_lattice_edges(data)
        data = self._get_inter_lattice_edges(data)
        return data

    def _get_intra_lattice_edges(self, data):
        data = self._get_inter_level_edges(data)
        data = self._get_intra_level_edges(data)
        return data

    def _get_inter_level_edges(self, data):
        print(f"Getting the inter-level edges...\n ------------------\n")
        with multiprocessing.Pool(processes=self.cores_to_use) as pool:
            gid_inter_edges_dict_list = list(pool.map(self._get_gid_inter_level_edges, range(self.subgroups_num)))
        return self._convert_gid_inter_edges_to_data(gid_inter_edges_dict_list, data)

    def _get_gid_inter_level_edges(self, g_id):
        edge_index = []
        for comb_size in range(1, self.feature_num):
            for comb in tqdm.tqdm(self.mappings_dict[g_id][comb_size]):
                node_id = self.mappings_dict[g_id][comb_size][comb]['node_id']
                for next_comb in self.mappings_dict[g_id][comb_size + 1]:
                    if set(comb).issubset(set(next_comb)):
                        next_node_id = self.mappings_dict[g_id][comb_size + 1][next_comb]['node_id']
                        edge_index.append([node_id, next_node_id])
                        edge_index.append([next_node_id, node_id])
        return {f"g{g_id}": {'edge_index': edge_index, 'edge_name': self._get_edge_name(g_id, g_id)}}

    @staticmethod
    def _convert_gid_inter_edges_to_data(gid_inter_edges_dict_list, data):
        for gid_inter_edges_dict in gid_inter_edges_dict_list:
            gid = list(gid_inter_edges_dict.keys())[0]
            edge_name = gid_inter_edges_dict[gid]['edge_name']
            edge_index = gid_inter_edges_dict[gid]['edge_index']
            data[f"{gid}", edge_name, f"{gid}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    @staticmethod
    def _get_edge_name(g_id1, g_id2):
        g_str1 = ''.join(('g', str(g_id1)))
        g_str2 = ''.join(('g', str(g_id2)))
        return ''.join((g_str1, 'TO', g_str2))

    def _get_intra_level_edges(self, data):
        print(f"Getting the intra-level edges...\n ------------------\n")
        with multiprocessing.Pool(processes=self.cores_to_use) as pool:
            gid_intra_edges_dict_list = list(pool.map(self._get_gid_intra_level_edges, range(self.subgroups_num)))
        return self._convert_gid_intra_edges_to_data(gid_intra_edges_dict_list, data)

    def _get_gid_intra_level_edges(self, g_id):
        edge_set = set()
        edge_index = []
        for comb_size in range(2, self.feature_num + 1):
            for comb in tqdm.tqdm(self.mappings_dict[g_id][comb_size]):
                node_id = self.mappings_dict[g_id][comb_size][comb]['node_id']
                for next_comb in self.mappings_dict[g_id][comb_size]:
                    # Check if the overlapping between the two combinations is at size comb_size - 1
                    if len(set(comb).intersection(set(next_comb))) == comb_size - 1:
                        if (next_comb, comb) not in edge_set and (comb, next_comb) not in edge_set:
                            edge_set.add((comb, next_comb))
                            next_node_id = self.mappings_dict[g_id][comb_size][next_comb]['node_id']
                            edge_index.append([node_id, next_node_id])
                            edge_index.append([next_node_id, node_id])
        return {f"g{g_id}": {'edge_index': edge_index, 'edge_name': self._get_edge_name(g_id, g_id)}}

    @staticmethod
    def _convert_gid_intra_edges_to_data(gid_intra_edges_dict_list, data):
        for gid_intra_edges_dict in gid_intra_edges_dict_list:
            gid = list(gid_intra_edges_dict.keys())[0]
            edge_name = gid_intra_edges_dict[gid]['edge_name']
            edge_index = gid_intra_edges_dict[gid]['edge_index']
            data[f"{gid}", edge_name, f"{gid}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    def _get_inter_lattice_edges(self, data):
        lattice_nodes_num = get_lattice_nodes_num(self.feature_num, self.feature_num)
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
    parser.add_argument('--dataset_path', type=str, default=None, help='path to the dataset file')
    args = parser.parse_args()

    if args.dataset_path is None:
        # For synthetic datasets
        dataset_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/dataset.pkl"
    else:
        # For real-world datasets
        dataset_path = args.dataset_path

    lattice = FeatureLatticeGraph(dataset_path)
