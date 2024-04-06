import numpy as np
from config import MissingDataConfig
from utils import convert_decimal_to_binary, convert_binary_to_decimal
import typing as t
import random
import numpy as np
from collections import defaultdict


class NodeSampler:
    def __init__(self, subgroups, num_base, missing_indices_dict, sampling_ratio, sampling_method='random'):
        self.subgroups = subgroups
        self.num_base = num_base
        self._features_name_to_int(missing_indices_dict)
        self.sampling_ratio = sampling_ratio
        self.sampling_method = sampling_method
        # self.selected_samples = self.get_selected_samples(sampling_method)

    def _features_name_to_int(self, missing_dict):
        self.missing_bit_idx = dict()
        self.present_bit_idx = dict()
        for subgroup in self.subgroups:
            missing_bits = [int(feat.split('_')[-1]) for feat in missing_dict[subgroup].keys() if 'f_' in feat]
            present_bits = sorted(list( set(range(self.num_base)) - set(missing_bits) ))
            self.missing_bit_idx[subgroup] = missing_bits
            self.present_bit_idx[subgroup] = present_bits

    def get_selected_samples(self):
        sampling_funcs = {
            'random': self._random_sampling,
            'uniform': self._uniform_sampling,
            'gibbs': self._gibbs_sampling,
        }
        if self.sampling_method not in sampling_funcs:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}")
        return sampling_funcs[self.sampling_method]()

    def _random_sampling(self):
        selected_samples = dict()
        for subgroup in self.subgroups:
            candidate_indices = self.rel_indices_dict[subgroup]
            num_samples = int(self.sampling_ratio * len(candidate_indices))
            selected_samples[subgroup] = np.random.choice(candidate_indices, num_samples, replace=False)
        return selected_samples

    def _random_walk(self, node_list: t.List[str], present_bits: t.List[int]) -> None:
        curr_node = node_list[-1]
        # curr_node_idx = len(node_list) - 1
        rand_idx = self.num_base-1 - random.choice(present_bits)
        rand_bit = np.random.choice(['0', '1'], p=[0.5, 0.5])
        if rand_bit != curr_node[rand_idx]:
            new_node =curr_node[ : rand_idx] + rand_bit + curr_node[rand_idx + 1 : ]
            if new_node != '0' * self.num_base:    # ensure non-empty comb
                node_list.append(new_node)
                # edge_list.append([curr_node_idx, curr_node_idx + 1])
                # edge_list.append([curr_node_idx + 1, curr_node_idx])

    def _uniform_sampling(self):
        indices = dict()
        for subgroup in self.subgroups:
            present_bits = self.present_bit_idx[subgroup]
            num_samples = int(self.sampling_ratio * (2**len(present_bits) - 1))
            ## sample starting node
            random_features = np.random.choice([0, 1], size=len(present_bits), p=[0.5, 0.5])   # from present features, select each with prob 0.5
            start_node = [0] * self.num_base
            for idx, bit in enumerate(random_features):
                start_node[self.num_base-1 - present_bits[idx]] = bit
            start_node = ''.join([str(bit) for bit in start_node])
            ## random walk in the hypercube
            strings = [start_node]
            while len(strings) < num_samples:
                self._random_walk(strings, present_bits)

            indices[subgroup] = list(map(lambda bstr: int(bstr, 2), strings))
            
        return indices

    def _gibbs_sampling(self):
        selected_samples = dict()

