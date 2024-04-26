import random
import numpy as np
import typing as t
from config import Sampling
from sklearn.model_selection import train_test_split


class NodeSampler:
    def __init__(self, config_num, feature_num, non_missing_dict, missing_indices_dict, restricted_graph_idxs_mapping,
                 sampling_ratio, sampling_method):
        np.random.seed(config_num)
        self.seed = config_num
        self.subgroups = non_missing_dict.keys()
        self.feature_num = feature_num
        self.non_missing_dict = non_missing_dict
        self.missing_indices_dict = missing_indices_dict
        self.restricted_graph_idxs_mapping = restricted_graph_idxs_mapping
        self.sampling_ratio = sampling_ratio
        self.sampling_method = sampling_method
        self.validation_ratio = Sampling.validation_ratio
        self.sampling_func = self._get_sampling_func_dict()
        self.train_indices_dict, self.val_indices_dict = self._get_samples()

    def _get_sampling_func_dict(self):
        sampling_funcs = {
            'arbitrary': self._arbitrary_sampling,
            'randwalk': self._uniform_sampling,
        }
        if self.sampling_method not in sampling_funcs:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}")
        return sampling_funcs[self.sampling_method]

    def _get_samples(self):
        if self.sampling_ratio < 1:
            sampled_indices_dict = self.sampling_func()
        else:
            sampled_indices_dict = self.non_missing_dict
        train_indices_dict, val_indices_dict = dict(), dict()
        for g_id, indices in sampled_indices_dict.items():
            train_indices, val_indices = train_test_split(indices, test_size=self.validation_ratio)
            train_indices_dict[g_id] = train_indices
            val_indices_dict[g_id] = val_indices
        return train_indices_dict, val_indices_dict

    def _arbitrary_sampling(self):
        sampled_indices_dict = dict()
        for g_id in self.subgroups:
            num_samples = int(self.sampling_ratio * len(self.non_missing_dict[g_id]))
            sampled_indices_dict[g_id] = list(np.random.choice(self.non_missing_dict[g_id], num_samples, replace=False))
        return sampled_indices_dict

    def _random_walk(self, node_list: t.List[str], present_bits: t.List[int]) -> None:
        curr_node = node_list[-1]
        rand_idx = self.feature_num-1 - random.choice(present_bits)
        rand_bit = np.random.choice(['0', '1'], p=[0.5, 0.5])
        if rand_bit != curr_node[rand_idx]:
            new_node =curr_node[ : rand_idx] + rand_bit + curr_node[rand_idx + 1 : ]
            if new_node != '0' * self.feature_num:    # ensure non-empty comb
                node_list.append(new_node)

    def _uniform_sampling(self):
        indices = dict()
        for subgroup in self.subgroups:
            missing_features = [int(feat.split('_')[-1]) for feat in self.missing_indices_dict[subgroup].keys() if
                                'f_' in feat]
            non_missing_features = sorted(list(set(range(self.feature_num)) - set(missing_features)))
            num_samples = int(self.sampling_ratio * (2 ** len(non_missing_features) - 1))
            ## sample a starting node
            random_features = np.random.choice([0, 1], size=len(non_missing_features),
                                               p=[0.5, 0.5])  # for each non-missing features, select it with prob 0.5
            start_node = [0] * self.feature_num
            for idx, bit in enumerate(random_features):
                start_node[self.feature_num - 1 - non_missing_features[idx]] = bit
            start_node = ''.join([str(bit) for bit in start_node])
            ## random walk in the hypercube
            strings = [start_node]
            while len(strings) < num_samples:
                self._random_walk(strings, non_missing_features)

            indices[subgroup] = list(map(lambda bstr: int(bstr, 2), strings))

        return indices
