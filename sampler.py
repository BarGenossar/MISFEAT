import random
import numpy as np
import typing as t


class NodeSampler:
    def __init__(self, subgroups, feature_num, missing_indices_dict, sampling_ratio, sampling_method='random'):
        self.subgroups = subgroups
        self.feature_num = feature_num
        self.missing_indices_dict = missing_indices_dict
        self.sampling_ratio = sampling_ratio
        self.sampling_method = 'random' if self.sampling_ratio == 1 else sampling_method
        self.selected_samples = self._get_selected_samples()

    def _get_selected_samples(self):
        sampling_funcs = {
            'random': self._random_sampling,
            'uniform': self._uniform_sampling,
            # 'gibbs': self._gibbs_sampling,
        }
        if self.sampling_method not in sampling_funcs:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}")
        return sampling_funcs[self.sampling_method]()

    def _random_sampling(self):
        indices = dict()
        for subgroup in self.subgroups:
            non_missing_idx = sorted(list( set(range(2**self.feature_num - 1)) - set(self.missing_indices_dict[subgroup]) ))
            num_samples = int(self.sampling_ratio * len(non_missing_idx))
            indices = np.random.choice(non_missing_idx, num_samples, replace=False)
        return list(indices)

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
            missing_features = [int(feat.split('_')[-1]) for feat in self.missing_indices_dict[subgroup].keys() if 'f_' in feat]
            non_missing_features = sorted(list( set(range(self.feature_num)) - set(missing_features) ))
            num_samples = int(self.sampling_ratio * (2**len(non_missing_features) - 1))
            ## sample a starting node
            random_features = np.random.choice([0, 1], size=len(non_missing_features), p=[0.5, 0.5])   # for each non-missing features, select it with prob 0.5
            start_node = [0] * self.feature_num
            for idx, bit in enumerate(random_features):
                start_node[self.feature_num-1 - non_missing_features[idx]] = bit
            start_node = ''.join([str(bit) for bit in start_node])
            ## random walk in the hypercube
            strings = [start_node]
            while len(strings) < num_samples:
                self._random_walk(strings, non_missing_features)

            indices[subgroup] = list(map(lambda bstr: int(bstr, 2), strings))
            
        return indices

    def _gibbs_sampling(self):
        f""" TBD """