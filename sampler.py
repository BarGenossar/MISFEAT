import numpy as np
from config import MissingDataConfig
from utils import convert_decimal_to_binary


class NodeSampler:
    def __init__(self, config_num, non_missing_indices, sampling_ratio, sampling_method='random'):
        np.random.seed(int(config_num))
        self.seed = config_num
        self.non_missing_idx = non_missing_indices
        self.sampling_ratio = sampling_ratio
        self.sampling_method = sampling_method
        self.selected_samples = self._get_selected_samples(sampling_method)


    def _get_selected_samples(self, sampling_method):
        # TODO: Implement other sampling methods
        sampling_funcs = {'random': self._random_sampling}
        if sampling_method not in sampling_funcs:
            raise ValueError(f"Invalid sampling method: {sampling_method}")
        return sampling_funcs[sampling_method]()

    def _random_sampling(self):
        # selected_samples = dict()
        num_samples = int(self.sampling_ratio * len(self.non_missing_idx))
        selected_samples = np.random.choice(self.non_missing_idx, num_samples, replace=False)
        return list(selected_samples)
