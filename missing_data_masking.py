import numpy as np
from config import MissingDataConfig
from utils import convert_decimal_to_binary


class MissingDataMasking:
    def __init__(self, feature_num, subgroups, config_num):
        np.random.seed(int(config_num))
        self.seed = int(config_num)
        self.feature_num = feature_num
        self.subgroups = subgroups
        self.general_missing_prob = MissingDataConfig.general_missing_prob
        # TODO: Add missing_rate_dict
        self.missing_indices_dict = self._set_missing_indices_dict()

    def _set_missing_indices_dict(self):
        missing_indices_dict = {subgroup: dict() for subgroup in self.subgroups}
        binary_vecs = [convert_decimal_to_binary(i, self.feature_num) for i in range(2**self.feature_num)]
        for subgroup in self.subgroups:
            for feature_idx in range(self.feature_num):
                if np.random.rand() < self.general_missing_prob:
                    missing_indices_dict[subgroup][f'f_{feature_idx}'] = self._get_feature_indices(feature_idx,
                                                                                                   binary_vecs)
                #  missing_indices_dict[subgroup]['all'] is a list of indices appear at least once in the subgroup
                missing_indices_dict[subgroup]['all'] = list(set().union(*missing_indices_dict[subgroup].values()))
        return missing_indices_dict

    def _get_feature_indices(self, feature_idx, binary_vecs):
        return [i for i in range(len(binary_vecs)) if binary_vecs[i][self.feature_num - feature_idx - 1] == '1']
