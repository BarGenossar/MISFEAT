import numpy as np
from config import MissingDataConfig
from utils import convert_decimal_to_binary


class MissingDataMasking:
    def __init__(self, feature_num, subgroups, config_num, manual=False):
        np.random.seed(int(config_num))
        self.seed = int(config_num)
        self.feature_num = feature_num
        self.subgroups = subgroups
        self.manual = manual
        self.general_missing_prob = MissingDataConfig.general_missing_prob
        # TODO: Add missing_rate_dict
        self.missing_indices_dict = self._set_missing_indices_dict()

    def _set_missing_indices_dict(self):
        missing_indices_dict = {subgroup: dict() for subgroup in self.subgroups}
        binary_vecs = [convert_decimal_to_binary(i, self.feature_num) for i in range(2**self.feature_num)]
        if self.manual:
            return self._get_manual_missing_indices_dict(missing_indices_dict, binary_vecs)
        else:
            return self._get_random_missing_indices_dict(missing_indices_dict, binary_vecs)

    def _get_feature_indices(self, feature_idx, binary_vecs):
        return [i - 1 for i in range(len(binary_vecs)) if binary_vecs[i][self.feature_num - feature_idx - 1] == '1']

    def _get_manual_missing_indices_dict(self, missing_indices_dict, binary_vecs):
        for subgroup in self.subgroups:
            print(f"Enter the missing feature indexes for subgroup {subgroup} separated by commas:")
            missing_features = input().split(',')
            # if the input is empty, skip
            if missing_features == ['']:
                continue
            for f_idx in missing_features:
                missing_indices_dict[subgroup][f'f_{f_idx}'] = self._get_feature_indices(int(f_idx), binary_vecs)
            missing_indices_dict[subgroup]['all'] = list(set().union(*missing_indices_dict[subgroup].values()))
        return missing_indices_dict

    def _get_random_missing_indices_dict(self, missing_indices_dict, binary_vecs):
        for subgroup in self.subgroups:
            for f_idx in range(self.feature_num):
                if np.random.rand() < self.general_missing_prob:
                    missing_indices_dict[subgroup][f'f_{f_idx}'] = self._get_feature_indices(f_idx, binary_vecs)
            missing_indices_dict[subgroup]['all'] = list(set().union(*missing_indices_dict[subgroup].values()))
        return missing_indices_dict