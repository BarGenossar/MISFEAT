import numpy as np
from config import MissingDataConfig
from utils import convert_decimal_to_binary


class MissingDataMasking:
    def __init__(self, feature_num, subgroups, config_num, general_missing_prob, manual=False):
        np.random.seed(config_num)
        self.seed = config_num
        self.feature_num = feature_num
        self.subgroups = subgroups
        self.manual = manual
        self.general_missing_prob = general_missing_prob
        self.max_missing_ratio = MissingDataConfig.max_missing_ratio
        self.missing_indices_dict = self._set_missing_indices_dict()

    def _set_missing_indices_dict(self):
        missing_indices_dict = {subgroup: dict() for subgroup in self.subgroups}
        binary_vecs = [convert_decimal_to_binary(i+1, self.feature_num) for i in range(2**self.feature_num+1)]
        if self.manual:
            return self._get_manual_missing_indices_dict(missing_indices_dict, binary_vecs)
        else:
            return self._get_random_missing_indices_dict(missing_indices_dict, binary_vecs)

    def _get_feature_indices(self, feature_idx, binary_vecs):
        return [i for i in range(len(binary_vecs)) if binary_vecs[i][self.feature_num - feature_idx - 1] == '1']

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
                if len(missing_indices_dict[subgroup]) > self.max_missing_ratio * self.feature_num:
                    break
                if np.random.rand() < self.general_missing_prob:
                    missing_indices_dict[subgroup][f'f_{f_idx}'] = self._get_feature_indices(f_idx, binary_vecs)
            missing_indices_dict = self._handle_non_missing_indices(missing_indices_dict, subgroup, binary_vecs)
            missing_indices_dict[subgroup]['all'] = list(set().union(*missing_indices_dict[subgroup].values()))
        return missing_indices_dict

    def _handle_non_missing_indices(self, missing_indices_dict, subgroup, binary_vecs):
        if len(missing_indices_dict[subgroup]) == 0:
            f_idx = np.random.randint(0, self.feature_num)
            missing_indices_dict[subgroup][f'f_{f_idx}'] = self._get_feature_indices(f_idx, binary_vecs)
        return missing_indices_dict
