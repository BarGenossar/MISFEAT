import json
import argparse
import typing as t
import numpy as np
import pandas as pd
from itertools import combinations
# from fancyimpute import IterativeImputer
from sklearn.metrics import mutual_info_score
from sklearn.impute import SimpleImputer, KNNImputer


class DataImputer:
    def __init__(self, missing_dict, imputation_method='mode'):
        self.missing_dict = missing_dict
        self.imputation_method = imputation_method

    def _replace_with_nan(self, data):
        for subgroup, features in self.missing_dict.items():
            g_id = int(subgroup.split('g')[-1])
            data.loc[data['subgroup'] == g_id, features] = np.nan
        return data

    def _impute_data(self, data):
        if self.imputation_method == 'mode':
            for column in data.columns:
                imputer = SimpleImputer(strategy='most_frequent')
                data[column] = imputer.fit_transform(data[[column]])
        elif self.imputation_method == 'KNN':
            imputer = KNNImputer(n_neighbors=5)
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif self.imputation_method == 'MICE':
            imputer = IterativeImputer()
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        else:
            raise ValueError("Unsupported imputation method")
        
        return data

    def process(self, data):
        data = self._replace_with_nan(data)
        data = self._impute_data(data)
        return data


def get_mi_score(df_subgroup: pd.DataFrame, combinations: t.List[t.Tuple[str]]):
    """
    Example:
        combinations = [('f_0', 'f_1'), ('f_5', 'f_6'), ('f_2', 'f_3')]
        output = [MI scores of the feature tuples]
    """
    df_subgroup = df_subgroup.astype(str)
    scores = []
    for comb in combinations:
        comb_series = df_subgroup[list(comb)].apply(lambda x: ''.join(x), axis=1)
        score = mutual_info_score(comb_series, df_subgroup['y'])
        scores.append(round(score, 5))
    return scores


def get_feature_importance(df_subgroup: pd.DataFrame, combinations: t.List[t.Tuple[str]]):
    """
    Example:
        combinations = [('f_0', 'f_1'), ('f_5', 'f_6'), ('f_2', 'f_3')]
        output = [feature importance scores of the combinations]
    """
    df_subgroup = df_subgroup.astype(str)
    # TODO: create new dataframe with new columns as combinations
    # run random forest and get feature importance
    new_df = [df_subgroup['y']]
    for comb in combinations:
        new_series = df_subgroup[list(comb)].apply(lambda x: ''.join(x), axis=1)
        new_df.append(new_series)
    new_df = pd.concat(new_df, axis=1)

    #     score = mutual_info_score(comb_series, df_subgroup['y'])  # TODO: replace this line with Random Forest to compute feature importance
    #     scores.append(round(score, 5))
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline experiments')
    parser.add_argument('--data_name', type=str, default='loan', help='name of the dataset')
    parser.add_argument('--imputation_method', type=str, default='mode', choices=['mode', 'KNN', 'MICE'],
                        help='Imputation method to use: mode, KNN, or MICE. Default is mode.')
    parser.add_argument('--comb_size', type=int, default=2, help='combination size to be tested')
    args = parser.parse_args()

    df_true = pd.read_pickle(f'./RealWorldData/{args.data_name}/dataset.pkl')

    ## load missing dict
    with open(f'./RealWorldData/{args.data_name}/missing.json', 'r') as file:
        missing_dict = json.load(file)

    ## imputation
    df_miss = df_true.copy(deep=True)
    # imputer = DataImputer(missing_dict, args.imputation_method)
    # df_miss = imputer.process(df_miss)
    # df_miss = df_miss.astype(int)

    ## dataframe information
    base_features = [feat for feat in df_true.columns if 'f_' in feat]
    feature_num = len(base_features)
    subgroups = [f'g{g_id}' for g_id in range(len(set(df_true.subgroup)))]
    
    ## get test results
    results = {subgroup: {} for subgroup in subgroups}
    for subgroup in subgroups:
        g_id = int(subgroup.split('g')[-1])
        subgroup_combs = []
        for feat in missing_dict[subgroup]:
            base_features.remove(feat)
            combs = combinations(base_features, args.comb_size - 1)
            combs = [tuple(sorted([feat] + list(comb))) for comb in combs]
            subgroup_combs.extend(combs)
            base_features.append(feat)

        subgroup_combs = list(set(subgroup_combs))   # remove duplicates
        results[subgroup]['true'] = get_mi_score(df_true[df_true.subgroup == g_id], subgroup_combs)
        # results[subgroup]['miss'] = get_mi_score(df_miss[df_miss.subgroup == g_id], subgroup_combs)

    print(results)
