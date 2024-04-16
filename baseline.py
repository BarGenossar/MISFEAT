import json
import argparse
import typing as t
import numpy as np
import pandas as pd
from itertools import combinations
# from fancyimpute import IterativeImputer
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from sklearn.impute import SimpleImputer, KNNImputer
#from utils import compute_eval_metrics, compute_ndcg, compute_precision, compute_RMSE
import torch
from Custom_KNN_Imputer import CustomKNNImputer
import math
from sklearn.metrics import ndcg_score
from utils import set_seed, Kendall_tau
import config


class DataImputer:
    def __init__(self, missing_dict, imputation_method='mode'):
        self.missing_dict = missing_dict
        self.imputation_method = imputation_method

    def _replace_with_nan(self, data):
        for subgroup, features in self.missing_dict.items():
            g_id = int(subgroup.split('g')[-1])
            data.loc[data['subgroup'] == g_id, features] = np.nan
            # print(features)
            # print(data.loc[data['subgroup'] == g_id, features])
            # exit()
        return data

    def _impute_data(self, data):
        if self.imputation_method == 'mode':
            for column in data.columns:
                imputer = SimpleImputer(strategy='most_frequent')
                data[column] = imputer.fit_transform(data[[column]])
        elif self.imputation_method == 'KNN':
            imputer = CustomKNNImputer(n_neighbors=3)
            # print(data.head())
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif self.imputation_method == 'MICE':
            imputer = IterativeImputer()
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif self.imputation_method == 'constant':
            print("imputation is constant")
            imputer = SimpleImputer(strategy='constant', fill_value=1000)
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        else:
            raise ValueError("Unsupported imputation method")
        
        # print(set(data.loc[data['subgroup'] == 0, ['f_1',]]['f_1']))
        # exit()
        
        return data

    def process(self, data):
        data = self._replace_with_nan(data)
        data = self._impute_data(data)
        return data


def log2(num):
    return math.log(num, 2) if num > 0 else 0


def entropy(distr: list) -> float:
    """
    calculate entropy of a random variable given its distribution
    Args:
        `distr`: distribution of the random variable
    """
    sum = 0.
    for p in distr:
        sum -= p * log2(p)
    return sum



# def get_mi_score(df_subgroup: pd.DataFrame, combinations: t.List[t.Tuple[str]]) -> float:
#     """
#     generate combined feature using base features, then calculate MI score, and cache results
#     Args:
#         `df`: input as pandas DataFrame obj
#         `bin_str`: binary string representing a combination of base features
#         `subpop`: graph ID (subpopulation ID)
#         `combinations`: = [('f_0', 'f_1'), ('f_5', 'f_6'), ('f_2', 'f_3')]
#     Return:
#         normalized mutual information between Z and bin_str feature
#     Example:
#         bin_str = "00110" means the combined features include base features 2 and 3 (zero-indexed)
#         return normalized MI(Z; a2_a3)
#     """
#     def _compute_mi(comb):
#         ## contingency table
#         table = pd.crosstab(
#             columns=[df_subgroup[col] for col in comb],
#             index=df_subgroup['y'],
#             margins=True
#         ).values

#         distr_x = [val / table[-1, -1] for val in table[-1][:-1]]     # distribution of attributes
#         distr_z = [val / table[-1, -1] for val in table[:, -1][:-1]]  # distribution of label

#         # entropy_x = entropy(distr_x)
#         entropy_z = entropy(distr_z)
        
#         entropy_z_given_x = -sum([p * sum([ table[label, idx] / table[-1, idx] * log2(table[label, idx] / table[-1, idx]) for label in range(len(table) - 1) ]) 
#                             for idx, p in enumerate(distr_x)])
#         return round(entropy_z - entropy_z_given_x, 5)

#     mi = list( map(_compute_mi, combinations) )
    
#     return mi





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
        # score = normalized_mutual_info_score(comb_series, df_subgroup['y']) + 0.5
        score = mutual_info_score(comb_series, df_subgroup['y'])
        scores.append(round(score, 5))
    return scores


def compute_precision(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices):
    precision = len(set.intersection(set(sorted_gt_indices[:k]), set(sorted_pred_indices[:k])))
    return round(precision / k, 4)


def compute_RMSE(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices):
    # Implement normalized MAE, such that the difference is divided by the maximum possible difference
    rmse = sum([(ground_truth[sorted_gt_indices[i]] -
                   predictions[sorted_gt_indices[i]])**2 for i in range(k)])
    return round(math.sqrt(rmse.item() / k), 4)


def compute_NDCG(mi_true, mi_pred, k):
    rank_true = np.argsort(mi_true)[::-1]
    rank_pred = np.argsort(mi_pred)[::-1]
    relevance = [0] * len(mi_true)
    if config.Evaluation.binary_relevance:
        for i in range(k): relevance[rank_true[i]] = 1
    else:
        for i in range(k): relevance[rank_true[i]] = k - i
    DCG = 0.
    IDCG = 0.
    for i in range(k):
        if config.Evaluation.binary_relevance:
            IDCG += 1 / math.log(i + 2, 2)
        else:
            IDCG += (k - i) / math.log(i + 2, 2)
        DCG += relevance[rank_pred[i]] / math.log(i + 2, 2)
    return round(DCG / IDCG, 4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline experiments')
    parser.add_argument('--data_name', type=str, default='loan', help='name of the dataset')
    parser.add_argument('--imputation_method', type=str, default='KNN', choices=['mode', 'KNN', 'MICE','constant'],
                        help='Imputation method to use: mode, KNN, or MICE. Default is mode.')
    parser.add_argument('--comb_size', type=int, default=5, help='combination size to be tested')
    parser.add_argument('--seed', type=int, default=1, help='combination size to be tested')
    args = parser.parse_args()

    # set_seed(args.seed)

    df_true = pd.read_pickle(f'./RealWorldData/{args.data_name}/dataset.pkl')
    print('num records:', len(df_true))
    # df_true.drop(columns=['f_1', 'f_8'], axis=1, inplace=True)
    # for col in df_true.columns:
    #     print(col, df_true[col].nunique())
    # exit()
    # print(df_true.head(100))
    # exit()

    ## load missing dict
    with open(f'./RealWorldData/{args.data_name}/missing_seed{args.seed}.json', 'r') as file:
        missing_dict = json.load(file)

    ## imputation
    df_miss = df_true.copy(deep=True)
    imputer = DataImputer(missing_dict, args.imputation_method)
    df_miss = imputer.process(df_miss)
    df_miss = df_miss.astype(int)

    ## dataframe information
    base_features = [feat for feat in df_true.columns if 'f_' in feat]
    feature_num = len(base_features)
    subgroups = [f'g{g_id}' for g_id in range(len(set(df_true.subgroup)))]
    
    ## get test results
    mi = {subgroup: {} for subgroup in subgroups}
    results = {metric: {} for metric in ['NDCG', 'PREC']}
    for subgroup in subgroups:
        g_id = int(subgroup.split('g')[-1])
        subgroup_combs = []
        for feat in missing_dict[subgroup]:
            base_features.remove(feat)
            combs = combinations(base_features, args.comb_size - 1)
            combs = [tuple(sorted([feat] + list(comb))) for comb in combs]
            subgroup_combs.extend(combs)
            base_features.append(feat)


        subgroup_combs = sorted(list(set(subgroup_combs)))   # remove duplicates
        mi[subgroup]['true'] = get_mi_score(df_true[df_true.subgroup == g_id], subgroup_combs)   # array of MI scores
        mi[subgroup]['miss'] = get_mi_score(df_miss[df_miss.subgroup == g_id], subgroup_combs)   # array of MI scores


        # print(mi[subgroup]['true'])
        # print(mi[subgroup]['miss'])
        # exit()


        true_rank = np.argsort(mi[subgroup]['true'])[::-1]
        pred_rank = np.argsort(mi[subgroup]['miss'])[::-1]

        if len(mi[subgroup]['true']) == 0:
            continue

        for at_k in config.Evaluation.at_k:
            results['NDCG'][at_k] = compute_NDCG(mi[subgroup]['true'], mi[subgroup]['miss'], at_k)
            print(f"subgroup: {subgroup}, comb_size: {args.comb_size}, nDCG @ {at_k}: {results['NDCG'][at_k]}")
            results['PREC'][at_k] = compute_precision(mi[subgroup]['true'], mi[subgroup]['miss'], at_k, true_rank, pred_rank)
            print(f"subgroup: {subgroup}, comb_size: {args.comb_size}, precision @ {at_k}: {results['PREC'][at_k]}")
            print()

        # print(f"subgroup: {subgroup}, at_k: {at_k}, combSize: {comb_size}, precision: {results['PREC'][at_k]}")


        # print(subgroup_results)



"""
Comb size = 3
    subgroup: g0
        nDCG @ 3: 0.7039
        nDCG @ 5: 1.0
        nDCG @ 10: 0.8572
        precision @ 3: 0.6667
        precision @ 5: 1.0
        precision @ 10: 0.8
    subgroup: g2
        nDCG @ 3: 0.7654
        nDCG @ 5: 0.8304
        nDCG @ 10: 0.7453
        precision @ 3: 0.6667
        precision @ 5: 0.8
        precision @ 10: 0.7
    subgroup: g5
        nDCG @ 3: 1.0
        nDCG @ 5: 0.8688
        precision @ 3: 1.0
        nDCG @ 10: 0.9364
        precision @ 5: 0.8
        precision @ 10: 0.9
    subgroup: g6
        nDCG @ 3: 0.2961
        nDCG @ 5: 0.5531
        nDCG @ 10: 0.5869
        precision @ 3: 0.3333
        precision @ 5: 0.4
        precision @ 10: 0.5



Comb size = 4
    subgroup: g0
        nDCG @ 3: 0.7039
        nDCG @ 5: 0.8688
        nDCG @ 10: 0.9306
        precision @ 3: 0.6667
        precision @ 5: 0.8
        precision @ 10: 0.9
    subgroup: g2
        nDCG @ 3: 0.4693
        nDCG @ 5: 0.6399
        nDCG @ 10: 0.6471
        precision @ 3: 0.3333
        precision @ 5: 0.6
        precision @ 10: 0.6
    subgroup: g5
        nDCG @ 3: 0.2346
        nDCG @ 5: 0.8539
        nDCG @ 10: 0.8358
        precision @ 3: 0.3333
        precision @ 5: 0.8
        precision @ 10: 0.8
    subgroup: g6
        nDCG @ 3: 0.4693
        nDCG @ 5: 0.4852
        nDCG @ 10: 0.7058
        precision @ 3: 0.3333
        precision @ 5: 0.4
        precision @ 10: 0.6



Comb size = 5
subgroup: g0, nDCG @ 3: 1.0
subgroup: g0, nDCG @ 5: 1.0
subgroup: g0, nDCG @ 10: 0.9364
subgroup: g0, precision @ 3: 1.0
subgroup: g0, precision @ 5: 1.0
subgroup: g0, precision @ 10: 0.9

subgroup: g2, nDCG @ 3: 0.2346
subgroup: g2, nDCG @ 5: 0.6548
subgroup: g2, nDCG @ 10: 0.6529
subgroup: g2, precision @ 3: 0.3333
subgroup: g2, precision @ 5: 0.6
subgroup: g2, precision @ 10: 0.6

subgroup: g5, nDCG @ 3: 0.7654
subgroup: g5, nDCG @ 5: 0.8304
subgroup: g5, nDCG @ 10: 0.8643
subgroup: g5, precision @ 3: 0.6667
subgroup: g5, precision @ 5: 0.8
subgroup: g5, precision @ 10: 0.8

subgroup: g6, nDCG @ 3: 0.4693
subgroup: g6, nDCG @ 5: 0.3392
subgroup: g6, nDCG @ 10: 0.5271
subgroup: g6, precision @ 3: 0.3333
subgroup: g6, precision @ 5: 0.2
subgroup: g6, precision @ 10: 0.4


"""