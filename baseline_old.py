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


def compute_NDCG(mi_true, mi_pred, k, true_rank, pred_rank):
    relevance = [0] * len(mi_true)
    if config.Evaluation.binary_relevance:
        for i in range(k): relevance[true_rank[i]] = 1
    else:
        for i in range(k): relevance[true_rank[i]] = k - i
    DCG = 0.
    IDCG = 0.
    for i in range(k):
        if config.Evaluation.binary_relevance:
            IDCG += 1 / math.log(i + 2, 2)
        else:
            IDCG += (k - i) / math.log(i + 2, 2)
        DCG += relevance[pred_rank[i]] / math.log(i + 2, 2)
    return round(DCG / IDCG, 4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline experiments')
    parser.add_argument('--data_name', type=str, default='loan', help='name of the dataset')
    parser.add_argument('--missing_prob', type=float, default=0.1, help='missing probability')
    parser.add_argument('--imputation_method', type=str, default='KNN', choices=['mode', 'KNN', 'MICE','constant'],
                        help='Imputation method to use: mode, KNN, or MICE. Default is mode.')
    # parser.add_argument('--comb_size', type=int, default=5, help='combination size to be tested')
    args = parser.parse_args()


    df_true = pd.read_pickle(f'./RealWorldData/{args.data_name}/dataset.pkl')
    # print(df_true)
    # exit()
    ## dataframe information
    base_features = [feat for feat in df_true.columns if 'f_' in feat]
    feature_num = len(base_features)
    subgroups = [f'g{g_id}' for g_id in range(len(set(df_true.subgroup)))]
    print('num records:', len(df_true))


    # for gid in range(5):
    #     print('group', gid)
    #     print(df_true[df_true.subgroup == gid])
    # exit()


    seeds = [1, 2, 3]
    # seeds = [32, 33, 34]
    results = {comb_size: {subgroup: {metric: {at_k: [0]*len(seeds) for at_k in config.Evaluation.at_k} for metric in ['NDCG', 'PREC']} for subgroup in subgroups} for comb_size in [3, 4, 5]}
    for idx, seed in enumerate(seeds):
        with open(f'./RealWorldData/{args.data_name}/missing_seed{seed}_{args.missing_prob}.json', 'r') as file:
            missing_dict = json.load(file)

        ## imputation
        df_miss = df_true.copy(deep=True)
        imputer = DataImputer(missing_dict, args.imputation_method)
        print("Imputing missing data...")
        df_miss = imputer.process(df_miss)
        df_miss = df_miss.astype(int)

        
        ## get test results
        for subgroup in subgroups:
            print(f'Computing subgroup: {subgroup}...')
            g_id = int(subgroup.split('g')[-1])

            for comb_size in [3, 4, 5]:
                subgroup_combs = []
                for feat in missing_dict[subgroup]:
                    base_features.remove(feat)
                    combs = combinations(base_features, comb_size - 1)
                    combs = [tuple(sorted([feat] + list(comb))) for comb in combs]
                    subgroup_combs.extend(combs)
                    base_features.append(feat)


                subgroup_combs = sorted(list(set(subgroup_combs)))   # remove duplicates
                mi_true_array = get_mi_score(df_true[df_true.subgroup == g_id], subgroup_combs)   # array of MI scores
                mi_pred_array = get_mi_score(df_miss[df_miss.subgroup == g_id], subgroup_combs)   # array of MI scores


                true_rank = np.argsort(mi_true_array)[::-1]
                pred_rank = np.argsort(mi_pred_array)[::-1]

                if comb_size == 5:
                    for r in range(5):
                        print(f'top {r+1}, true: {subgroup_combs[true_rank[r]]}, pred: {subgroup_combs[pred_rank[r]]}')

                for at_k in config.Evaluation.at_k:
                    results[comb_size][subgroup]['NDCG'][at_k][idx] = compute_NDCG(mi_true_array, mi_pred_array, at_k, true_rank, pred_rank)
                    results[comb_size][subgroup]['PREC'][at_k][idx] = compute_precision(mi_true_array, mi_pred_array, at_k, true_rank, pred_rank)


    for comb_size in [3, 4, 5]:
        print(f"Comb size = {comb_size}")
        for at_k in config.Evaluation.at_k:
            print(f"\tnDCG @ {at_k}")
            avg = 0.
            for subgroup in subgroups:
                avg += sum(results[comb_size][subgroup]['NDCG'][at_k])/len(results[comb_size][subgroup]['NDCG'][at_k])
                # print(f"\t\tsubgroup: {subgroup} = {round(sum(results[subgroup]['NDCG'][at_k])/len(results[subgroup]['NDCG'][at_k]), 2)}")
            print(f"\t\tavg: {avg/len(subgroups)}")

        for at_k in config.Evaluation.at_k:
            print(f"\tprecision @ {at_k}")
            avg = 0.
            for subgroup in subgroups:
                avg += sum(results[comb_size][subgroup]['PREC'][at_k])/len(results[comb_size][subgroup]['PREC'][at_k])
                # print(f"\t\tsubgroup: {subgroup} = {round(sum(results[subgroup]['PREC'][at_k])/len(results[subgroup]['PREC'][at_k]), 2)}")
            print(f"\t\tavg: {avg/len(subgroups)}")
        


    