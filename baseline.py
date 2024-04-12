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
            imputer = CustomKNNImputer(n_neighbors=15)
            print(data.head())
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif self.imputation_method == 'MICE':
            imputer = IterativeImputer()
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif self.imputation_method == 'constant':
            print("imputation is constatn")
            imputer = SimpleImputer(strategy='constant', fill_value=1000)
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
        # score = normalized_mutual_info_score(comb_series, df_subgroup['y']) + 0.5
        score = mutual_info_score(comb_series, df_subgroup['y'])
        scores.append(round(score, 5))
    return scores


# def get_feature_importance(df_subgroup: pd.DataFrame, combinations: t.List[t.Tuple[str]]):
#     """
#     Example:
#         combinations = [('f_0', 'f_1'), ('f_5', 'f_6'), ('f_2', 'f_3')]
#         output = [feature importance scores of the combinations]
#     """
#     df_subgroup = df_subgroup.astype(str)
#     # TODO: create new dataframe with new columns as combinations
#     # run random forest and get feature importance
#     new_df = [df_subgroup['y']]
#     for comb in combinations:
#         new_series = df_subgroup[list(comb)].apply(lambda x: ''.join(x), axis=1)
#         new_df.append(new_series)
#     new_df = pd.concat(new_df, axis=1)

    #     score = mutual_info_score(comb_series, df_subgroup['y'])  # TODO: replace this line with Random Forest to compute feature importance
    #     scores.append(round(score, 5))
    # return 

# def compute_dcg(ground_truth, sorted_indices, at_k):
#     DCG = 0
#     for i in range(1, min(at_k + 1, len(sorted_indices))):
#         DCG += (math.pow(2, ground_truth[sorted_indices[i-1]].item()) - 1) / math.log2(i+1)
#     return DCG


# def compute_ndcg(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices):
#     IDCG = compute_dcg(ground_truth, sorted_gt_indices, k)
#     DCG = compute_dcg(ground_truth, sorted_pred_indices, k)
#     return round(DCG / IDCG, 4)


def compute_precision(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices):
    precision = len(set.intersection(set(sorted_gt_indices[:k]), set(sorted_pred_indices[:k])))
    return round(precision / k, 4)


def compute_RMSE(ground_truth, predictions, k, sorted_gt_indices, sorted_pred_indices):
    # Implement normalized MAE, such that the difference is divided by the maximum possible difference
    rmse = sum([(ground_truth[sorted_gt_indices[i]] -
                   predictions[sorted_gt_indices[i]])**2 for i in range(k)])
    return round(math.sqrt(rmse.item() / k), 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline experiments')
    parser.add_argument('--data_name', type=str, default='loan', help='name of the dataset')
    parser.add_argument('--imputation_method', type=str, default='KNN', choices=['mode', 'KNN', 'MICE','constant'],
                        help='Imputation method to use: mode, KNN, or MICE. Default is mode.')
    parser.add_argument('--comb_size', type=int, default=2, help='combination size to be tested')
    parser.add_argument('--seed', type=int, default=1, help='combination size to be tested')
    args = parser.parse_args()

    df_true = pd.read_pickle(f'./RealWorldData/{args.data_name}/dataset.pkl')

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
    results = {metric: {} for metric in ['NDCG', 'PREC', 'RMSE']}
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
        mi[subgroup]['true'] = get_mi_score(df_true[df_true.subgroup == g_id], subgroup_combs)   # array of MI scores
        mi[subgroup]['miss'] = get_mi_score(df_miss[df_miss.subgroup == g_id], subgroup_combs)   # array of MI scores
        array_1 = mi[subgroup]['true']
        array_2 = mi[subgroup]['miss']

        
        if len(array_1) == 0:
            continue

        print(subgroup)
        # import torch.nn.functional as F
        # array_1 = F.softmax(torch.tensor(array_1))
        # array_2 = F.softmax(torch.tensor(array_2))
        # print(array_1)
        # print(array_2)
        # exit()
        at_k = 5
        comb_size = 3
        
        true_rank = torch.argsort(torch.tensor(mi[subgroup]['true']), descending=True).tolist()
        relevance = [1 if rank < 5 else 0 for rank in true_rank]
        # pred_rank = torch.argsort(torch.tensor(mi[subgroup]['miss']), descending=True).tolist()
        # print(true_rank)
        # print()
        results['NDCG'][at_k] = ndcg_score(np.array([relevance]), np.array([array_2]), k=at_k)
        print(f"subgroup: {subgroup}, combSize: {comb_size}, nDCG @ {at_k}: {results['NDCG'][at_k]}")
        exit()



        
        
        # results['NDCG'][at_k] = compute_ndcg(mi[subgroup]['true'], mi[subgroup]['miss'], at_k, true_rank, pred_rank)
        # results['PREC'][at_k] = compute_precision(np.array([array_1]), np.array([array_2]), at_k, true_rank, pred_rank)

        # print(f"subgroup: {subgroup}, at_k: {at_k}, combSize: {comb_size}, precision: {results['PREC'][at_k]}")


        # print(subgroup_results)



"""
For Mouinul's experiments:
comb size = {3, 4, 5}
at_k = {3, 5, 10, 20}

Complete the evaluation framework for baseline

"""