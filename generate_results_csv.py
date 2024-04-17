
import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling, MissingDataConfig
from missing_data_masking import MissingDataMasking
from sampler import NodeSampler
from utils import *
import pandas as pd
from torch_geometric.nn import to_hetero
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_results_dict(results_dict, tmp_dict, eval_metrics, at_k, comb_size):
    for eval_metric in eval_metrics:
        for k in at_k:
            for subgroup in tmp_dict.keys():
                results_dict[eval_metric][k][comb_size] += tmp_dict[subgroup][eval_metric][k]
    return results_dict


def generate_df(dir_path, final_config_dict, eval_metrics, comb_size_list, at_k):
    data = []
    index = []
    columns = [f"comb_size={comb_size}" for comb_size in comb_size_list]

    for eval_metric in eval_metrics:
        for k in at_k:
            index.append(f"{eval_metric}@{k}")
            data.append([final_config_dict[eval_metric][k][comb_size] for comb_size in comb_size_list])
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_csv(f"{dir_path}Config{config_idx}_sampling={sampling_ratio}_missing_ratio={missing_prob}.csv")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds_num', type=int, default=3)
    parser.add_argument('--missing_prob', type=str, default=str(MissingDataConfig.missing_prob))
    parser.add_argument('--sampling_ratio', type=str, default=str(Sampling.sampling_ratio))
    parser.add_argument('--sampling_method', type=str, default=Sampling.method)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size_list', type=int, default=Evaluation.comb_size_list)
    parser.add_argument('--dir_path', type=str, default='GeneratedData/', help='Path to the results directory')
    args = parser.parse_args()


    formula_idx_list = Evaluation.formula_idx_list
    config_idx_list = Evaluation.config_idx_list
    missing_prob = args.missing_prob
    sampling_ratio = args.sampling_ratio
    seeds_num = args.seeds_num
    eval_metrics = Evaluation.eval_metrics
    comb_size_list = args.comb_size_list
    at_k = verify_at_k(args.at_k)
    dir_path = args.dir_path


    for config_idx in config_idx_list:
        config_results_dict = {eval_metric: {k: {comb_size: 0.0 for comb_size in comb_size_list} for k in at_k}
                               for eval_metric in eval_metrics}
        for formula_idx in formula_idx_list:
            for comb_size in args.comb_size_list:
                pkl_path = (f"{dir_path}Formula{formula_idx}/Config{config_idx}/results_size={comb_size}|"
                            f"sampling={sampling_ratio}|missing_ratio={missing_prob}.pkl")
                tmp_dict = pd.read_pickle(pkl_path)
                config_results_dict = update_results_dict(config_results_dict, tmp_dict, eval_metrics, at_k, comb_size)
        subgroup_num = len(tmp_dict.keys())
        final_config_dict = {eval_metric: {k: {comb_size: round(config_results_dict[eval_metric][k][comb_size] /
                                                                (subgroup_num*len(formula_idx_list)), 3)
                                               for comb_size in comb_size_list}
                                           for k in at_k} for eval_metric in eval_metrics}
        generate_df(dir_path, final_config_dict, eval_metrics, comb_size_list, at_k)

