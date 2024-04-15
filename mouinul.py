import argparse
import pandas as pd
from missing_data_masking import MissingDataMasking
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(data, test_indices, model, g_id, at_k, comb_size, feature_num):
    data.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
    labels = data[g_id].y
    predictions = out[g_id]
    tmp_results_dict = compute_eval_metrics(labels, predictions, at_k, comb_size, feature_num)
    print_results(tmp_results_dict, at_k, comb_size, g_id)
    return tmp_results_dict

def convert_decimal_to_binary(decimal, feature_num):
    binary = bin(decimal)[2:]
    return '0' * (feature_num - len(binary)) + binary


def convert_binary_to_tuple(binary):
    combination = []
    for pos, bit in enumerate(binary[::-1]):
        if bit == '1':  combination.append(f'f_{pos}')
    return tuple(combination)



def get_test_nodes(missing_indices_dict, subgroup, comb_size, feature_num):
    all_test_node_indices = missing_indices_dict[subgroup]['all']
    node_tuples = []
    node_indices = []
    for idx in all_test_node_indices:
        binary_vec = convert_decimal_to_binary(idx + 1, feature_num)
        if binary_vec.count('1') == comb_size:
            node_tuples.append(convert_binary_to_tuple(binary_vec))
            node_indices.append(idx)
    return node_tuples, node_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds_num', type=int, default=1)
    parser.add_argument('--sampling_ratio', type=float, default=0.33)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size', type=int, default=3)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--workers', default=8, type=int, help="number of parallel workers")
    args = parser.parse_args()

    ## load dataset (Mouinul: modify path)
    data_path = 'GeneratedData/Formula2/Config2/dataset.pkl'
    df = pd.read_pickle(data_path)
    base_features = [feat for feat in list(df.columns) if 'f_' in feat]
    feature_num = len(base_features)
    subgroups = [f'g{gid}' for gid in range(df.subgroup.nunique())]

    ## load hetero data (Mouinul: modify path)
    graph_path = "GeneratedData/Formula2/Config2/dataset_hetero_graph.pt"
    data = torch.load(graph_path)


    for seed in range(1, args.seeds_num + 1):
        set_seed(seed)

        missing_indices_dict = MissingDataMasking(feature_num, subgroups, args.manual_md).missing_indices_dict
        """
        structure of missing_indices_dict:
            subgroup ('g0'):
                1st missing ('f_7'): [list of node indices containing f_7]
                2nd missing ('f_9'): ...
                            ...
                            ('all'): [combine all lists above]
            subgroup ('g1')
                ...
        """
        with open(f'missing_indices_dict_seed{seed}.txt', 'w') as f:
            f.write(str(missing_indices_dict))

        model = None
        for subgroup in subgroups:
            ## get `indices of all test nodes` of the given `args.comb_size`
            node_tuples, node_indices = get_test_nodes(missing_indices_dict, subgroup, args.comb_size, feature_num)
            scores = list(map(lambda idx: data[subgroup].y[idx].item(), node_indices))   # this is MI score of the test nodes

            for tup, score in zip(node_tuples, scores):
                print(tup, score)


            # results_dict[seed][subgroup] = test(data, test_indices, model, subgroup, args.at_k, comb_size, feature_num)
    # save_results(results_dict, dir_path, comb_size, args)

    # TODO (done): txt file {subgroup: [missing_features]}
    # TODO (done): a function(comb_size) -> test_nodes with corresponding MI scores


"""
NOTE:
1. you may see a feature missing across all subgroups because this condition is not included in the code. 
   Therefore, pls check carefully if this happens in every seed.
2. remember to run the lattice_graph_generator.py  before running this code
3. make sure you modify data_path (line 55) and graph_path (line 62) when running this code
"""
