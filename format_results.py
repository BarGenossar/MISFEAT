import pickle
import matplotlib.pyplot as plt

path = './RealWorldData/loan'
comb_size = 5


if __name__ == "__main__":
    metrics = ['NDCG', 'PREC', 'RMSE']
    ratios = [0.25, 0.5, 0.75]
    scores = {sampling_ratio: {k: {metric: 0. for metric in metrics} for k in [3, 5, 10]} for sampling_ratio in ratios}
    
    for sampling_ratio in ratios:
        with open(f'{path}/results_size={comb_size}_sampling={sampling_ratio}.pkl', 'rb') as f:
            results = pickle.load(f)

        # g0 : [NCGD, PREC] : [3, 5, 10]
        num_subgroups = len(results)

        for at_k in [3, 5, 10]:
            NDCG = 0.
            PREC = 0.
            RMSE = 0.
            for subgroup in results:
                NDCG += results[subgroup]['NDCG'][at_k]
                PREC += results[subgroup]['PRECISION'][at_k]
                RMSE += results[subgroup]['RMSE'][at_k]
            # print(f"NDCG@{at_k} = {NDCG/num_subgroups}")
            # print(f"PREC@{at_k} = {PREC/num_subgroups}")
            # print(f"RMSE@{at_k} = {RMSE/num_subgroups}")

            scores[sampling_ratio][at_k]['NDCG'] = NDCG/num_subgroups
            scores[sampling_ratio][at_k]['PREC'] = PREC/num_subgroups
            scores[sampling_ratio][at_k]['RMSE'] = RMSE/num_subgroups

    print(scores)
    
