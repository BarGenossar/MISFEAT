import argparse
import pickle
import matplotlib.pyplot as plt




def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='loan', help="name of dataset")
    parser.add_argument('--comb_size', type=int, default=3, help="combination size")
    # parser.add_argument('', type=, default=, help="")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parser()
    path = f'./RealWorldData/{args.data_name}'


    metrics = ['NDCG', 'PREC', 'RMSE']
    ratios = [0.25, 0.5, 0.75, 1.0]
    scores = {sampling_ratio: {k: {metric: 0. for metric in metrics} for k in [3, 5, 10]} for sampling_ratio in ratios}
    
    for sampling_ratio in ratios:
        print(f'ratio: {sampling_ratio}, comb_size: {args.comb_size}')
        with open(f'{path}results_size={args.comb_size}_sampling={sampling_ratio}.pkl', 'rb') as f:
            results = pickle.load(f)

        # g0 : [NCGD, PREC] : [3, 5, 10]
        num_subgroups = len(results)

        for at_k in [3, 5, 10]:
            NDCG = 0.
            PREC = 0.
            RMSE = 0.
            # for subgroup in results:
            for subgroup in ['g0',]:
                NDCG = results[subgroup]['NDCG'][at_k]
                PREC = results[subgroup]['PREC'][at_k]
                RMSE = results[subgroup]['RMSE'][at_k]
                
                # print(f"NDCG@{at_k} for subgroup {subgroup} = {round(NDCG, 2)}")
                print(f"PREC@{at_k} for subgroup {subgroup} = {round(PREC, 2)}")
                # print(f"PREC@{at_k} for subgroup {subgroup} = {PREC/num_subgroups}")
                # print(f"RMSE@{at_k} for subgroup {subgroup} = {RMSE/num_subgroups}")

            # scores[sampling_ratio][at_k]['NDCG'] = NDCG/num_subgroups
            # scores[sampling_ratio][at_k]['PREC'] = PREC/num_subgroups
            # scores[sampling_ratio][at_k]['RMSE'] = RMSE/num_subgroups
        
        print()

    # at_k = 3
    # prec = []
    # for r in ratios:
    #     prec.append(scores[r][at_k]['PREC'])

    # plt.plot(ratios, prec)
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.show()

    
    
