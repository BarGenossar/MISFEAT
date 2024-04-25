#!/bin/zsh

models=("GNN" "MLP")
sampling_ratios=(1.0 0.75 0.5 0.25);
missing_probs=(0.2 0.5)


#if [ -z "$1" ]; then
#  K=4
#else
#  K=$1
#fi

# echo "Beginning feature selection task with K=$K"
for model in "${models[@]}"; do
  for missing_prob in "${missing_probs[@]}"; do
    for sampling_ratio in "${sampling_ratios[@]}"; do
        echo "Generating results csv for model $model, sampling_ratio $sampling_ratio, missing_prob $missing_prob"
        python generate_results_csv.py --model $model --sampling_ratio $sampling_ratio --missing_prob $missing_prob
        echo "*********************************************************************************************************"
    done
  done
done
