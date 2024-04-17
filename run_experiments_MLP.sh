#!/bin/zsh

formulas=(0 1 2 3 4);
configs=(1 3);
sampling_ratios=(1.0 0.75 0.5 0.25);
missing_probs=(0.2 0.5)


#if [ -z "$1" ]; then
#  K=4
#else
#  K=$1
#fi

# echo "Beginning feature selection task with K=$K"
for missing_prob in "${missing_probs[@]}"; do
  for sampling_ratio in "${sampling_ratios[@]}"; do
    for j in "${configs[@]}"; do
      for i in "${formulas[@]}"; do
        echo "Running on  config $j, formula $i, sampling_ratio $sampling_ratio, missing_prob $missing_prob"
        python MLP_baseline.py --formula $i --config $j --sampling_ratio $sampling_ratio --missing_prob $missing_prob
        echo "*********************************************************************************************************"
        echo "*********************************************************************************************************"
      done
    done
  done
done
