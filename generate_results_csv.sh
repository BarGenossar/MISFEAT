#!/bin/zsh



missing_probs=(0.2 0.5)
sampling_ratio_list=(1.0)
missing_prob_list=(0.2)
edge_sampling_ratio_list=(0.5)
sampling_method_list=('randwalk')
gnn_model_list=('SAGE')

if [ $# -eq 0 ]; then
    echo "Please Insert Dataset Names as Command Line Arguments"
    exit 1
fi
dataset_name_list=("$@")

# echo "Beginning feature selection task with K=$K"
for dataset_name in "${dataset_name_list[@]}"; do

  for model in "${gnn_model_list[@]}"; do
    for missing_prob in "${missing_prob_list[@]}"; do
      for sampling_method in "${sampling_method_list[@]}"; do
        for sampling_ratio in "${sampling_ratio_list[@]}"; do
          for edge_sampling_ratio in "${edge_sampling_ratio_list[@]}"; do
              echo "Generating results csv for model $model, sampling_ratio $sampling_ratio, missing_prob $missing_prob"
              python generate_results_csv.py --model $model --sampling_ratio $sampling_ratio \
                    --missing_prob $missing_prob --model $model --data_name $dataset_name \
                    --sampling_method $sampling_method --edge_sampling_ratio $edge_sampling_ratio
              echo "*********************************************************************************************************"
          done
        done
      done
    done
  done
done
