#!/bin/zsh

formulas=(1 2 3)
configs=(3 1)
sampling_ratio_list=(1.0)
missing_prob_list=(0.2)
edge_sampling_ratio_list=(0.5)
hidden_channels_list=(128)
sampling_method_list=('randwalk')
gnn_model_list=('SAGE')

# Check if GPU number argument is provided
if [ $# -lt 1 ]; then
    echo "Error: GPU number not provided."
    exit 1
fi

# Assign GPU number
export NVIDIA_VISIBLE_DEVICES=$1
export CUDA_VISIBLE_DEVICES=$1
shift  # Remove GPU number from the argument list

# Check if the dataset names are provided as command-line arguments
if [ $# -eq 0 ]; then
    echo "Please insert dataset names as command line arguments after the GPU number."
    exit 1
fi
dataset_name_list=("$@")
# dataset_name_list=("synthetic" "attrition" "mobile" "loan")

for dataset_name in "${dataset_name_list[@]}"; do
  for model in "${gnn_model_list[@]}"; do
    for missing_prob in "${missing_prob_list[@]}"; do
      for sampling_ratio in "${sampling_ratio_list[@]}"; do
        for edge_sampling_ratio in "${edge_sampling_ratio_list[@]}"; do
          for sampling_method in "${sampling_method_list[@]}"; do
            for hidden_channels in "${hidden_channels_list[@]}"; do
              if [ $dataset_name != "synthetic" ]; then
                  echo "Running on dataset $dataset_name, sampling_ratio $sampling_ratio, missing_prob $missing_prob,"\
                  "sampling_method $sampling_method edge_sampling_ratio $edge_sampling_ratio,"\
                  "hidden_channels $hidden_channels model $model"

                  python train_and_evaluate.py --sampling_ratio $sampling_ratio --missing_prob $missing_prob \
                        --edge_sampling_ratio $edge_sampling_ratio --data_name $dataset_name  \
                        --sampling_method $sampling_method --hidden_channels $hidden_channels --model $model
                  echo "***************************************************************************************"
              else
                for j in "${configs[@]}"; do
                  for i in "${formulas[@]}"; do
                    echo "Running on  config $j, formula $i, sampling_ratio $sampling_ratio, missing_prob"\
                    "$missing_prob, sampling_method $sampling_method edge_sampling_ratio $edge_sampling_ratio,"\
                    "hidden_channels $hidden_channels model $model"

                    python train_and_evaluate.py --formula $i --config $j --sampling_ratio $sampling_ratio \
                          --missing_prob $missing_prob --data_name $dataset_name --sampling_method $sampling_method \
                          --edge_sampling_ratio $edge_sampling_ratio --hidden_channels $hidden_channels --model $model
                    echo "***************************************************************************************"
                  done
                done
              fi
            done
          done
        done
      done
    done
  done
done
