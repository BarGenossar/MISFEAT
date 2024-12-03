#!/bin/zsh

formulas=(1 2 3)
configs=(3 1)
missing_prob_list=(0.2)
baseline_method_list=('KNN')


# Check if the dataset names are provided as command-line arguments
if [ $# -eq 0 ]; then
    echo "Please Insert Dataset Names as Command Line Arguments"
    exit 1
fi
dataset_name_list=("$@")
# dataset_name_list=("synthetic" "loan" "startup" "mobile")

for dataset_name in "${dataset_name_list[@]}"; do
  for method in "${baseline_method_list[@]}"; do
    for missing_prob in "${missing_prob_list[@]}"; do
      if [ $dataset_name != "synthetic" ]; then
          echo "Running on dataset $dataset_name, , missing_prob $missing_prob, imputation_method $method "

          python baseline.py --missing_prob $missing_prob --data_name $dataset_name \
                --imputation_method $method
          echo "***************************************************************************************"
      else
        for j in "${configs[@]}"; do
          for i in "${formulas[@]}"; do
            echo "Running on  config $j, formula $i, missing_prob $missing_prob, imputation_method $method "
            python baseline.py --formula $i --config $j --missing_prob $missing_prob \
                  --data_name $dataset_name --imputation_method $method
            echo "***************************************************************************************"
          done
        done
      fi
    done
  done
done
