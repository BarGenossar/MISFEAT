#!/bin/zsh

formulas=(0 1 2 3 4 5 6);
configs=(1 2 3);
hidden=(32 64 128);

#if [ -z "$1" ]; then
#  K=4
#else
#  K=$1
#fi

# echo "Beginning feature selection task with K=$K"

for j in "${configs[@]}"; do
  for i in "${formulas[@]}"; do
    echo "Generating data for formula $i and config $j"
    python logical_synthetic_data_generator.py --formula $i --config $j
    echo "Generating lattice for formula $i and config $j"
    python lattice_graph_generator_multiprocessing.py --formula $i --config $j
#    for k in "${hidden[@]}"; do
#      echo "Training model for formula $i, config $j and hidden_layer $k. This may take some time."
#      python3 graph_lattice_train_and_evaluate.py --formula $i --config $j --hidden_channels $k --comb_size $K
#    done

  done
done
