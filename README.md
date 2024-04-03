
# Experimental Instructions for a Single Dataset
## Raw Data Preparation
1. The raw data should be saved as a pandas dataframe object in a pkl file.
2. The dataframe should have the following columns:

    f_0, f_1,...,f_{n-1}, y, subgroup

    where f_0, f_1,...,f_{n-1} are the feature columns, y is the target column, and subgroup is the subgroup column 
(integer values starting from 0).

The features and the subgroups in the real-world datasets have actual meaning so please save mapping dictionary for both, 
For example: {'f_0': 'BMI', 'f_1': 'cholesterol',...} and {0: 'over 40 male', 1: 'over 40 female',...}.

## From Raw Data to Feature Lattice Graph
To generate the lattice graph, run the following command:

```python lattice_graph_generator.py --data_path <path_to_data.pkl>```


## Running the GNN
For each dataset we have a single representative feature lattice graph. We apply the various modifications
with the injection of the systematic missing data. The difficulty level of the dataset is determined by the
the missingness pattern. Pls explore your real-world datasets and come with suggestions for each of the following:
soft, medium, and hard.

When you run the training and evaluation script over the real-world datasets, you need to set the 
hyperparameter manual_md as True, an then you will be asked to type the missing features  for each subgroup.
Let's assume that we have decided on the 3 configurations for the missingness pattern for the dataset.
Then you will need to run the following command:

```python train_and_evaluate.py --data_path <path_to_data.pkl> --manual_md True```

The results dictinary, presenting the average performance over several seeds (the default value is 3)
will be saved in the the same folder as the dataset.
Some other hyperparameters can be set in the train_and_evaluate.py script, but there is
no need to change them for the initial experiments.

## Summary- What Do We Run? 
For each dataset (feature lattice graph):

- Generate 3 views of the feature lattice graph with different missingness patterns:
    - soft
    - medium
    - hard

For each one of the three views:
- Run the training and evaluation script with the manual_md set to True (for the real-world 
dataset) with each of the following sampling rate:
    - 1.0 (no sampling)
    - 0.5 
    - 0.25

- Run Each one of the baseline over (but in a different script)