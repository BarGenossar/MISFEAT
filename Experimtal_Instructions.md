
# Experimental Instructions
## Section 5.2: MISFEAT vs. Baselines
**1)** Generate the multiple lattice graph for each dataset:

```bash generate_multiple_lattice_graphs.sh <dataset_name1> <dataset_name2> ...```

where dataset_name1, dataset_name2, ... are the names of the datasets.
The script currently supports the following options: {"synthetic", "loan", "startup", "mobile"}.
Note that for the synthetic dataset, the formula and configs indexes are determined inside the script.\
The list of edge_sampling_ratio is also defined in the script.

**2)** Train and evaluate MISFEAT:

```bash run_experiments.sh <dataset_name1> <dataset_name2> ...```

where dataset_name1, dataset_name2, ... are the names of the datasets.
The script currently supports the following options: {"synthetic", "loan", "startup", "mobile"}.
Note that for the synthetic dataset, the formula and configs indexes are determined inside the script.\
The following parameter lists are defined in the script:
- sampling_ratios (run the basic experiments 1.0)
- missing_probs (run the basic experiments with 0.2 0.5)
- edge_sampling_ratio (for now, use only the value 0.5)
- gnn models ('SAGE' 'SAGE_HEAD')
- sampling_method_list ('randwalk' for now, but it is not significant because sampling_ratio is 1.0 for this setting)

**3)** Create the results csv files:

```bash generate_results_csv.sh <dataset_name1> <dataset_name2> ...```

where dataset_name1, dataset_name2, ... are the names of the datasets.
The following parameter lists are defined in the script:
- sampling_ratios (run the basic experiments 1.0)
- missing_probs (run the basic experiments with 0.2 0.5)
- edge_sampling_ratio (for now, use only the value 0.5)
- gnn models ('SAGE' 'SAGE_HEAD')
- sampling_method_list ('randwalk' for now, but it is not significant because sampling_ratio is 1.0 for this setting)
- Note that for the synthetic datasets, the formula and configs indexes are determined inside the config file.

********************************************************************************
********************************************************************************
## Section 5.3: Sampling Ratio Analysis
Run 2 and 3 from above with the following parameter lists:
- sampling_ratios (0.75 0.5 0.25)
- missing_probs (run the basic experiments with 0.2)
- edge_sampling_ratio (for now, use only the value 0.5)
- gnn models ('SAGE' 'SAGE_HEAD') -- Run with the better model among the two (the one that achieved better 
results in 5.2
- sampling_method_list ('randwalk' 'arbitrary')

********************************************************************************
********************************************************************************
## Section 5.4-Forwards: Ablation Study
I'll run it
