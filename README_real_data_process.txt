# Dataset Preprocessing Script

## Overview
This script automates the preprocessing of real world datasets for LogicalDatasetGenerator class.

## Usage
Run the script from the command line, specifying the dataset file path, target column, and any columns to be combined into a 'subgroup'.

### Parameters
- `--file_path`: Path to the CSV dataset file. Default: `data/loan.csv`.
- `--target_col`: Name of the target column. Default: `Loan Status`.
- `--subgroup_cols`: Comma-separated list of columns to combine into a 'subgroup' column. Default: `Grade,Sub Grade`.

### Command
```bash
python process_real_data.py --file_path "your/dataset/path.csv" --target_col "TargetColumn" --subgroup_cols "Column1,Column2"
```

Example: 

# python process_real_data.py --file_path "data/startup.csv" --target_col "status" --subgroup_cols "is_CA,is_NY,is_MA,is_TX,is_otherstate" --threshold 0.2 --num_feature 10
# python process_real_data.py --file_path "data/loan.csv" --target_col "Loan Status" --subgroup_cols "Grade" --threshold 0.2 --num_feature 10



## Output
The processed dataset is saved as a pickle file in `RealWorldData/<filename>/dataset.pickle`. A descriptive text file (`description.txt`) outlining the preprocessing steps and feature mappings is also saved in the same directory.
