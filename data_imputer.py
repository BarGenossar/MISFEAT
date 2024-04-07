


import os
import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from fancyimpute import IterativeImputer
import json

class DataImputer:
    def __init__(self, data_file, dict_file, imputation_method='mode'):
        self.data_file = data_file
        self.dict_file = dict_file
        self.imputation_method = imputation_method
        self.data = pd.read_pickle(data_file)  # Unchanged, still reading from pickle
        self.subgroup_features = self.load_subgroup_features()

    def load_subgroup_features(self):
        with open(self.dict_file, 'r') as file:
            subgroup_features = json.load(file)
        return subgroup_features

    def replace_with_nan(self):
        for subgroup, features in self.subgroup_features.items():
            self.data.loc[self.data['subgroup'] == subgroup, features] = np.nan

    def impute_data(self):
        if self.imputation_method == 'mode':
            for column in self.data.columns:
                imputer = SimpleImputer(strategy='most_frequent')
                self.data[column] = imputer.fit_transform(self.data[[column]])
        elif self.imputation_method == 'KNN':
            imputer = KNNImputer(n_neighbors=5)
            self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        elif self.imputation_method == 'MICE':
            imputer = IterativeImputer()
            self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        else:
            raise ValueError("Unsupported imputation method")

    def process(self):
        self.replace_with_nan()
        self.impute_data()
        return self.data

def main():
    parser = argparse.ArgumentParser(description='Data Imputation Tool')
    parser.add_argument('data_file', type=str, help='Path to the data file (pickle format)')
    parser.add_argument('dict_file', type=str, help='Path to the JSON dictionary file')
    parser.add_argument('--imputation_method', type=str, default='mode', choices=['mode', 'KNN', 'MICE'],
                        help='Imputation method to use: mode, KNN, or MICE. Default is mode.')

    args = parser.parse_args()

    imputer = DataImputer(args.data_file, args.dict_file, args.imputation_method)
    imputed_data = imputer.process()

    # Constructing the output file name based on the input file name
    base_dir = os.path.dirname(args.data_file)
    base_filename = os.path.splitext(os.path.basename(args.data_file))[0]
    output_file = os.path.join(base_dir, f"{base_filename}_imputed.pickle")

    # Saving the DataFrame in pickle format
    imputed_data.to_pickle(output_file)
    print(f'Imputed data saved to {output_file}')

if __name__ == '__main__':
    main()


# python data_imputer.py RealWorldData/loan/dataset.pkl RealWorldData/loan/missing.json --imputation_method KNN

#missing.json

#{
#    "0": ["f_1", "f_2", "f_3"],
#    "1": ["f_4", "f_5"],
#    "2": ["f_7", "f_8"],
#    "3": ["f_9"],
#    "4": ["f_1", "f_4", "f_6"]
#}
