#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class RealWorldDatasetPreprocessor:
    def __init__(self, file_path, target_col, subgroup_cols, threshold_select_category=0.01):
        self.file_path = file_path
        self.target_col = target_col
        self.subgroup_cols = subgroup_cols.split(',')  # Assume subgroup_cols is passed as a comma-separated string
        self.threshold_select_category = threshold_select_category
        self.feature_mapping = {}  # To store the mapping of original to new column names

    def load_data(self):
        """Load data from a CSV file."""
        return pd.read_csv(self.file_path)
    
    def sample_data(self, df, num_sample):
        """Sample from the df."""
        sampled_df = df.sample(n=num_sample)
        return sampled_df


    def select_random_features(df, k):
        features_not_to_discard = {'y','subgroup'}
        columns_to_consider = [col for col in df.columns if col not in features_not_to_discard]
        k = min(k, len(columns_to_consider))
        selected_features = np.random.choice(columns_to_consider, size=k, replace=False)
        final_columns = list(features_not_to_discard) + list(selected_features)
        return df[final_columns]

    def encode_categorical_features(self, df):
        """Encode categorical features based on a uniqueness threshold."""
        le = LabelEncoder()
        numberOfrows = df.shape[0]
        for col in df.columns:
            if len(set(df[col])) < numberOfrows * self.threshold_select_category:
                df[col] = le.fit_transform(df[col].astype(str))
        return df

    def combine_subgroup_columns(self, df):
        """Combine specified columns into a single 'subgroup' column."""
        le = LabelEncoder()
        if self.subgroup_cols:  # Only proceed if subgroup_cols is not empty
            df['subgroup'] = df[self.subgroup_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
            df['subgroup'] = le.fit_transform(df['subgroup'].astype(str))
            df = df.drop(columns=self.subgroup_cols)
        return df

    def rename_columns(self, df):
        """Rename columns, preserving the target column and the new 'subgroup' column."""
        column_mapping = {}
        self.feature_mapping = {}  # Reset feature mapping for each process call
        for i, col in enumerate(df.columns):
            if col == self.target_col:
                column_mapping[col] = 'y'
                self.feature_mapping['y'] = self.target_col
            elif col == 'subgroup':
                self.feature_mapping['subgroup'] = "Combined: " + ", ".join(self.subgroup_cols)
            else:
                new_col_name = f'f_{i}'
                column_mapping[col] = new_col_name
                self.feature_mapping[new_col_name] = col
        df = df.rename(columns=column_mapping)
        return df

    def process(self):
        """Process the dataset by encoding, combining, and renaming as specified."""
        df = self.load_data()
        df = self.encode_categorical_features(df)
        df = self.combine_subgroup_columns(df)
        df = self.rename_columns(df)
        df = self.sample_data(df, 10000)
        df = select_random_features(df,10)
        return df
    
    def save(self, df):
        """
        Saves the given DataFrame to a pickle file inside a folder structure:
        'RealWorldData/<filename>/dataset.pkl', and writes a description.txt file with details
        about the processed dataset, including feature mappings, the number of features
        excluding the target column and 'subgroup', and the count of unique categories in the 'subgroup' column.
        """
        base_name = os.path.basename(self.file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        dir_path = os.path.join('RealWorldData', name_without_ext)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        pickle_file_path = os.path.join(dir_path, "dataset.pkl")
        df.to_pickle(pickle_file_path)

        # Count the number of unique categories in 'subgroup' if it exists
        subgroup_count = df['subgroup'].nunique() if 'subgroup' in df.columns else 0

        # Calculate the number of features excluding 'y' and 'subgroup'
        feature_columns = [col for col in df.columns if col not in ['y', 'subgroup']]
        feature_num = len(feature_columns)

        description_content = "Feature mappings and dataset details.\n"
        description_content += f"feature_num': {feature_num}\n"
        description_content += f"Number of unique categories in 'subgroup': {subgroup_count}\n"
        for new_col, orig_col in self.feature_mapping.items():
            description_content += f"{new_col}: {orig_col}\n"
        description_file_path = os.path.join(dir_path, "description.txt")
        with open(description_file_path, 'w') as file:
            file.write(description_content)
        print(f"DataFrame saved to {pickle_file_path}")
        print(f"Description saved to {description_file_path}")



def main():
    parser = argparse.ArgumentParser(description='Process a dataset with specified parameters.')
    parser.add_argument('--file_path', type=str, help='Path to the dataset file.')
    parser.add_argument('--target_col', type=str, help='Name of the target column.')
    parser.add_argument('--subgroup_cols', type=str, help='Comma-separated list of subgroup column names.')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold to select categorical features.')
    args = parser.parse_args()
    processor = RealWorldDatasetPreprocessor(file_path=args.file_path, target_col=args.target_col, subgroup_cols=args.subgroup_cols, threshold_select_category=args.threshold)
    df_processed = processor.process()
    processor.save(df_processed)

if __name__ == '__main__':
    main()
