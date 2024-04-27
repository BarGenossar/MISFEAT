#!/usr/bin/env python
# coding: utf-8
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# class RealWorldDatasetPreprocessor:
#     def __init__(self, file_path, target_col, subgroup_cols, threshold_select_category=0.01, num_feature = 1000000):
#         self.file_path = file_path
#         self.target_col = target_col
#         self.num_feature = num_feature
#         self.subgroup_cols = subgroup_cols.split(',')  # Assume subgroup_cols is passed as a comma-separated string
#         self.threshold_select_category = threshold_select_category
#         self.feature_mapping = {}  # To store the mapping of original to new column names

#     def load_data(self):
#         """Load data from a CSV file."""
#         df = pd.read_csv(self.file_path)
#         return df

#     def encode_categorical_features(self, df):
#         """Encode categorical features based on a uniqueness threshold and remove non-categorical features."""
#         le = LabelEncoder()
#         categorical_features = []
#         #print("th = ",df.shape[0] * self.threshold_select_category)
#         for col in df.columns:
#             if len(set(df[col])) <= df.shape[0] * self.threshold_select_category:
#                 categorical_features.append(col)
#         df = df[categorical_features]
#         for col in categorical_features:
#             df.loc[:, col] = le.fit_transform(df[col].astype(str))
#         #print(df.head())
#         return df

#     def combine_subgroup_columns(self, df):
#         """Combine specified columns into a single 'subgroup' column."""
#         le = LabelEncoder()
#         if self.subgroup_cols:  # Only proceed if subgroup_cols is not empty
#             df['subgroup'] = df[self.subgroup_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
#             df['subgroup'] = le.fit_transform(df['subgroup'].astype(str))
#             df = df.drop(columns=self.subgroup_cols)
#         return df

#     def rename_columns(self, df):
#         """Rename columns, preserving the target column and the new 'subgroup' column."""
#         column_mapping = {}
#         self.feature_mapping = {}  # Reset feature mapping for each process call
#         #print(df.columns)
#         count = 0
#         for col in df.columns:

#             if col == self.target_col:
#                 column_mapping[col] = 'y'
#                 self.feature_mapping['y'] = self.target_col
#             elif col == 'subgroup':
#                 self.feature_mapping['subgroup'] = "Combined: " + ", ".join(self.subgroup_cols)
#             else:
#                 #print(count)
#                 new_col_name = f'f_{count}'
#                 column_mapping[col] = new_col_name
#                 self.feature_mapping[new_col_name] = col
#                 count = count + 1
#         df = df.rename(columns=column_mapping)
#         return df

#     def rearrange_columns(self,df):
#         other_cols = [col for col in df.columns if col not in ['y', 'subgroup']]
#         new_order = other_cols + ['y', 'subgroup']
#         df = df[new_order]
#         return df
    
#     def sample_data(self, df, num_sample):
#         """Sample from the df."""
#         sampled_df = df.sample(n=num_sample)
#         return sampled_df
    
#     def select_random_features(self, df):
#         features_not_to_discard = [self.target_col] + self.subgroup_cols
#         columns_to_consider = [col for col in df.columns if col not in features_not_to_discard]
#         k = min(self.num_feature, len(columns_to_consider))
#         selected_features = np.random.choice(columns_to_consider, size=k, replace=False)
#         final_columns = list(features_not_to_discard) + list(selected_features)
#         return df[final_columns]
    
#     def process(self):
#         """Process the dataset by encoding, combining, and renaming as specified."""
#         df = self.load_data()
#         df = self.encode_categorical_features(df)
#         df = self.select_random_features(df)
#         df = self.combine_subgroup_columns(df)
#         df = self.rename_columns(df)
#         df = self.rearrange_columns(df)
#         return df
    
#     def save(self, df):
#         """
#         Saves the given DataFrame to a pickle file inside a folder structure:
#         'RealWorldData/<filename>/dataset.pkl', and writes a description.txt file with details
#         about the processed dataset, including feature mappings, the number of features
#         excluding the target column and 'subgroup', and the count of unique categories in the 'subgroup' column.
#         """
#         base_name = os.path.basename(self.file_path)
#         name_without_ext = os.path.splitext(base_name)[0]
#         dir_path = os.path.join('RealWorldData', name_without_ext)
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#         pickle_file_path = os.path.join(dir_path, "dataset.pkl")
#         df.to_pickle(pickle_file_path)

#         # Count the number of unique categories in 'subgroup' if it exists
#         subgroup_count = df['subgroup'].nunique() if 'subgroup' in df.columns else 0

#         # Calculate the number of features excluding 'y' and 'subgroup'
#         feature_columns = [col for col in df.columns if col not in ['y', 'subgroup']]
#         feature_num = len(feature_columns)

#         description_content = "Feature mappings and dataset details.\n"
#         description_content += f"feature_num': {feature_num}\n"
#         description_content += f"Number of unique categories in 'subgroup': {subgroup_count}\n"
#         for new_col, orig_col in self.feature_mapping.items():
#             description_content += f"{new_col}: {orig_col}\n"
#         description_file_path = os.path.join(dir_path, "description.txt")
#         with open(description_file_path, 'w') as file:
#             file.write(description_content)
#         print(f"DataFrame saved to {pickle_file_path}")
#         print(f"Description saved to {description_file_path}")




def preprocess_mobile(data, num_features=1000, split_feature='dual_sim'):
    data['battery_power'] = data['battery_power'].apply(lambda x : 0 if x <= 1000
                                                        else (1 if 1000 < x <= 1500                                                     
                                                        else  2))
    data['clock_speed'] = data['clock_speed'].apply(lambda x : 0 if x <= 1 
                                                    else (1 if 1 < x <= 2                                                        
                                                    else  2))
    data['int_memory'] = data['int_memory'].apply(lambda x : 0 if x <= 10
                                                else (1 if 10 < x <= 20                                                      
                                                else (2 if 20 < x <= 30
                                                else (3 if 30 < x <= 40
                                                else (4 if 40 < x <= 50
                                                else (5 if 50 < x <= 60
                                                else  6))))))
    data['m_dep'] = data['m_dep'].apply(lambda x : 0 if x <= 0.5 
                                        else 1)
    data['mobile_wt'] = data['mobile_wt'].apply(lambda x : 0 if x <= 120
                                                else (1 if 120 < x <= 160                                                      
                                                else  2))
    data['n_cores'] = data['n_cores'].apply(lambda x : 0 if x <= 2
                                                else (1 if 2 < x <= 4                                                      
                                                else (2 if 4 < x <= 6                                                      
                                                else  3)))
    data['px_height'] = data['px_height'].apply(lambda x : 0 if x <= 720
                                                else (1 if 720 < x <= 1080                                                      
                                                else  2))
    data['px_width'] = data['px_width'].apply(lambda x : 0 if x <= 1080
                                              else (1 if 1080 < x <= 1440                                                      
                                              else  2))
    data['ram'] = data['ram'].apply(lambda x : 0 if x <= 1024
                                    else (1 if 1024 < x <= 2048                                                      
                                    else (2 if 2048 < x <= 3072                                                      
                                    else  3)))
    data['sc_h'] = data['sc_h'].apply(lambda x : 0 if x <= 12
                                    else 1)
    data['sc_w'] = data['sc_w'].apply(lambda x : 0 if x <= 7.2
                                    else 1)
    data['fc'] = data['fc'].apply(lambda x : 0 if x <= 3.8
                                  else (1 if 3.8 < x <= 7.6                                                      
                                  else  2))
    data['pc'] = data['pc'].apply(lambda x : 0 if x <= 10
                                  else 1)
    data['talk_time'] = data['talk_time'].apply(lambda x : 0 if x <= 11
                                                else 1)
    subgroup_dict = {subgroup: i for i, subgroup in enumerate(sorted(set(data['dual_sim'])))}
    data['dual_sim'] = data['dual_sim'].replace(subgroup_dict)

    all_columns = list(data.columns)
    all_columns.remove('dual_sim')
    all_columns.remove('price_range')
    feature_list = random.sample(all_columns, min(len(all_columns), num_features))

    data = data[feature_list + ['dual_sim'] + ['price_range']]

    feature_dict = {'price_range': 'y',
                    'dual_sim': 'subgroup'}

    for i, feat in enumerate(feature_list):
        # count_missing = data[feat].value_counts().get('?')
        # print(feat, ', num distinct =', len(set(data[feat])), ', missing tuples =', count_missing)
        feature_dict[feat] = f"f_{i}"

    data = data.rename(columns=feature_dict)
    data.reset_index(drop=True, inplace=True)

    data = data.drop(columns=['f_4', 'f_5', 'f_13', 'f_14'], axis=1)
    cols = [col for col in data.columns if 'f_' in col]
    new_names = {col: f'f_{idx}' for idx, col in enumerate(cols)}
    data = data.rename(columns = new_names)
    data.to_pickle('./RealWorldData/mobile/dataset.pkl')
    return data



def preprocess_diabetes(data, num_features=1000, split_feature=''):
    drop_features = ['weight', 'encounter_id', 'payer_code', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'examide', 'citoglipton', 'glimepiride-pioglitazone', 'patient_nbr']
    data.dropna(inplace=True)
    data.drop_duplicates(['patient_nbr'], keep='first', inplace=True)
    data.drop(drop_features, axis=1, inplace=True)
    data['time_in_hospital'] = data['time_in_hospital'].apply(lambda x : 1 if 0 <= int(x) <= 7 
                                                                           else 2)
    data['admission_type_id'] = data['admission_type_id'].apply(lambda x : 1 if int(x) in [1, 7]
                                                                else ( 3 if int(x) in [5, 6, 8]
                                                                else 2))

    # for col in ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]:
    #     data[col] = data[col].apply(lambda x : 10 if x == 'up' 
    #                                             else ( -10 if x == 'down'                                                          
    #                                             else ( 0 if x == 'steady'
    #                                             else  -20)))

    # data['change'] = data['change'].apply(lambda x : 1 if x == 'Ch'
    #                                                 else -1)

    # data['diabetesMed'] = data['diabetesMed'].apply(lambda x : -1 if x == 'No'
    #                                                 else 1)
    data['number_diagnoses'] = data['number_diagnoses'].apply(lambda x : 1 if int(x) in [1, 2, 3, 4] 
                                                            else (2 if int(x) in [5, 6, 7, 8]                                                        
                                                            else (3 if int(x) in [9, 10, 11, 12]
                                                            else  4)))
    data['num_procedures'] = data['num_procedures'].apply(lambda x : 1 if int(x) == 0 
                                                            else (2 if int(x) in [1, 2, 3]                                                        
                                                            else 3))
    data['num_lab_procedures'] = data['num_lab_procedures'].apply(lambda x : 1 if int(x) <= 32
                                                            else (2 if 32 < int(x) <= 64                                                      
                                                            else (3 if 64 < int(x) <= 96
                                                            else  4)))
    data['num_medications'] = data['num_medications'].apply(lambda x : 1 if int(x) <= 27
                                                            else (2 if 27 < int(x) <= 54                                                      
                                                            else  3))
    data['number_outpatient'] = data['number_outpatient'].apply(lambda x : 1 if int(x) <= 10
                                                            else (2 if 10 < int(x) <= 20                                                      
                                                            else  3))
    data['number_emergency'] = data['number_emergency'].apply(lambda x : 1 if int(x) <= 10
                                                            else 2)
    data['number_inpatient'] = data['number_inpatient'].apply(lambda x : 1 if int(x) <= 6
                                                            else 2)
    ### reference paper for binning diagnoses ICD-9 code: https://pubmed.ncbi.nlm.nih.gov/24804245/
    for diag in ['diag_1', 'diag_2', 'diag_3']:
        data[diag] = data[diag].apply(lambda x : '?' if '?' in x
                                                else ('non-disease condition' if 'V' in x
                                                else ('external injury/poisoning' if 'E' in x  
                                                else ('diabetes' if re.compile(r'250.*').search(x)
                                                else ('circulatory' if 390 <= float(x) <= 459 or float(x) == 785 
                                                else ('respiratory' if 460 <= float(x) <= 519 or float(x) == 786                                                      
                                                else ('digestive' if 520 <= float(x) <= 579 or float(x) == 787
                                                else ('injury' if 800 <= float(x) <= 999
                                                else ('musculoskeletal' if 710 <= float(x) <= 739
                                                else ('genitourinary' if 580 <= float(x) <= 629 or float(x) == 788
                                                else ('neoplasms' if 140 <= float(x) <= 239
                                                else  'other')))))))))))
    data = data[data['race'] != '?']
    data = data[data['diag_1'] != '?']
    data = data[data['diag_2'] != '?']
    data = data[data['diag_3'] != '?']

    subgroup_dict = {subgroup: f"s{i}" for i, subgroup in enumerate(sorted(set(data[split_feature])))}
    data[split_feature] = data[split_feature].replace(subgroup_dict)

    all_columns = list(data.columns)
    all_columns.remove(split_feature)
    all_columns.remove('readmitted')
    feature_list = random.sample(all_columns, min(len(all_columns), num_features))
    data = data[feature_list + [split_feature] + ['readmitted']]
    feature_dict = {}

    for i, feat in enumerate(feature_list):
        # count_missing = data[feat].value_counts().get('?')
        # print(feat, ', num distinct =', len(set(data[feat])), ', missing tuples =', count_missing)
        feature_dict[feat] = f"f{i}"
    
    feature_dict['readmitted'] = 'label'

    data = data.rename(columns=feature_dict)
    data.reset_index(drop=True, inplace=True)



def preprocess_loan(data, num_features=1000, split_feature='Grade'):
    drop_features = ['ID', 'Term', 'Batch Enrolled', 'Sub Grade', 'Payment Plan', 'Inquires - six months',
                     'Public Record', 'Total Received Late Fee', 'Recoveries', 'Collection Recovery Fee',
                     'Collection 12 months Medical', 'Application Type', 'Accounts Delinquent',
                     'Total Collection Amount']
    data.dropna(inplace=True)
    data.drop(drop_features, axis=1, inplace=True)
    print(data.columns)

    data['Loan Amount'] = data['Loan Amount'].apply(lambda x : 0 if int(x) < 8000 
                                                            else (1 if 8000 <= int(x) < 16000                                                        
                                                            else (2 if 16000 <= int(x) < 24000
                                                            else  3)))
    data['Funded Amount'] = data['Funded Amount'].apply(lambda x : 0 if int(x) < 8000 
                                                            else (1 if 8000 <= int(x) < 16000                                                        
                                                            else (2 if 16000 <= int(x) < 24000
                                                            else  3)))
    data['Funded Amount Investor'] = data['Funded Amount Investor'].apply(lambda x : 0 if int(x) < 8000 
                                                            else (1 if 8000 <= int(x) < 16000                                                        
                                                            else (2 if 16000 <= int(x) < 24000
                                                            else  3)))
    data['Interest Rate'] = data['Interest Rate'].apply(lambda x : 0 if int(x) < 10
                                                            else ( 1 if 10 <= int(x) < 18
                                                            else   2))
    data['Grade'] = data['Grade'].apply(lambda x : 0 if x == 'C' 
                                             else (1 if x == 'B'                                                        
                                             else  2))
    data['Employment Duration'] = data['Employment Duration'].apply(lambda x : 0 if x == 'MORTGAGE' 
                                             else (1 if x == 'RENT'                                                        
                                             else  2))
    data['Home Ownership'] = data['Home Ownership'].apply(lambda x : 0 if int(x) < 50000
                                                            else ( 1 if 50000 <= int(x) < 100000
                                                            else   2))
    data['Verification Status'] = data['Verification Status'].apply(lambda x : 0 if x == 'Source Verified' 
                                             else (1 if x == 'Verified'                                                        
                                             else  2))
    data['Loan Title'] = data['Loan Title'].apply(lambda x : 0 if x == 'Credit card refinancing' 
                                             else (1 if x == 'Debt consolidation'                                                        
                                             else  2))
    data['Debit to Income'] = data['Debit to Income'].apply(lambda x : 0 if int(x) < 15
                                                            else ( 1 if 15 <= int(x) < 30
                                                            else   2))
    data['Open Account'] = data['Open Account'].apply(lambda x : 0 if int(x) < 15
                                                            else 1)
    data['Delinquency - two years'] = data['Delinquency - two years'].apply(lambda x : 0 if int(x) < 2
                                                            else 1)
    data['Revolving Balance'] = data['Revolving Balance'].apply(lambda x : 0 if int(x) < 5000
                                                            else ( 1 if 5000 <= int(x) < 10000
                                                            else   2))
    data['Revolving Utilities'] = data['Revolving Utilities'].apply(lambda x : 0 if int(x) < 25 
                                                            else (1 if 25 <= int(x) < 50
                                                            else (2 if 50 <= int(x) < 75
                                                            else  3)))
    data['Total Accounts'] = data['Total Accounts'].apply(lambda x : 0 if int(x) < 8
                                                            else ( 1 if 8 <= int(x) < 15
                                                            else   2))
    data['Initial List Status'] = data['Initial List Status'].apply(lambda x : 0 if x == 'w'
                                                            else 1)
    data['Total Received Interest'] = data['Total Received Interest'].apply(lambda x : 0 if int(x) < 1500
                                                            else ( 1 if 1500 <= int(x) < 3000
                                                            else   2))
    data['Last week Pay'] = data['Last week Pay'].apply(lambda x : 0 if int(x) < 40 
                                                            else (1 if 40 <= int(x) < 80
                                                            else (2 if 80 <= int(x) < 120
                                                            else  3)))
    data['Total Current Balance'] = data['Total Current Balance'].apply(lambda x : 0 if int(x) < 100000
                                                            else ( 1 if 100000 <= int(x) < 250000
                                                            else   2))
    data['Total Revolving Credit Limit'] = data['Total Revolving Credit Limit'].apply(lambda x : 0 if int(x) < 15000
                                                            else ( 1 if 15000 <= int(x) < 30000
                                                            else ( 2 if 30000 <= int(x) < 45000
                                                            else   3)))
    
    subgroup_dict = {subgroup: i for i, subgroup in enumerate(sorted(set(data[split_feature])))}
    data[split_feature] = data[split_feature].replace(subgroup_dict)

    all_columns = list(data.columns)
    all_columns.remove(split_feature)
    all_columns.remove('Loan Status')
    feature_list = random.sample(all_columns, min(len(all_columns), num_features))

    data = data[feature_list + [split_feature] + ['Loan Status']]
    feature_dict = {}

    for i, feat in enumerate(feature_list):
        # count_missing = data[feat].value_counts().get('?')
        # print(feat, ', num distinct =', len(set(data[feat])), ', missing tuples =', count_missing)
        feature_dict[feat] = f"f_{i}"
    
    feature_dict['Loan Status'] = 'y'
    feature_dict[split_feature] = 'subgroup'

    data = data.rename(columns=feature_dict)

    other_cols = [col for col in data.columns if col not in ['y', 'subgroup']]
    new_order = other_cols + ['y', 'subgroup']
    data = data[new_order]

    data.reset_index(drop=True, inplace=True)
    data.to_pickle('./RealWorldData/loan/dataset.pkl')
    # data.to_csv('./RealWorldData/loan/loan_processed.csv', index=False)



def preprocess_startup(data, num_features=1000, split_feature='state_code'):
    drop_features = [
        'zip_code',
        'id',
        'Unnamed: 6',
        'name',
        'funding_total_usd',
        'is_consulting',
        'object_id',
        'state_code.1',
        'founded_at',
        'closed_at',
        'is_CA',
        'is_TX',
        'is_NY',
        'is_MA',
        'is_otherstate',
        'is_software',
        'is_web',
        'is_mobile',
        'is_enterprise',
        'is_advertising',
        'is_gamesvideo',
        'is_ecommerce',
        'is_biotech',
        'is_consulting',
        'is_othercategory'
    ]
    # data.dropna(inplace=True)
    data.drop(drop_features, axis=1, inplace=True)
    data['Unnamed: 0'] = data['Unnamed: 0'].apply(lambda x : 0 if int(x) < 400 
                                                       else (1 if 400 <= int(x) < 800                                                        
                                                       else  2))
    data['latitude'] = data['latitude'].apply(lambda x : 0 if int(x) < 39 
                                                    else 1)
    data['longitude'] = data['longitude'].apply(lambda x : 0 if int(x) < -94 
                                                      else 1)
    data['city'] = data['city'].apply(lambda x : 0 if x == "San Francisco"
                                           else (1 if x == "New York"
                                           else 2))
    # data['founded_at'] = data['founded_at'].apply(lambda x : 0 if datetime.strptime(x, "%m/%d/%Y") < datetime.strptime("06/07/2007", "%m/%d/%Y")
    #                                                     else 1)
    # data['closed_at'] = data['closed_at'].apply(lambda x : 0 if datetime.strptime(x, "%m/%d/%Y") < datetime.strptime("07/18/2012", "%m/%d/%Y")
    #                                                   else 1)
    data['first_funding_at'] = data['first_funding_at'].apply(lambda x : 0 if datetime.strptime(x, "%m/%d/%Y") < datetime.strptime("05/01/2008", "%m/%d/%Y")
                                                                    else 1)
    data['last_funding_at'] = data['last_funding_at'].apply(lambda x : 0 if datetime.strptime(x, "%m/%d/%Y") < datetime.strptime("01/08/2010", "%m/%d/%Y")
                                                                    else 1)
    data['age_first_funding_year'] = data['age_first_funding_year'].apply(lambda x : 0 if float(x) < 1
                                                       else (1 if 1 <= float(x) < 3                                                        
                                                       else  2))
    data['age_last_funding_year'] = data['age_last_funding_year'].apply(lambda x : 0 if float(x) < 1
                                                                             else (1 if 1 <= float(x) < 3                                                        
                                                                             else  2))
    data['age_first_milestone_year'] = data['age_first_milestone_year'].apply(lambda x : 0 if float(x) < 1.5
                                                                                   else (1 if 1.5 <= float(x) < 3                                                        
                                                                                   else  2))
    data['age_last_milestone_year'] = data['age_last_milestone_year'].apply(lambda x : 0 if float(x) < 2.5
                                                                                 else (1 if 2.5 <= float(x) < 5
                                                                                 else  2))
    data['relationships'] = data['relationships'].apply(lambda x : 0 if int(x) <= 5 
                                                       else (1 if 5 < int(x) <= 10                                                        
                                                       else  2))
    data['funding_rounds'] = data['funding_rounds'].apply(lambda x : 0 if int(x) <= 2 
                                                       else (1 if 2 < int(x) <= 4                                                    
                                                       else  2))
    data['milestones'] = data['milestones'].apply(lambda x : 0 if int(x) <= 2 
                                                       else (1 if 2 < int(x) <= 4
                                                       else  2))
    data['category_code'] = data['category_code'].apply(lambda x : 0 if x == "software"
                                                             else (1 if x == "web"
                                                             else  2))
    data['avg_participants'] = data['avg_participants'].apply(lambda x : 0 if float(x) < 4.0
                                                                    else 1)
    data['state_code'] = data['state_code'].apply(lambda x : 0 if x == "CA"
                                                       else (1 if x == "NY"
                                                       else (2 if x == "TX"
                                                       else (3 if x == "MA"
                                                       else  4))))
    data['status'] = data['status'].apply(lambda x : 0 if x.lower() == 'closed'
                                                else 1)


    all_columns = list(data.columns)
    all_columns.remove(split_feature)
    all_columns.remove('status')
    feature_list = random.sample(all_columns, min(len(all_columns), num_features))

    data = data[feature_list + [split_feature] + ['status']]
    feature_dict = {}

    for i, feat in enumerate(feature_list):
        # count_missing = data[feat].value_counts().get('?')
        # print(feat, ', num distinct =', len(set(data[feat])), ', missing tuples =', count_missing)
        feature_dict[feat] = f"f_{i}"
    
    feature_dict['status'] = 'y'
    feature_dict[split_feature] = 'subgroup'

    data = data.rename(columns=feature_dict)

    other_cols = [col for col in data.columns if col not in ['y', 'subgroup']]
    new_order = other_cols + ['y', 'subgroup']
    data = data[new_order]

    data.reset_index(drop=True, inplace=True)
    data.to_pickle('./RealWorldData/startup/dataset.pkl')


def main():
    parser = argparse.ArgumentParser(description='Process a dataset with specified parameters.')
    parser.add_argument('--file_path', type=str, help='Path to the dataset file.')
    parser.add_argument('--target_col', type=str, help='Name of the target column.')
    parser.add_argument('--subgroup_cols', type=str, help='Comma-separated list of subgroup column names.')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold to select categorical features.')
    parser.add_argument('--num_feature', type=int, default=100, help='Number of features to select')
    args = parser.parse_args()
    processor = RealWorldDatasetPreprocessor(file_path=args.file_path, target_col=args.target_col, subgroup_cols=args.subgroup_cols, threshold_select_category=args.threshold, num_feature= args.num_feature)
    df_processed = processor.process()
    # print(df_processed)
    drop_columns = []
    for col in df_processed.columns:
        if len(set(df_processed[col])) > 8 or len(set(df_processed[col])) < 2:
            drop_columns.append(col)

    # for col in df_processed.columns:
    #     print(col, len(set(df_processed[col])))
    # exit()
    # drop_columns.extend(['f_0', 'f_4', 'f_13', 'f_14'])
    # df_processed.drop(columns=drop_columns, axis=1, inplace=True)

    column_mapping = {col: f'f_{idx}' for idx, col in enumerate(list(df_processed.columns)) if 'f_' in col}
    df_processed = df_processed.rename(columns=column_mapping)
    print(df_processed)
    # processor.save(df_processed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a dataset with specified parameters.')
    parser.add_argument('--data_name', type=str, default='mobile', help='Path to the dataset file.')
    parser.add_argument('--num_feature', type=int, default=18, help='Number of features to select')
    args = parser.parse_args()

    data = pd.read_csv(f'./data/{args.data_name}.csv')
    
    random.seed(42)

    if args.data_name == 'startup':
        preprocess_startup(data, args.num_feature, 'state_code')
    elif args.data_name == 'loan':
        preprocess_loan(data, args.num_feature, 'Grade')
    elif args.data_name == 'mobile':
        preprocess_mobile(data, args.num_feature, 'dual_sim')



#example
# python process_real_data.py --file_path "data/startup.csv" --target_col "status" --subgroup_cols "is_CA,is_NY,is_MA,is_TX,is_otherstate" --threshold 0.2 --num_feature 10
# python process_real_data.py --file_path "data/loan.csv" --target_col "Loan Status" --subgroup_cols "Grade" --threshold 0.2 --num_feature 10
