import json
import pickle
import numpy as np
import os
import pandas as pd
from itertools import combinations
from sklearn.metrics import mutual_info_score


class LogicalDatasetGenerator:
    def __init__(self, formula_name, config_num, formula, hyperparams):
        np.random.seed(int(config_num))
        self.seed = int(config_num)
        self.name = formula_name
        self.subgroups_num = hyperparams['subgroups_num']
        self.feature_num = hyperparams['feature_num']
        self.formula = formula
        self.hyperparams = hyperparams
        self.operation_pool = ['and', 'or', 'xor']
        self.feature_pool = [f'x_{i}' for i in range(self.feature_num)]
        self.relevant_features = self._get_relevant_features()
        self.redundant_features = self._get_redundant_features()
        self.correlated_features = self._get_correlated_features()
        self.noisy_features = self.feature_pool
        self.dataset = self.generate_values()

    def _get_relevant_features(self):
        relevant_features = [expr for expr in self.formula.split(' ') if expr not in self.operation_pool]
        self.feature_pool = [f for f in self.feature_pool if f not in relevant_features]
        return relevant_features

    def _get_redundant_features(self):
        redundant_features = list(np.random.choice(self.feature_pool, self.hyperparams['redundant_num'], replace=False))
        self.feature_pool = [f for f in self.feature_pool if f not in redundant_features]
        min_subset_size = self.hyperparams['redundant_min_subgroup']
        max_subset_size = self.hyperparams['redundant_max_subgroup']
        redundant_dict = {f: self._generate_redundant_expression(min_subset_size, max_subset_size)
                          for f in redundant_features}
        return redundant_dict

    def _generate_redundant_expression(self, min_subset_size, max_subset_size):
        subset_size = np.random.randint(min_subset_size, max_subset_size + 1)
        subset = np.random.choice(self.relevant_features, subset_size)
        operations = np.random.choice(self.operation_pool, subset_size - 1)
        interlacing_string = ' '.join([f'{subset[i]} {operations[i]}'
                                       for i in range(subset_size - 1)]) + f' {subset[-1]}'
        return interlacing_string

    def _get_correlated_features(self):
        correlated_features = list(np.random.choice(self.feature_pool,
                                   self.hyperparams['correlated_num'], replace=False))
        self.feature_pool = [f for f in self.feature_pool if f not in correlated_features]
        min_prob = self.hyperparams['correlated_min_probability']
        max_prob = self.hyperparams['correlated_max_probability']
        correlated_dict = {f: np.random.uniform(min_prob, max_prob) for f in correlated_features}
        return correlated_dict

    def generate_values(self):
        sample_size = self.hyperparams['sample_size']
        tmp_dataset = dict()
        for feature in self.relevant_features:
            tmp_dataset[feature] = np.random.randint(0, 2, sample_size)
        tmp_dataset['y'] = self._compute_by_formula(tmp_dataset, self.formula)
        tmp_dataset = self._modify_by_features(tmp_dataset, sample_size)
        df = self._generate_dataframe(tmp_dataset)
        return df

    def _modify_by_features(self, tmp_dataset, sample_size):
        tmp_dataset = self._get_correlated_vals(tmp_dataset, sample_size)
        tmp_dataset = self._get_redundant_vals(tmp_dataset, sample_size)
        tmp_dataset = self._get_noisy_vals(tmp_dataset, sample_size)
        # tmp_dataset = self._add_random_noise(tmp_dataset, sample_size)
        return tmp_dataset

    def _generate_dataframe(self, tmp_dataset):
        df = pd.DataFrame(tmp_dataset)
        df = df[[f'x_{i}' for i in range(self.feature_num)] + ['y']]
        df['subgroup'] = np.random.randint(0, self.subgroups_num, len(df))
        df = self._add_random_noise(df)
        return df

    def _compute_by_formula(self, data, formula):
        expressions = formula.split(' ')
        op_dict = {'and': np.logical_and, 'or': np.logical_or, 'xor': np.logical_xor, 'not': np.logical_not}
        clause_results = []
        for i, clause in enumerate(expressions):
            if clause not in self.operation_pool:
                clause_results.append(data[clause])
            else:
                op_func = op_dict[clause]
                clause1_result = clause_results.pop()
                clause2_result = data[expressions[i + 1]]
                result = op_func(clause1_result, clause2_result)
                clause_results.append(result)
                expressions.pop(i + 1)
        return clause_results[0].astype(int)

    def _get_correlated_vals(self, dataset, sample_size):
        for feature in self.correlated_features:
            dataset[feature] = np.where(np.random.rand(sample_size) <
                                        self.correlated_features[feature],
                                        dataset['y'], 1 - dataset['y'])
        return dataset

    def _get_redundant_vals(self, dataset, sample_size):
        redundant_flip_prob = self.hyperparams['redundant_flip_probability']
        for feature in self.redundant_features:
            computed_vals = self._compute_by_formula(dataset, self.redundant_features[feature])
            dataset[feature] = np.where(np.random.rand(sample_size) > redundant_flip_prob,
                                        computed_vals, 1 - computed_vals)
        return dataset

    def _get_noisy_vals(self, dataset, sample_size):
        for feature in self.noisy_features:
            dataset[feature] = np.random.randint(0, 2, sample_size)
        return dataset

    def _add_random_noise(self, df):
        random_noise_mean = self.hyperparams['random_noise_mean']
        random_noise_std = self.hyperparams['random_noise_std']
        feature_list = [f'x_{i}' for i in range(self.feature_num)]
        for subgroup in range(self.subgroups_num):
            tmp_df = df[df['subgroup'] == subgroup].copy()
            for feature in feature_list:
                noise_flip_prob = np.random.normal(random_noise_mean, random_noise_std)
                tmp_df[feature] = np.where(np.random.rand(len(tmp_df)) > noise_flip_prob,
                                           tmp_df[feature], 1 - tmp_df[feature])
            df.loc[tmp_df.index] = tmp_df
        return df

    def create_description(self):
        description = """"""
        description += f"Formula {self.name}: {self.formula}\n"
        description += f"Seed: {self.seed}\n"
        description = self._add_hyperparams_to_description(description)
        description += f"Relevant features: {', '.join(sorted(list(self.relevant_features)))}\n\n"
        description += f"Redundant features and the formulas that define them: \n\n"
        for feature, formula in self.redundant_features.items():
            description += f"\t{feature}: {formula}\n"
        description += f"\nCorrelated features and probabilities of being equal to label: \n"
        for feature, prob in self.correlated_features.items():
            description += f"\t{feature}: {round(prob, 3)}\n"
        return description

    def _add_hyperparams_to_description(self, description):
        description += "========================================\n"
        description += "Hyperparameters:\n"
        for key, value in self.hyperparams.items():
            description += f"\t{key}: {value}\n"
        description += "========================================\n"
        description += "\n"
        return description

    def save(self):
        if not self.save:
            return
        path = f'GeneratedData/Formula{self.name}/Config{self.seed}'
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(self.dataset, open(f'{path}/dataset.pkl', 'wb'))
        with open(f'{path}/description.txt', 'w') as f:
            f.write(self.create_description())


if __name__ == '__main__':
    configs = json.load(open('data_generation_config.json'))
    for formula in configs['formulas']:
        for config_num in configs['hyperparams']:
            dataset = LogicalDatasetGenerator(formula, config_num, configs['formulas'][formula]['formula'],
                                              configs['hyperparams'][config_num])
            dataset.save()

