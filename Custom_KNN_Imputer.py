import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KDTree
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
import warnings
import matplotlib.pyplot as plt


class CustomKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        warnings.filterwarnings('ignore')

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy='most_frequent')
        data_filled = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        self.data_encoded = pd.get_dummies(data_filled)
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='hamming')
        self.nn.fit(self.data_encoded)
        return self

    def transform(self, X, y=None):
        data_filled = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        data_encoded = pd.get_dummies(data_filled)

       
        for idx in range(data_filled.shape[0]):
            if X.iloc[idx].isnull().any():
                sample_encoded = data_encoded.iloc[[idx]]
                neighbors = self.nn.kneighbors(sample_encoded, return_distance=False)[0]
                # print(neighbors)
                # exit()
                for col in X.columns[X.iloc[idx].isnull()]:
                    neighbor_vals = self.data_encoded.iloc[neighbors][col]
                    # neighbor_vals.plot(kind='hist')
                    # plt.savefig(f'{col}.png')
                    # plt.close()
                    most_common = mode(neighbor_vals).mode
                    X.at[idx, col] = most_common
                # exit()
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)



# np.random.seed(42)
# data = pd.DataFrame(np.random.randn(10000, 10), columns=[f'Feature{i}' for i in range(1, 11)])

# est = KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='uniform')
# data_categorical = pd.DataFrame(est.fit_transform(data), columns=data.columns)

# for col in data_categorical.columns:
#     data_categorical.loc[data.sample(frac=0.3).index, col] = np.nan

# print("Original Data with Missing Values (Categorical):")
# print(data_categorical.head())

# imputer = CustomKNNImputer(n_neighbors=10)
# imputed_data = imputer.fit_transform(data_categorical)
# imputed_data = pd.DataFrame(imputed_data, columns=data.columns)

# print("\nImputed Data:")
# print(imputed_data.head())
















# class CustomKNNImputer(BaseEstimator, TransformerMixin):
#     def __init__(self, n_neighbors=3):
#         self.n_neighbors = n_neighbors
#         warnings.filterwarnings('ignore')  # Optionally ignore warnings

#     def fit(self, X, y=None):
#         # Impute missing values using the most frequent value
#         self.imputer = SimpleImputer(strategy='most_frequent')
#         self.data_encoded = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
#         # Build NearestNeighbors using the Hamming distance
#         self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='hamming', algorithm='ball_tree')
#         self.nn.fit(self.data_encoded)
#         return self

#     def transform(self, X, y=None):
#         # Transform and impute the incoming data
#         data_encoded = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        
#         # Iterate through each column and replace missing values using KNN
#         for col in data_encoded.columns:
#             missing_indices = data_encoded[col].isnull()  # Identifying missing indices
#             if not missing_indices.any():
#                 continue
            
#             X_missing = data_encoded.loc[missing_indices]
#             # Ensure the data structure is appropriate for querying
#             if X_missing.ndim == 1:
#                 X_missing = X_missing.values.reshape(1, -1)

#             # Get indices of neighbors for all missing data points
#             neighbors = self.nn.kneighbors(X_missing, return_distance=False)
            
#             # Compute mode of the nearest neighbors' categories
#             neighbor_data = self.data_encoded.iloc[neighbors.flatten(), data_encoded.columns.get_loc(col)]
#             y_imputed, _ = mode(neighbor_data, axis=0)
#             y_imputed = y_imputed.mode[0] if y_imputed.size else data_encoded[col].mode()[0]

#             # Replace missing values with the computed mode
#             data_encoded.loc[missing_indices, col] = y_imputed

#         return data_encoded

#     def fit_transform(self, X, y=None):
#         return self.fit(X).transform(X)
    

# # Sample data creation
# # data = pd.DataFrame({
# #     'Feature1': [1, 2, 0, 1, 2, np.nan, 0, 1],
# #     'Feature2': [np.nan, 1, 0, 1, np.nan, 0, 1, 0]
# # })

# # print("Original Data:")
# # print(data)

# # # Initialize and use the CustomKNNImputer
# # imputer = CustomKNNImputer(n_neighbors=5)
# # imputed_data = imputer.fit_transform(data)

# # print("\nImputed Data:")
# # print(imputed_data)