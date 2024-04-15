import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KDTree
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
from sklearn.preprocessing import KBinsDiscretizer
import warnings

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

                for col in X.columns[X.iloc[idx].isnull()]:
                    neighbor_vals = self.data_encoded.iloc[neighbors][col]
                    most_common = mode(neighbor_vals).mode
                    X.at[idx, col] = most_common
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
