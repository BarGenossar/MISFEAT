import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.X_filled = X.reset_index(drop=True)
        return self

    def _hamming_distance(self, row, other_rows):
        distances = np.zeros(len(other_rows))
        for i, other_row in enumerate(other_rows.itertuples(index=False, name=None)):
            valid_cols = ~np.isnan(np.array(row)) & ~np.isnan(np.array(other_row))
            if not valid_cols.any():
                distances[i] = 0
            else:
                hamming_dist = np.sum(np.array(row)[valid_cols] != np.array(other_row)[valid_cols])
                distances[i] = hamming_dist / valid_cols.sum()
        return distances

    def transform(self, X, y=None):
        X_imputed = X.copy()
        for i, row in X_imputed.iterrows():
            if row.isnull().any():
                dists = self._hamming_distance(row.values, self.X_filled)
                nn_indices = np.argsort(dists)[:self.n_neighbors]
                for col in row[row.isnull()].index:
                    nn_cats = self.X_filled.iloc[nn_indices][col].dropna()
                    if not nn_cats.empty:
                        imputed_val = nn_cats.value_counts().idxmax()
                        X_imputed.at[i, col] = imputed_val
        return X_imputed

# Sample data creation
data = pd.DataFrame({
    'Feature1': [1, 2, 0, 1, 2, np.nan, 0, 1],
    'Feature2': [np.nan, 1, 0, 1, np.nan, 0, 1, 0]
})

print("Original Data:")
print(data)

# Initialize and use the CustomKNNImputer
imputer = CustomKNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(data)

print("\nImputed Data:")
print(imputed_data)