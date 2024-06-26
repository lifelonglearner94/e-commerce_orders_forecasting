import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DateToOrdinal(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.date_to_ordinal_mapping = {}
        self.ordinal_to_date_mapping = {}
        self.last_ordinal = -1

    def fit(self, X, y=None):
        # Ensure that X is a 1D series or a 1D array
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            X = X.ravel()

        unique_dates = pd.Series(X).unique()
        self.date_to_ordinal_mapping = {date: idx for idx, date in enumerate(unique_dates)}
        self.ordinal_to_date_mapping = {idx: date for idx, date in enumerate(unique_dates)}
        self.last_ordinal = len(unique_dates) - 1

        return self

    def transform(self, X):
        # Ensure that X is a 1D series or a 1D array
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            X = X.ravel()

        X_transformed = []
        for date in X:
            if date not in self.date_to_ordinal_mapping:
                self.last_ordinal += 1
                self.date_to_ordinal_mapping[date] = self.last_ordinal
                self.ordinal_to_date_mapping[self.last_ordinal] = date
            X_transformed.append(self.date_to_ordinal_mapping[date])

        return np.array(X_transformed).reshape(-1, 1)

    def inverse_transform(self, X):
        return pd.Series(X.flatten()).map(self.ordinal_to_date_mapping).values
