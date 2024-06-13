import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DateToOrdinal(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.date_to_ordinal_mapping = {}
        self.ordinal_to_date_mapping = {}

    def fit(self, X, y=None):
        # Ensure that X is a 1D series or a 1D array
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            X = X.ravel()
        unique_dates = pd.Series(X).unique()
        self.date_to_ordinal_mapping = {date: idx for idx, date in enumerate(unique_dates)}
        self.ordinal_to_date_mapping = {idx: date for idx, date in enumerate(unique_dates)}
        return self

    def transform(self, X):
        # Ensure that X is a 1D series or a 1D array
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            X = X.ravel()
        return pd.Series(X).map(self.date_to_ordinal_mapping).values.reshape(-1, 1)

    def inverse_transform(self, X):
        return pd.Series(X.flatten()).map(self.ordinal_to_date_mapping).values
