import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit, StratifiedShuffleSplit

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_cols_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.keep_cols_ = [col for col in X.columns if col not in self.drop_cols_]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.keep_cols_]

def custom_flag_split(df, flag_col):
    train_df = df[df[flag_col] == 0].copy()
    test_df  = df[df[flag_col] == 1].copy()
    
    return train_df, test_df

def stratified_split(X, y, stratify_col, test_size, seed, n_bins=10):
    stratify_vals = pd.qcut(stratify_col, q=n_bins, duplicates="drop")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, stratify_vals))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx]
    )

def split_data(X, y, groups, stratify_col, test_size, seed):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    elif stratify_col is not None:
        return stratified_split(X, y, stratify_col, test_size, seed)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=seed)
