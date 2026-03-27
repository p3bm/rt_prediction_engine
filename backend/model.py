import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd

def stratified_split(X, y, stratify_col, test_size, seed, n_bins=10):
    # Bin the stratification column
    stratify_vals = pd.qcut(stratify_col, q=n_bins, duplicates="drop")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed
    )

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

def get_model_space(seed):
    return [
        {
            "model": [Ridge()],
            "model__alpha": np.logspace(-3, 3, 50)
        },
        {
            "model": [Lasso(max_iter=10000)],
            "model__alpha": np.logspace(-4, 1, 50)
        },
        {
            "model": [ElasticNet(max_iter=10000)],
            "model__alpha": np.logspace(-4, 1, 30),
            "model__l1_ratio": np.linspace(0.1, 0.9, 10)
        },
        {
            "model": [RandomForestRegressor(random_state=seed)],
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 5, 10, 20]
        },
        {
            "model": [GradientBoostingRegressor(random_state=seed)],
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5]
        }
    ]

def train_model(X_train, y_train, seed, groups=None):

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())  # placeholder
    ])

    param_dist = get_model_space(seed)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=60,
        scoring="r2",
        cv=5 if groups is None else GroupShuffleSplit(n_splits=5, random_state=seed),
        n_jobs=-1,
        random_state=seed,
        verbose=1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search
