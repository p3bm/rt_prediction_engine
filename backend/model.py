import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
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

def get_model_space(seed, selected_models=None):
    """
    Build parameter space based on selected models.

    selected_models: list of strings
    Options:
        "ridge", "lasso", "elasticnet", "rf", "gbr", "lgbm", "catboost"
    """

    model_space = []

    if selected_models is None:
        selected_models = ["ridge", "lasso", "elasticnet", "rf", "gbr"]

    if "ridge" in selected_models:
        model_space.append({
            "model": [Ridge()],
            "model__alpha": np.logspace(-3, 3, 50)
        })

    if "lasso" in selected_models:
        model_space.append({
            "model": [Lasso(max_iter=10000)],
            "model__alpha": np.logspace(-4, 1, 50)
        })

    if "elasticnet" in selected_models:
        model_space.append({
            "model": [ElasticNet(max_iter=10000)],
            "model__alpha": np.logspace(-4, 1, 30),
            "model__l1_ratio": np.linspace(0.1, 0.9, 10)
        })

    if "rf" in selected_models:
        model_space.append({
            "model": [RandomForestRegressor(random_state=seed)],
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 5, 10, 20]
        })

    if "gbr" in selected_models:
        model_space.append({
            "model": [GradientBoostingRegressor(random_state=seed)],
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5]
        })

    if "lgbm" in selected_models:
        model_space.append({
            "model": [LGBMRegressor(random_state=seed, verbose=-1)],
            "model__n_estimators": [200, 500, 1000],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [-1, 5, 10],
            "model__num_leaves": [31, 64, 128],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0]
        })
    
    if "catboost" in selected_models:
        model_space.append({
            "model": [CatBoostRegressor(random_state=seed, verbose=0)],
            "model__iterations": [300, 600, 1000],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__depth": [4, 6, 8],
            "model__l2_leaf_reg": [1, 3, 5, 7],
            "model__bagging_temperature": [0, 1, 5]
        })

    if len(model_space) == 0:
        raise ValueError("No valid models selected.")

    return model_space


# -----------------------------
# Training
# -----------------------------
def train_model(
    X_train,
    y_train,
    seed,
    groups=None,
    selected_models=None,
    n_iter=30
):
    """
    Train model using RandomizedSearchCV.

    Parameters:
        selected_models: list of model keys
        n_iter: number of random search iterations
    """

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())  # placeholder
    ])

    param_dist = get_model_space(seed, selected_models)

    cv_strategy = (
        GroupShuffleSplit(n_splits=5, random_state=seed)
        if groups is not None
        else 5
    )

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv_strategy,
        n_jobs=-1,
        random_state=seed,
        verbose=1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search

def fit_model_with_params(
    X_train,
    y_train,
    seed,
    model_key,
    best_params,
    scale=True
):
    """
    Fit a model using a fixed set of parameters (no hyperparameter search).

    Parameters:
        model_key (str): one of ["ridge", "lasso", "elasticnet", "rf", "gbr"]
        best_params (dict): parameters from RandomizedSearchCV (search.best_params_)
        scale (bool): whether to include StandardScaler in pipeline

    Returns:
        fitted pipeline
    """

    # Map model key to actual model
    model_map = {
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=10000),
        "elasticnet": ElasticNet(max_iter=10000),
        "rf": RandomForestRegressor(random_state=seed),
        "gbr": GradientBoostingRegressor(random_state=seed),
        "lgbm": LGBMRegressor(random_state=seed, verbose=-1),
        "catboost": CatBoostRegressor(random_state=seed, verbose=0),
    }

    if model_key not in model_map:
        raise ValueError(f"Invalid model_key: {model_key}")

    # Build pipeline
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model_map[model_key]))

    pipe = Pipeline(steps)

    # Apply parameters (must match pipeline format, e.g. model__alpha)
    pipe.set_params(**best_params)

    # Fit model
    pipe.fit(X_train, y_train)

    return pipe
