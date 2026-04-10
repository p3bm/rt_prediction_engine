import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from data import CorrelationFilter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd

def base_pipe(model, var_thresh, corr_thresh):
    return Pipeline([
        ("var", VarianceThreshold(var_thresh)),
        ("corr", CorrelationFilter(corr_thresh)),
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def get_model_space(seed, selected_models, var_thresh, corr_thresh):
    model_space = []

    if "ridge" in selected_models:
        model_space.append({
            "model": [base_pipe(Ridge(), var_thresh, corr_thresh)],
            "model__model__alpha": np.logspace(-3, 3, 50)
        })

    if "lasso" in selected_models:
        model_space.append({
            "model": [base_pipe(Lasso(max_iter=10000), var_thresh, corr_thresh)],
            "model__model__alpha": np.logspace(-4, 1, 50)
        })

    if "elasticnet" in selected_models:
        model_space.append({
            "model": [base_pipe(ElasticNet(max_iter=10000), var_thresh, corr_thresh)],
            "model__model__alpha": np.logspace(-4, 1, 30),
            "model__model__l1_ratio": np.linspace(0.1, 0.9, 10)
        })

    if "rf" in selected_models:
        model_space.append({
            "model": [base_pipe(RandomForestRegressor(random_state=seed), var_thresh, corr_thresh)],
            "model__model__n_estimators": [100, 200, 500],
            "model__model__max_depth": [None, 5, 10, 20]
        })

    if "gbr" in selected_models:
        model_space.append({
            "model": [base_pipe(GradientBoostingRegressor(random_state=seed), var_thresh, corr_thresh)],
            "model__model__n_estimators": [100, 300],
            "model__model__learning_rate": [0.01, 0.05, 0.1]
        })

    if "lgbm" in selected_models:
        model_space.append({
            "model": [base_pipe(LGBMRegressor(random_state=seed, verbose=-1), var_thresh, corr_thresh)],
            "model__model__n_estimators": [200, 500],
            "model__model__learning_rate": [0.01, 0.05, 0.1]
        })

    if "catboost" in selected_models:
        model_space.append({
            "model": [base_pipe(CatBoostRegressor(random_state=seed, verbose=0), var_thresh, corr_thresh)],
            "model__model__iterations": [300, 600]
        })

    return model_space

# -----------------------------
# Training
# -----------------------------
def train_model(X_train, y_train, seed, selected_models, var_thresh, corr_thresh, n_iter=30):

    param_dist = get_model_space(seed, selected_models, var_thresh, corr_thresh)

    search = RandomizedSearchCV(
        estimator=Pipeline([("model", Ridge())]),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=5,
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
    var_thresh,
    corr_thresh
):
    """
    Fit a model using fixed parameters with FULL pipeline (no leakage).

    Must mirror the pipeline used during RandomizedSearchCV.
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

    # ✅ FULL pipeline (same as training)
    pipe = Pipeline([
        ("var", VarianceThreshold(var_thresh)),
        ("corr", CorrelationFilter(corr_thresh)),
        ("scaler", StandardScaler()),
        ("model", model_map[model_key])
    ])

    # ✅ Apply parameters (already in pipeline format)
    pipe.set_params(**best_params)

    # Fit model
    pipe.fit(X_train, y_train)

    return pipe
