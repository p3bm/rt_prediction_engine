import autosklearn.regression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def split_data(X, y, groups, test_size, seed):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    else:
        return train_test_split(X, y, test_size=test_size, random_state=seed)

def train_model(X_train, y_train, time_limit, per_run_time, seed):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time,
        n_jobs=-1,
        seed=seed
    )

    automl.fit(X_train_scaled, y_train)

    return automl, scaler
