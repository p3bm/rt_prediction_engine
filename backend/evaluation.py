import numpy as np
from sklearn.metrics import r2_score
from sklearn.base import clone

def evaluate(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "overfit": r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test)
    }

def leverage(X):
    X = np.array(X)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return np.diag(H)

def applicability_domain(X, y_true, y_pred):
    h = leverage(X)
    residuals = y_true - y_pred

    n, p = X.shape
    h_star = 3 * (p + 1) / n
    std_res = residuals / np.std(residuals)

    flags = [
        "In domain" if (h[i] <= h_star and abs(std_res[i]) <= 3)
        else "Outlier" if abs(std_res[i]) > 3
        else "High leverage"
        for i in range(len(h))
    ]

    return h, h_star, std_res, flags

def bootstrap_ci(model, X_train, y_train, X_test, n_boot=50):
    preds = []
    n = len(X_train)

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        m = clone(model)
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        preds.append(m.predict(X_test))

    preds = np.array(preds)
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)
