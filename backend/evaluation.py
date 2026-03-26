import numpy as np
from sklearn.metrics import r2_score

def evaluate(model, scaler, X_train, X_test, y_train, y_test):
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return {
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "overfit": r2_train - r2_test
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

    flags = []
    for i in range(len(h)):
        if h[i] > h_star:
            flags.append("High leverage")
        elif abs(std_res[i]) > 3:
            flags.append("Outlier")
        else:
            flags.append("In domain")

    return h, h_star, std_res, flags

def bootstrap_ci(model, scaler, X_train, y_train, X_test, n_boot=50):
    preds = []
    n = len(X_train)

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        X_s = scaler.fit_transform(X_train.iloc[idx])
        model.fit(X_s, y_train.iloc[idx])
        preds.append(model.predict(scaler.transform(X_test)))

    preds = np.array(preds)
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)
