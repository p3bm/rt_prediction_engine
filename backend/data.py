import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def preprocess_data(df, target_col, var_thresh, corr_thresh):
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # Variance filter
    var_sel = VarianceThreshold(var_thresh)
    X_var = var_sel.fit_transform(X)
    X_var = pd.DataFrame(X_var)

    # Correlation filter
    corr = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    X_final = X_var.drop(columns=drop_cols)

    return X_final, y, var_sel, drop_cols
