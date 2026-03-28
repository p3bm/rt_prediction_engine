import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def preprocess_data(df, target_col, drop_cols, var_thresh, corr_thresh):

    y = df[target_col]

    # Drop unwanted columns
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    original_columns = X.columns

    # Variance filter
    var_sel = VarianceThreshold(var_thresh)
    X_var = var_sel.fit_transform(X)

    # Get kept columns after variance filter
    var_mask = var_sel.get_support()
    X_var = pd.DataFrame(X_var, columns=original_columns[var_mask])

    # Correlation filter
    corr = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > corr_thresh)]

    X_final = X_var.drop(columns=drop_corr)

    # FINAL FEATURE NAMES
    feature_names = X_final.columns.tolist()

    return X_final, y, var_sel, drop_corr, feature_names
