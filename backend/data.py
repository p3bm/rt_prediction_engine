import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def custom_flag_split(df, flag_col):
    train_df = df[df[flag_col] == 0].copy()
    test_df  = df[df[flag_col] == 1].copy()
    
    return train_df, test_df

def fit_preprocessing(X, target_col, drop_cols, var_thresh, corr_thresh):

    original_columns = X.columns

    # Variance filter
    var_sel = VarianceThreshold(var_thresh)
    X_var = var_sel.fit_transform(X)

    var_mask = var_sel.get_support()
    kept_cols = original_columns[var_mask]

    X_var = pd.DataFrame(X_var, columns=kept_cols, index=X.index)

    # Correlation filter (fit only on training)
    corr = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > corr_thresh)]

    X_final = X_var.drop(columns=drop_corr)

    feature_names = X_final.columns.tolist()

    return X_final, var_sel, drop_corr, feature_names

def transform_preprocessing(X, target_col, drop_cols, var_sel, drop_corr, feature_names):

    # Apply variance selector
    X_var = var_sel.transform(X)

    var_mask = var_sel.get_support()
    kept_cols = X.columns[var_mask]

    X_var = pd.DataFrame(X_var, columns=kept_cols, index=X.index)

    # Drop correlated features from training
    X_final = X_var.drop(columns=drop_corr, errors="ignore")

    # Ensure same column order as training
    X_final = X_final.reindex(columns=feature_names, fill_value=0)

    return X_final
