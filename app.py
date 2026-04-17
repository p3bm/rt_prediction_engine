import streamlit as st
import pandas as pd
import tempfile
import zipfile
import numpy as np
import json

from backend.data import custom_flag_split, split_data
from backend.model import train_model, fit_model_with_params
from backend.evaluation import evaluate, applicability_domain, bootstrap_ci
from backend.plotting import parity_plot, williams_plot
from backend.shap_analysis import compute_shap, shap_summary_plot
from backend.utils import set_seed, save_model, save_log, create_run_dir, clean_params

st.title("RT Prediction Engine")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    filter_cols = st.multiselect("Select columns to filter data by", numeric_cols)

    filters = []
    for col in filter_cols:
        col1, col2 = st.columns(2)
        with col1:
            operator = st.selectbox(f"Operator for {col}", [">", "=", "<"], key=f"op_{col}")
        with col2:
            value = st.number_input(f"Value for {col}", key=f"val_{col}")
        filters.append((col, operator, value))
    
    # Apply filters
    for col, op, val in filters:
        if op == ">":
            df = df[df[col] > val]
        elif op == "=":
            df = df[df[col] == val]
        elif op == "<":
            df = df[df[col] < val]

    target = st.selectbox("Target", df.columns)
    drop_cols = st.multiselect(
        "Columns to exclude from modelling",
        [col for col in df.columns if col != target]
    )

    st.write("Data Preview:")
    st.dataframe(df.head())

    if st.toggle("Use fraction of training data"):
        frac = st.number_input("Enter a fraction between 0 and 1", min_value=0.1, max_value=1.0, step=0.1, value=1.0)
        df = df.sample(frac=frac, axis=0, ignore_index=True)

    split_mode = st.radio(
            "Splitting mode",
            ["Traditional", "Custom"]
        )

    if split_mode == "Traditional":
        group = st.selectbox("Group column", ["None"] + list(df.columns))
        stratify = st.selectbox("Stratify by column", ["None"] + list(df.columns))
        split_ratio = st.number_input("Split ratio", min_value=0.10, max_value=1.00, value=0.2, step=0.05)
        split_col = None
        split_frac = None
    else:
        group = "None"
        stratify = "None"
        split_ratio = None
        split_col = st.selectbox("Select column to split data by", numeric_cols)
        split_frac = st.number_input(f"What fraction of {split_col} split data to add back to training set?",
                                     min_value=0.0, max_value=0.9, step=0.1, value=0.0)

    seed = st.number_input("Random seed", value=42)
    set_seed(seed)

    var_thresh = st.slider("Variance threshold", 0.0, 0.2, 0.01)
    corr_thresh = st.slider("Correlation threshold", 0.7, 0.99, 0.95)

    mode = st.radio(
            "Training mode",
            ["Random Search", "Use fixed parameters"]
        )

    if mode == "Random Search":
        selected_models = st.multiselect("Select the models to use in the randomised search CV",
                                         ["ridge", "lasso", "elasticnet", "rf", "gbr", "lgbm", "catboost"])
        n_iter = st.number_input("Number of randomised search iterations", min_value=1, max_value=1000, value=30, step=1)
    else:
        selected_models = None
        n_iter = None

    if mode == "Use fixed parameters":
        model_choice = st.selectbox(
            "Select model",
            ["ridge", "lasso", "elasticnet", "rf", "gbr", "lgbm", "catboost"]
        )

        log_file = st.file_uploader("Upload previous log.json (optional)", type="json")
    
        params_input = st.text_area(
            "Enter model parameters (JSON format)",
            value='{\n  "model__alpha": 1.0\n}'
        )

        if log_file:
            log_data = json.load(log_file)
            best_params = log_data.get("best_params", {})
            st.write("Loaded parameters:", best_params)
        else:
            try:
                best_params = json.loads(params_input)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

    feature_selection = st.toggle("Enable feature selection")

    if feature_selection:
        selected_features_file = st.file_uploader("Upload previously generated selected_features CSV file", type="csv")
        if selected_features_file:
            features_to_use = pd.read_csv(selected_features_file, sep=",")
            feature_list = features_to_use["feature"].to_list()
    
    shap_toggle = st.toggle("Perform SHAP Analysis")
    ci_toggle = st.toggle("Calculate confidence interval (takes a long time)")

    aim = st.text_area("Experiment aim")

    if st.button("Train model"):
        
        groups = df[group] if group != "None" else None
        stratify_col = df[stratify] if stratify != "None" else None

        if target in drop_cols:
            st.error("Target column cannot be dropped.")
            st.stop()
        
        if group in drop_cols:
            st.warning("Group column is excluded from features but still used for splitting.")
        
        if stratify in drop_cols:
            st.warning("Stratify column is excluded from features but still used for splitting.")

        if split_mode == "Traditional":
            y = df[target]
            
            X = df.drop(columns=[target] + drop_cols, errors="ignore")
            X = X.select_dtypes(include=[np.number])
            
            X_train, X_test, y_train, y_test = split_data(
                X, y,
                groups,
                stratify_col,
                split_ratio,
                seed
            )
            
        else:
            train_df, test_df = custom_flag_split(df, flag_col=split_col)

            if split_frac != 0.0:
                train_df_additional = test_df.sample(frac=split_frac, random_state=seed)
                test_df = test_df.drop(train_df_additional.index)
                train_df = pd.concat((train_df,train_df_additional), axis=0)

            y_train = train_df[target]
            y_test = test_df[target]

            X_train = train_df.drop(columns=[target] + drop_cols, errors="ignore")
            X_train = X_train.select_dtypes(include=[np.number])

            X_test = test_df.drop(columns=[target] + drop_cols, errors="ignore")
            X_test = X_test.select_dtypes(include=[np.number])

        if feature_selection:
                X_train = X_train[feature_list]
                X_test = X_test[feature_list]
        
        if mode == "Random Search":
            model, search = train_model(
                X_train,
                y_train,
                seed,
                selected_models=selected_models,
                n_iter=n_iter,
                var_thresh=var_thresh,
                corr_thresh=corr_thresh
            )
            best_params_clean = clean_params(search.best_params_)
        
        else:
            model = fit_model_with_params(
                X_train,
                y_train,
                seed,
                model_key=model_choice,
                best_params=best_params,
                var_thresh=var_thresh,
                corr_thresh=corr_thresh,
            )
            search = None
            best_params_clean = best_params

        results = evaluate(model, X_train, X_test, y_train, y_test)

        st.write(f"Training set RMSE: {results['rmse_train']}")
        st.write(f"Test set RMSE: {results['rmse_test']}")
        st.write(f"Training set R2: {results['r2_train']}")
        st.write(f"Test set R2: {results['r2_test']}")

        # Plots
        fig1 = parity_plot(y_test, results["y_pred_test"])
        st.pyplot(fig1)

        # Apply full preprocessing pipeline (excluding final model)
        X_processed = model[:-1].transform(X_test)
        
        # Compute leverage + Williams plot
        h, h_star, std_res, flags = applicability_domain(
            X_processed,
            y_test,
            results["y_pred_test"]
        )
        
        fig2 = williams_plot(h, std_res, h_star)
        st.pyplot(fig2)

        # SHAP
        if shap_toggle:          
            shap_values, X_sample_named = compute_shap(model, X_train, sample_size=100, seed=seed)
            
            shap_fig = shap_summary_plot(shap_values, X_sample_named)
            st.pyplot(shap_fig)

            shap_importance = np.abs(shap_values.values).mean(axis=0)

            feature_importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": shap_importance
            }).sort_values(by="importance", ascending=False)

            feature_importance_df["cumulative"] = (
                feature_importance_df["importance"].cumsum() /
                feature_importance_df["importance"].sum()
            )
            
            selected_features = feature_importance_df[feature_importance_df["cumulative"] <= 0.95]["feature"]
            
        # CI
        if ci_toggle:
            lower, upper = bootstrap_ci(model, X_train, y_train, X_test)
            ci_width = upper - lower
            mean_ci_width = np.mean(ci_width)
            rel_uncertainty = ci_width / np.abs(results["y_pred_test"])
            mean_rel_uncertainty = np.mean(rel_uncertainty)
            st.write("Mean CI width:", mean_ci_width)
            st.write("Mean relative uncertainty:", mean_rel_uncertainty)
        else:
            mean_ci_width = None
            mean_rel_uncertainty = None

        run_dir = create_run_dir()
        
        model_path = f"{run_dir}/model.joblib"
        log_path = f"{run_dir}/log.json"
        parity_path = f"{run_dir}/parity.png"
        williams_path = f"{run_dir}/williams.png"
        cv_path = f"{run_dir}/cv_results.csv"

        if shap_toggle:
            shap_path = f"{run_dir}/shap.png"
            selected_features_path = f"{run_dir}/selected_features.csv"
            
        if search is not None:
            cv_df = pd.DataFrame(search.cv_results_)
            cv_df.to_csv(cv_path, index=False)

        save_model(model_path, {"model": model})
        save_log(log_path, {
            "aim": aim,
            "dataset_filename": file.name if file else None,
            "r2_train": results["r2_train"],
            "r2_test": results["r2_test"],
            "rmse_train": results["rmse_train"],
            "rmse_test": results["rmse_test"],
            "n_features": X_train.shape[1],
            "filters": filters ,
            "split_mode": split_mode,
            "split_col": split_col,
            "models_tested": selected_models,
            "no_of_iterations": n_iter,
            "best_model_type": str(model.named_steps["model"]),
            "best_params": best_params_clean,
            "Mean CI width": mean_ci_width,
            "Mean relative uncertainty": mean_rel_uncertainty,
            "manually_dropped_columns": drop_cols,
            "stratified_by": stratify if stratify != "None" else None,
            "grouped_by": group if group != "None" else None,
            "split_ratio": split_ratio,
            "seed": seed,
        })

        fig1.savefig(parity_path, dpi=300)
        fig2.savefig(williams_path, dpi=300)
        if shap_toggle:
            shap_fig.savefig(shap_path, dpi=300)
            selected_features.to_csv(selected_features_path, sep=",", index=False)
        if feature_selection:
            feature_list_path = f"{run_dir}/feature_list_for_filtering.csv"
            features_to_use.to_csv(feature_list_path, sep=",", index=False)
