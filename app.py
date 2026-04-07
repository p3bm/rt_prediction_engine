import streamlit as st
import pandas as pd
import tempfile
import zipfile
import numpy as np
import json

from backend.data import preprocess_data
from backend.model import split_data, train_model, fit_model_with_params
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
    
    group = st.selectbox("Group column", ["None"] + list(df.columns))
    stratify = st.selectbox("Stratify by column", ["None"] + list(df.columns))
    split_ratio = st.number_input("Split ratio", min_value=0.10, max_value=1.00, value=0.2, step=0.05)

    seed = st.number_input("Random seed", value=42)
    set_seed(seed)

    var_thresh = st.slider("Variance threshold", 0.0, 0.2, 0.01)
    corr_thresh = st.slider("Correlation threshold", 0.7, 0.99, 0.95)

    mode = st.radio(
            "Training mode",
            ["Random Search", "Use fixed parameters"]
        )

    if mode == "Random Search":
        selected_models = st.multiselect("Select the models to use in the randomised search CV", ["ridge", "lasso", "elasticnet", "rf", "gbr"])
        n_iter = st.number_input("Number of randomised search iterations", min_value=1, max_value=1000, value=30, step=1)
    else:
        selected_models = None
        n_iter = None

    if mode == "Use fixed parameters":
        model_choice = st.selectbox(
            "Select model",
            ["ridge", "lasso", "elasticnet", "rf", "gbr"]
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
    
    shap_toggle = st.toggle("Perform SHAP Analysis")
    ci_toggle = st.toggle("Calculate confidence interval (takes a long time)")

    aim = st.text_area("Experiment aim")

    if st.button("Train model"):

        X, y, var_sel, dropped_corr, feature_names = preprocess_data(
            df,
            target,
            drop_cols,
            var_thresh,
            corr_thresh
        )
        
        groups = df[group] if group != "None" else None
        stratify_col = df[stratify] if stratify != "None" else None

        if target in drop_cols:
            st.error("Target column cannot be dropped.")
            st.stop()
        
        if group in drop_cols:
            st.warning("Group column is excluded from features but still used for splitting.")
        
        if stratify in drop_cols:
            st.warning("Stratify column is excluded from features but still used for splitting.")

        X_train, X_test, y_train, y_test = split_data(
            X, y,
            groups,
            stratify_col,
            split_ratio,
            seed
        )

        if mode == "Random Search":
            model, search = train_model(
                X_train,
                y_train,
                seed,
                selected_models=selected_models,
                n_iter=n_iter
            )
            best_params_clean = clean_params(search.best_params_)
        
        else:
            model = fit_model_with_params(
                X_train,
                y_train,
                seed,
                model_key=model_choice,
                best_params=best_params
            )
            search = None
            best_params_clean = best_params

        results = evaluate(model, X_train, X_test, y_train, y_test)

        st.write(f"Training set R2: {results['r2_train']}")
        st.write(f"Test set R2: {results['r2_test']}")

        # Plots
        fig1 = parity_plot(y_test, results["y_pred_test"])
        st.pyplot(fig1)

        X_scaled = model.named_steps["scaler"].transform(X_test)
        h, h_star, std_res, flags = applicability_domain(X_scaled, y_test, results["y_pred_test"])
        fig2 = williams_plot(h, std_res, h_star)
        st.pyplot(fig2)

        # SHAP
        if shap_toggle:
            X_sample = X_test.sample(min(100, len(X_test)), random_state=seed)
            
            shap_values, X_sample_named = compute_shap(
                model,
                X_sample,
                feature_names
            )
            
            shap_fig = shap_summary_plot(shap_values, X_sample_named)
            st.pyplot(shap_fig)

        # CI
        if ci_toggle:
            lower, upper = bootstrap_ci(model, X_train, y_train, X_test)
            mean_ci_width = np.mean(ci_width)
            rel_uncertainty = ci_width / np.abs(results["y_pred_test"])
            st.write("Mean CI width:", mean_ci_width)
            st.write("Relative uncertainty:", rel_uncertainty)
        else:
            mean_ci_width = None
            rel_uncertainty = None

        run_dir = create_run_dir()
        
        model_path = f"{run_dir}/model.joblib"
        log_path = f"{run_dir}/log.json"
        parity_path = f"{run_dir}/parity.png"
        williams_path = f"{run_dir}/williams.png"
        cv_path = f"{run_dir}/cv_results.csv"

        if shap_toggle:
            shap_path = f"{run_dir}/shap.png"
        if search is not None:
            cv_df = pd.DataFrame(search.cv_results_)
            cv_df.to_csv(cv_path, index=False)

        save_model(model_path, {"model": model})
        save_log(log_path, {
            "aim": aim,
            "dataset_filename": file.name if file else None,
            "r2_train": results["r2_train"],
            "r2_test": results["r2_test"],
            "n_features": X.shape[1],
            "filters": filters ,
            "models_tested": selected_models,
            "no_of_iterations": n_iter,
            "best_model_type": str(model.named_steps["model"]),
            "best_params": best_params_clean,
            "Mean CI width": mean_ci_width,
            "Relative uncertainty": rel_uncertainty,
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
