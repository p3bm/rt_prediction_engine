import streamlit as st
import pandas as pd
import tempfile
import zipfile

from backend.data import preprocess_data
from backend.model import split_data, train_model
from backend.evaluation import evaluate, applicability_domain, bootstrap_ci
from backend.plotting import parity_plot, williams_plot
from backend.utils import set_seed, save_model, save_log

st.title("AutoML RT Prediction Platform")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    target = st.selectbox("Target", df.columns)
    group = st.selectbox("Group (optional)", ["None"] + list(df.columns))

    seed = st.number_input("Random Seed", value=42)
    set_seed(seed)

    var_thresh = st.slider("Variance Threshold", 0.0, 0.2, 0.01)
    corr_thresh = st.slider("Correlation Threshold", 0.7, 0.99, 0.95)

    aim = st.text_area("Experiment Aim")

    if st.button("Run AutoML"):

        X, y, var_sel, dropped = preprocess_data(df, target, var_thresh, corr_thresh)

        groups = df[group] if group != "None" else None

        X_train, X_test, y_train, y_test = split_data(X, y, groups, 0.2, seed)

        model, scaler = train_model(X_train, y_train, 300, 30, seed)

        results = evaluate(model, scaler, X_train, X_test, y_train, y_test)

        st.write(results)

        # Applicability domain
        h, h_star, std_res, flags = applicability_domain(
            X_test, y_test, results["y_pred_test"]
        )

        # Plots
        fig1 = parity_plot(y_test, results["y_pred_test"])
        fig2 = williams_plot(h, std_res, h_star)

        st.pyplot(fig1)
        st.pyplot(fig2)

        # Confidence intervals
        lower, upper = bootstrap_ci(model, scaler, X_train, y_train, X_test)

        st.write("CI example:", list(zip(lower[:5], upper[:5])))

        # Save outputs
        tmpdir = tempfile.mkdtemp()

        model_path = f"{tmpdir}/model.joblib"
        log_path = f"{tmpdir}/log.json"

        save_model(model_path, {
            "model": model,
            "scaler": scaler,
            "var_selector": var_sel,
            "dropped_corr": dropped
        })

        save_log(log_path, {
            "aim": aim,
            "r2_train": results["r2_train"],
            "r2_test": results["r2_test"],
            "overfit": results["overfit"]
        })

        zip_path = f"{tmpdir}/results.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(model_path, "model.joblib")
            z.write(log_path, "log.json")

        with open(zip_path, "rb") as f:
            st.download_button("Download Results", f, "results.zip")
