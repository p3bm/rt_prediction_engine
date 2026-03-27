import streamlit as st
import pandas as pd
import tempfile
import zipfile

from backend.data import preprocess_data
from backend.model import split_data, train_model
from backend.evaluation import evaluate, applicability_domain, bootstrap_ci
from backend.plotting import parity_plot, williams_plot
from backend.shap_analysis import compute_shap, shap_summary_plot
from backend.utils import set_seed, save_model, save_log

st.title("RT Prediction Platform (Sklearn AutoML)")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    target = st.selectbox("Target", df.columns)
    group = st.selectbox("Group column", ["None"] + list(df.columns))
    stratify = st.selectbox("Stratify by column (optional)", ["None"] + list(df.columns))
    split_ratio = st.number_input("Split ratio", min_value=0.10, max_value=1.00, value=0.2, step=0.05)

    seed = st.number_input("Random seed", value=42)
    set_seed(seed)

    var_thresh = st.slider("Variance threshold", 0.0, 0.2, 0.01)
    corr_thresh = st.slider("Correlation threshold", 0.7, 0.99, 0.95)

    aim = st.text_area("Experiment aim")

    if st.button("Train model"):

        X, y, var_sel, dropped = preprocess_data(df, target, var_thresh, corr_thresh)
        
        groups = df[group] if group != "None" else None
        stratify_col = df[stratify] if stratify != "None" else None

        X_train, X_test, y_train, y_test = split_data(
            X, y,
            groups,
            stratify_col,
            split_ratio,
            seed
        )

        model, search = train_model(X_train, y_train, seed)

        results = evaluate(model, X_train, X_test, y_train, y_test)

        st.write(results)

        # Plots
        fig1 = parity_plot(y_test, results["y_pred_test"])
        st.pyplot(fig1)

        h, h_star, std_res, flags = applicability_domain(X_test, y_test, results["y_pred_test"])
        fig2 = williams_plot(h, std_res, h_star)
        st.pyplot(fig2)

        # SHAP
        X_sample = X_test.sample(min(100, len(X_test)), random_state=seed)
        shap_values = compute_shap(model, X_sample)
        shap_fig = shap_summary_plot(shap_values, X_sample)
        st.pyplot(shap_fig)

        # CI
        lower, upper = bootstrap_ci(model, X_train, y_train, X_test)

        st.write("CI (first 5):", list(zip(lower[:5], upper[:5])))

        from backend.utils import create_run_dir

        run_dir = create_run_dir()
        
        model_path = f"{run_dir}/model.joblib"
        log_path = f"{run_dir}/log.json"
        parity_path = f"{run_dir}/parity.png"
        williams_path = f"{run_dir}/williams.png"
        shap_path = f"{run_dir}/shap.png"
        cv_path = f"{run_dir}/cv_results.csv"

        save_model(model_path, {"model": model})
        save_log(log_path, {
            "aim": aim,
            "r2_train": results["r2_train"],
            "r2_test": results["r2_test"],
            "overfit": results["overfit"],
            "n_features": X.shape[1],
            "dropped_corr": dropped,
            "stratified_by": stratify if stratify != "None" else None,
            "grouped_by": group if group != "None" else None,
            "split_ratio": split_ratio
            "seed": seed
        })

        fig1.savefig(parity_path, dpi=300)
        fig2.savefig(williams_path, dpi=300)
        shap_fig.savefig(shap_path, dpi=300)

        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(cv_path, index=False)

        zip_path = f"{run_dir}.zip"

        with zipfile.ZipFile(zip_path, "w") as z:
            for file in os.listdir(run_dir):
                z.write(os.path.join(run_dir, file), file)
        
        with open(zip_path, "rb") as f:
            st.download_button("Download results", f, "results.zip")
