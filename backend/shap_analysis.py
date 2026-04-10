import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_shap(model, X_train, sample_size=100, seed=42):
    X_sample = pd.DataFrame(X_train).sample(min(sample_size, len(X_train)), random_state=seed)

    try:
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)
    except:
        explainer = shap.TreeExplainer(model.named_steps["model"].named_steps["model"])
        shap_values = explainer.shap_values(X_sample)

    return shap_values, X_sample

def shap_summary_plot(shap_values, X_sample):
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    return fig
