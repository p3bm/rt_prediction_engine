import shap
import numpy as np
import matplotlib.pyplot as plt

def compute_shap(model, X_sample):

    try:
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)
    except:
        # fallback for tree models
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values = explainer.shap_values(X_sample)

    return shap_values

def shap_summary_plot(shap_values, X_sample):
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    return fig
