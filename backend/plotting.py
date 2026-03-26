import matplotlib.pyplot as plt
import numpy as np

def set_style():
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "axes.grid": True
    })

def parity_plot(y_true, y_pred):
    set_style()
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, c="blue", alpha=0.7)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Observed RT")
    ax.set_ylabel("Predicted RT")
    ax.set_title("Parity Plot")
    return fig

def williams_plot(leverage, std_res, h_star):
    set_style()
    fig, ax = plt.subplots()

    colors = ["green" if (h <= h_star and abs(r) <= 3) else "red"
              for h, r in zip(leverage, std_res)]

    ax.scatter(leverage, std_res, c=colors)
    ax.axhline(3, linestyle="--", color="black")
    ax.axhline(-3, linestyle="--", color="black")
    ax.axvline(h_star, linestyle="--", color="black")

    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardised Residuals")
    ax.set_title("Williams Plot")

    return fig
