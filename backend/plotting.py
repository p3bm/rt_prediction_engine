import matplotlib.pyplot as plt

def set_style():
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "axes.grid": True
    })

def parity_plot(y_true, y_pred):
    #set_style()
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, color="blue", alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    ax.set_title("Parity Plot")
    ax.set_xlabel("Experimental RRT")
    ax.set_ylabel("Predicted RRT")
    return fig

def williams_plot(h, std_res, h_star):
    #set_style()
    fig, ax = plt.subplots()

    colors = [
        "green" if (h[i] <= h_star and abs(std_res[i]) <= 3) else "red"
        for i in range(len(h))
    ]

    ax.scatter(h, std_res, c=colors, alpha=0.5)
    ax.axvline(h_star, linestyle="--")
    ax.axhline(3, linestyle="--")
    ax.axhline(-3, linestyle="--")

    ax.set_title("Williams Plot")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardised Residual")
    return fig
