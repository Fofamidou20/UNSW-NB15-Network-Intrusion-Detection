import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_comparison(table, path):
    ax = table.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot.bar(figsize=(10, 6), ylim=(0, 1))
    ax.set_ylabel("Score")
    ax.set_title("UNSW-NB15 independent test performance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
