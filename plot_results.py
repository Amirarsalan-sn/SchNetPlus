import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "./output/mlff_project/runs"
OUT_DIR = "./output/mlff_project/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_history(model_name):
    path = os.path.join(RUNS_DIR, model_name, "history.json")
    with open(path, "r") as f:
        return json.load(f)


def curve_df(hist, split):
    df = pd.DataFrame(hist[split])
    df["model"] = hist["model"]
    return df


def plot_loss_histories(histories):
    plt.figure(figsize=(10, 6))
    for hist in histories:
        df = curve_df(hist, "train")
        plt.plot(df["epoch"], df["loss"], label=f"{hist['model']} train")
        dfv = curve_df(hist, "val")
        plt.plot(dfv["epoch"], dfv["loss"], linestyle="--", label=f"{hist['model']} val")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Training and validation loss histories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_histories.png"), dpi=200)
    plt.close()


def plot_force_histories(histories):
    plt.figure(figsize=(10, 6))
    for hist in histories:
        df = curve_df(hist, "val")
        plt.plot(df["epoch"], df["force_mae"], label=f"{hist['model']} val force MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Force MAE")
    plt.title("Validation force MAE across models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "force_mae_histories.png"), dpi=200)
    plt.close()


def plot_param_bar(histories):
    names = [h["model"] for h in histories]
    params = [h["num_parameters"] for h in histories]
    plt.figure(figsize=(8, 5))
    plt.bar(names, params)
    plt.ylabel("Trainable parameters")
    plt.title("Model parameter counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "parameter_counts.png"), dpi=200)
    plt.close()


def plot_test_bar(histories):
    names = [h["model"] for h in histories]
    force_mae = [h["test"]["force_mae"] for h in histories]
    energy_mae = [h["test"]["energy_mae"] for h in histories]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, force_mae)
    axes[0].set_title("Test force MAE")
    axes[1].bar(names, energy_mae)
    axes[1].set_title("Test energy MAE")
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "test_metrics.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    model_names = ["schnet", "schnet_plus", "schnet_plusplus"]
    histories = [load_history(m) for m in model_names]
    plot_loss_histories(histories)
    plot_force_histories(histories)
    plot_param_bar(histories)
    plot_test_bar(histories)
