import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "./output/mlff_project/runs"
OUT_DIR = "./output/mlff_project/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def config_dirs():
    out = []
    for name in os.listdir(RUNS_DIR):
        path = os.path.join(RUNS_DIR, name)
        if os.path.isdir(path) and name.startswith("lr_"):
            out.append(name)
    return sorted(out)


def load_history(config_name, model_name):
    path = os.path.join(RUNS_DIR, config_name, model_name, "history.json")
    with open(path, "r") as f:
        return json.load(f)


def curve_df(hist, split):
    df = pd.DataFrame(hist[split])
    df["model"] = hist["model"]
    return df


def plot_histories_per_config(config_name, histories):
    plt.figure(figsize=(10, 6))
    for hist in histories:
        dft = curve_df(hist, "train")
        dfv = curve_df(hist, "val")
        plt.plot(dft["epoch"], dft["loss"], label=f"{hist['model']} train")
        plt.plot(dfv["epoch"], dfv["loss"], linestyle="--", label=f"{hist['model']} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train/val loss histories - {config_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{config_name}_loss_histories.png"), dpi=200)
    plt.close()

    # first 20 epochs

    plt.figure(figsize=(10, 6))
    for hist in histories:
        dft = curve_df(hist, "train")
        dfv = curve_df(hist, "val")
        plt.plot(dft["epoch"][0:20], dft["loss"][0:20], label=f"{hist['model']} train")
        plt.plot(dfv["epoch"][0:20], dfv["loss"][0:20], linestyle="--", label=f"{hist['model']} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train/val loss histories (first 20) - {config_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{config_name}_loss_histories_20.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for hist in histories:
        dfv = curve_df(hist, "val")
        plt.plot(dfv["epoch"], dfv["force_mae"], label=f"{hist['model']} val force MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Force MAE")
    plt.title(f"Validation force MAE - {config_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{config_name}_val_force_mae.png"), dpi=200)
    plt.close()


def plot_test_metrics_per_config(config_name, histories):
    names = [h["model"] for h in histories]
    test_loss = [h["test"]["loss"] for h in histories]
    test_force = [h["test"]["force_mae"] for h in histories]
    test_energy = [h["test"]["energy_mae"] for h in histories]
    params = [h["num_parameters"] for h in histories]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].bar(names, test_loss)
    axes[0].set_title("Test total loss")
    axes[1].bar(names, test_force)
    axes[1].set_title("Test force MAE")
    axes[2].bar(names, test_energy)
    axes[2].set_title("Test energy MAE")
    axes[3].bar(names, params)
    axes[3].set_title("Parameters")
    for ax in axes:
        ax.tick_params(axis='x', rotation=20)
    plt.suptitle(config_name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{config_name}_test_metrics.png"), dpi=200)
    plt.close()


def plot_cross_config_summary(all_histories):
    rows = []
    for config_name, histories in all_histories.items():
        for hist in histories:
            rows.append({
                "config": config_name,
                "model": hist["model"],
                "test_loss": hist["test"]["loss"],
                "test_force_mae": hist["test"]["force_mae"],
                "test_energy_mae": hist["test"]["energy_mae"],
                "params": hist["num_parameters"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "all_test_metrics.csv"), index=False)

    for metric in ["test_loss", "test_force_mae", "test_energy_mae"]:
        pivot = df.pivot(index="config", columns="model", values=metric)
        pivot.plot(kind="bar", figsize=(10, 6))
        plt.title(metric.replace('_', ' ').title())
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cross_config_{metric}.png"), dpi=200)
        plt.close()


def main():
    all_histories = {}
    for cfg_name in config_dirs():
        histories = [load_history(cfg_name, m) for m in ["schnet", "schnet_plus", "schnet_plusplus"]]
        all_histories[cfg_name] = histories
        plot_histories_per_config(cfg_name, histories)
        plot_test_metrics_per_config(cfg_name, histories)
    plot_cross_config_summary(all_histories)


if __name__ == "__main__":
    main()