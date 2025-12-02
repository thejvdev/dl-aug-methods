import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from kornia.geometry import resize
from utils.load_model import load_model


PLOT_DIR = "plots"


def plot_train(method_name):
    method = method_name.lower()

    df = pd.read_csv(f"history/{method}_history.csv")

    df_train_acc = df[df["tag"] == "train/acc"]
    df_train_loss = df[df["tag"] == "train/loss"]

    df_val_acc = df[df["tag"] == "val/acc"]
    df_val_loss = df[df["tag"] == "val/loss"]

    # Visualization
    colors = plt.cm.tab10(range(5))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(method_name)

    # Train accuracy
    axes[0, 0].set_title("Train Accuracy")

    for i in range(5):
        axes[0, 0].plot(
            df_train_acc["step"].iloc[i::5],
            df_train_acc["value"].iloc[i::5],
            c=colors[i],
            label=f"Model {i + 1}",
        )

    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Train loss
    axes[0, 1].set_title("Train Loss")

    for i in range(5):
        axes[0, 1].plot(
            df_train_loss["step"].iloc[i::5],
            df_train_loss["value"].iloc[i::5],
            c=colors[i],
            label=f"Model {i + 1}",
        )

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Val accuracy
    axes[1, 0].set_title("Val Accuracy")

    for i in range(5):
        axes[1, 0].plot(
            df_val_acc["step"].iloc[i::5],
            df_val_acc["value"].iloc[i::5],
            c=colors[i],
            label=f"Model {i + 1}",
        )

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Val loss
    axes[1, 1].set_title("Val Loss")

    for i in range(5):
        axes[1, 1].plot(
            df_val_loss["step"].iloc[i::5],
            df_val_loss["value"].iloc[i::5],
            c=colors[i],
            label=f"Model {i + 1}",
        )

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    plt.savefig(f"{PLOT_DIR}/{method}/train.png")
    plt.close()


def predict(model, test_loader, device="cpu"):
    model.eval()

    y_pred = []

    with torch.inference_mode():
        for X, y in test_loader:
            X = resize(X.to(device), (224, 224))
            y = y.to(device)
            logits = model(X)
            y_pred.append(logits.argmax(1))

    return torch.cat(y_pred)


def plot_eval(method_name):
    method = method_name.lower()

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Device:", device)

    test_ds = CIFAR10("data", train=False, transform=T.ToTensor(), download=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    y = test_ds.targets

    print(f"Testing {method_name} models...")

    n_models = 5  # Param

    for i in range(n_models):
        print(f"Model [{i + 1}/{n_models}]")

        model, _ = load_model(
            f"models/{method}/best_model_fold{i + 1}.pt", device=device
        )

        y_pred = predict(model, test_loader, device)
        y_pred = y_pred.cpu().numpy()

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=test_ds.classes)

        text = f"Accuracy: {accuracy * 100:.2f}%\n\nClassification Report:\n{report}"

        plt.figure(figsize=(6, 5))
        plt.title(f"Model {i + 1}")

        plt.text(0.01, 0.99, text, ha="left", va="top", fontsize=10, family="monospace")

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{PLOT_DIR}/{method}/eval_metrics{i + 1}.png",
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()

        # Confusion matrix
        plt.figure(figsize=(10, 8))
        plt.title(f"Model {i + 1}")

        sns.heatmap(
            confusion_matrix(y, y_pred),
            annot=True,
            fmt="d",
            cmap="crest",
            xticklabels=test_ds.classes,
            yticklabels=test_ds.classes,
        )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{method}/eval_conf{i + 1}.png")
        plt.close()


def main():
    method_name = "MixUp"

    os.makedirs(f"{PLOT_DIR}/{method_name.lower()}", exist_ok=True)
    plot_train(method_name)
    plot_eval(method_name)


if __name__ == "__main__":
    main()
