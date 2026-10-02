from pathlib import Path

import matplotlib.pyplot as plt


def save_actual_vs_predicted(dates, actual, predicted, path, title):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Runoff")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_loss(history, path, title):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_model_comparison(ann_rmse, lstm_rmse, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.bar(["ANN", "LSTM"], [ann_rmse, lstm_rmse])
    plt.ylabel("RMSE")
    plt.title("ANN vs LSTM - Overall RMSE")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
