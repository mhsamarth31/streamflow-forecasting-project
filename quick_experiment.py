import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from preprocessing.preprocess import (
    load_data,
    get_train_validation_rows,
    scale_full_series_using_train_max,
    make_supervised_samples,
    make_lstm_samples,
    select_training_samples,
    select_validation_samples,
)


DATA_PATH = Path("data/streamflow.csv")
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "hyperparameter_results.csv"

EPOCHS = 200
BATCH_SIZE = 32

# Fixed settings from the paper/baseline.
ANN_LEARNING_RATE = 0.009
ANN_RAINFALL_LAG = 3
ANN_RUNOFF_LAG = 4

LSTM_LEARNING_RATE = 0.001
LSTM_SEQUENCE_LENGTH = 5


def save_result(row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame([row])

    if RESULTS_FILE.exists():
        result.to_csv(
            RESULTS_FILE,
            mode="a",
            header=False,
            index=False,
        )
    else:
        result.to_csv(
            RESULTS_FILE,
            index=False,
        )


def build_ann(input_size, hidden_size, learning_rate):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(
                hidden_size,
                activation="sigmoid",
            ),
            tf.keras.layers.Dense(
                1,
                activation="linear",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
    )

    return model


def build_lstm(sequence_length, hidden_size, learning_rate):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(sequence_length, 2)
            ),
            tf.keras.layers.LSTM(hidden_size),
            tf.keras.layers.Dense(
                1,
                activation="linear",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
    )

    return model


def run_ann(
    hidden_size,
    learning_rate,
    rainfall_lag,
    runoff_lag,
):
    print()
    print("========================================")
    print("STARTING ANN EXPERIMENT")
    print("========================================")
    print(f"Hidden size   : {hidden_size}")
    print(f"Learning rate : {learning_rate}")
    print(f"Rainfall lag  : {rainfall_lag}")
    print(f"Runoff lag    : {runoff_lag}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print("========================================")
    print()

    df = load_data(DATA_PATH)

    train_rows, _ = get_train_validation_rows(df)

    scaled_df, _, runoff_max = (
        scale_full_series_using_train_max(
            df,
            train_rows,
        )
    )

    X, y, dates = make_supervised_samples(
        scaled_df,
        rainfall_lag=rainfall_lag,
        runoff_lag=runoff_lag,
    )

    X_train, y_train, _ = select_training_samples(
        X,
        y,
        dates,
    )

    X_valid, y_valid, _ = select_validation_samples(
        X,
        y,
        dates,
    )

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_valid)}")
    print()
    print("Training ANN...")
    print()

    model = build_ann(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        learning_rate=learning_rate,
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, y_valid),
        verbose=1,
    )

    predictions_norm = model.predict(
        X_valid,
        verbose=0,
    ).reshape(-1)

    predictions = predictions_norm * runoff_max
    actual = y_valid * runoff_max

    mse = mean_squared_error(
        actual,
        predictions,
    )

    save_result(
        {
            "model": "ANN",
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "rainfall_lag": rainfall_lag,
            "runoff_lag": runoff_lag,
            "sequence_length": "",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "mse": mse,
        }
    )

    print()
    print("========================================")
    print("ANN EXPERIMENT COMPLETE")
    print("========================================")
    print(f"MSE              : {mse:.6f}")
    print(f"Results saved to : {RESULTS_FILE}")
    print("========================================")
    print()


def run_lstm(
    hidden_size,
    learning_rate,
    sequence_length,
):
    print()
    print("========================================")
    print("STARTING LSTM EXPERIMENT")
    print("========================================")
    print(f"Hidden size     : {hidden_size}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Sequence length : {sequence_length}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print("========================================")
    print()

    df = load_data(DATA_PATH)

    train_rows, _ = get_train_validation_rows(df)

    scaled_df, _, runoff_max = (
        scale_full_series_using_train_max(
            df,
            train_rows,
        )
    )

    X, y, dates = make_lstm_samples(
        scaled_df,
        sequence_length=sequence_length,
    )

    X_train, y_train, _ = select_training_samples(
        X,
        y,
        dates,
    )

    X_valid, y_valid, _ = select_validation_samples(
        X,
        y,
        dates,
    )

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_valid)}")
    print()
    print("Training LSTM...")
    print()

    model = build_lstm(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, y_valid),
        verbose=1,
    )

    predictions_norm = model.predict(
        X_valid,
        verbose=0,
    ).reshape(-1)

    predictions = predictions_norm * runoff_max
    actual = y_valid * runoff_max

    mse = mean_squared_error(
        actual,
        predictions,
    )

    save_result(
        {
            "model": "LSTM",
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "rainfall_lag": "",
            "runoff_lag": "",
            "sequence_length": sequence_length,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "mse": mse,
        }
    )

    print()
    print("========================================")
    print("LSTM EXPERIMENT COMPLETE")
    print("========================================")
    print(f"MSE              : {mse:.6f}")
    print(f"Results saved to : {RESULTS_FILE}")
    print("========================================")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="ANN/LSTM hyperparameter experiment"
    )

    parser.add_argument(
        "--model",
        choices=["ann", "lstm"],
        required=True,
        help="Model to experiment with",
    )

    parser.add_argument(
        "--hidden",
        type=int,
        required=True,
        help="Number of hidden units",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--rain-lag",
        type=int,
        default=None,
        help="Rainfall lag for ANN",
    )

    parser.add_argument(
        "--runoff-lag",
        type=int,
        default=None,
        help="Runoff lag for ANN",
    )

    parser.add_argument(
        "--sequence",
        type=int,
        default=None,
        help="Sequence length for LSTM",
    )

    args = parser.parse_args()

    if args.model == "ann":
        learning_rate = (
            args.lr
            if args.lr is not None
            else ANN_LEARNING_RATE
        )

        rainfall_lag = (
            args.rain_lag
            if args.rain_lag is not None
            else ANN_RAINFALL_LAG
        )

        runoff_lag = (
            args.runoff_lag
            if args.runoff_lag is not None
            else ANN_RUNOFF_LAG
        )

        run_ann(
            hidden_size=args.hidden,
            learning_rate=learning_rate,
            rainfall_lag=rainfall_lag,
            runoff_lag=runoff_lag,
        )

    elif args.model == "lstm":
        learning_rate = (
            args.lr
            if args.lr is not None
            else LSTM_LEARNING_RATE
        )

        sequence_length = (
            args.sequence
            if args.sequence is not None
            else LSTM_SEQUENCE_LENGTH
        )

        run_lstm(
            hidden_size=args.hidden,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
        )


if __name__ == "__main__":
    main()
