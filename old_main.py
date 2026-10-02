from pathlib import Path

import pandas as pd

from preprocessing.preprocess import (
    load_data,
    get_train_validation_rows,
    scale_full_series_using_train_max,
    make_supervised_samples,
    make_lstm_samples,
    select_training_samples,
    select_validation_samples,
)
from training.train_ann import train_ann
from training.train_lstm import train_lstm
from evaluation.evaluate import peak_nonpeak_rmse
from visualization.plots import (
    save_actual_vs_predicted,
    save_loss,
    save_model_comparison,
)


DATA_PATH = Path("data/streamflow.csv")
RESULTS_DIR = Path("results")

# Best standard-network configuration reported in the paper
RAINFALL_LAG_ANN = 3
RUNOFF_LAG_ANN = 4
ANN_HIDDEN_SIZE = 12
ANN_LEARNING_RATE = 0.009

# Best LSTM configuration reported in the paper
SEQUENCE_LENGTH_LSTM = 5
LSTM_HIDDEN_SIZE = 12
LSTM_LEARNING_RATE = 0.001

# Paper's peak/non-peak analysis threshold
PEAK_THRESHOLD = 1500

EPOCHS = 200
BATCH_SIZE = 32


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load and clean
    # ---------------------------------------------------------
    print("\n1. Loading dataset...")
    df = load_data(DATA_PATH)

    print(f"Rows after cleaning: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # ---------------------------------------------------------
    # 2. Identify train/validation periods
    # ---------------------------------------------------------
    print("\n2. Identifying training and validation periods...")
    train_rows, valid_rows = get_train_validation_rows(df)

    print(f"Training rows (2003-2007): {len(train_rows)}")
    print(f"Validation rows (2008-2012): {len(valid_rows)}")

    # ---------------------------------------------------------
    # 3. Scale complete series using TRAINING maxima
    # ---------------------------------------------------------
    print("\n3. Scaling complete time series using training maxima...")
    scaled_df, rain_max, runoff_max = scale_full_series_using_train_max(
        df, train_rows
    )

    print(f"Training rainfall max: {rain_max}")
    print(f"Training runoff max: {runoff_max}")

    # ---------------------------------------------------------
    # 4. ANN: create samples from the COMPLETE series
    # ---------------------------------------------------------
    print("\n4. Creating ANN lagged samples...")
    X_ann, y_ann, dates_ann = make_supervised_samples(
        scaled_df,
        rainfall_lag=RAINFALL_LAG_ANN,
        runoff_lag=RUNOFF_LAG_ANN,
    )

    X_train_ann, y_train_ann, _ = select_training_samples(
        X_ann, y_ann, dates_ann
    )
    X_valid_ann, y_valid_ann, valid_dates_ann = select_validation_samples(
        X_ann, y_ann, dates_ann
    )

    print("ANN training shape:", X_train_ann.shape)
    print("ANN validation shape:", X_valid_ann.shape)

    # ---------------------------------------------------------
    # 5. Train ANN
    # ---------------------------------------------------------
    print("\n5. Training ANN...")
    ann_model, ann_history, ann_pred_norm, _ = train_ann(
        X_train_ann,
        y_train_ann,
        X_valid_ann,
        y_valid_ann,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=ANN_HIDDEN_SIZE,
        learning_rate=ANN_LEARNING_RATE,
    )

    ann_pred = ann_pred_norm * runoff_max
    ann_actual = y_valid_ann * runoff_max

    ann_metrics = peak_nonpeak_rmse(
        ann_actual,
        ann_pred,
        threshold=PEAK_THRESHOLD,
    )

    ann_model.save(RESULTS_DIR / "ann_model.keras")

    pd.DataFrame({
        "date": valid_dates_ann,
        "actual_runoff": ann_actual,
        "predicted_runoff": ann_pred,
    }).to_csv(RESULTS_DIR / "ann_predictions.csv", index=False)

    save_actual_vs_predicted(
        valid_dates_ann,
        ann_actual,
        ann_pred,
        RESULTS_DIR / "ann_actual_vs_predicted.png",
        "ANN - Actual vs Predicted Streamflow",
    )

    save_loss(
        ann_history,
        RESULTS_DIR / "ann_training_loss.png",
        "ANN Training and Validation Loss",
    )

    # ---------------------------------------------------------
    # 6. LSTM: create sequences from COMPLETE series
    # ---------------------------------------------------------
    print("\n6. Creating LSTM sequences...")
    X_lstm, y_lstm, dates_lstm = make_lstm_samples(
        scaled_df,
        sequence_length=SEQUENCE_LENGTH_LSTM,
    )

    X_train_lstm, y_train_lstm, _ = select_training_samples(
        X_lstm, y_lstm, dates_lstm
    )
    X_valid_lstm, y_valid_lstm, valid_dates_lstm = select_validation_samples(
        X_lstm, y_lstm, dates_lstm
    )

    print("LSTM training shape:", X_train_lstm.shape)
    print("LSTM validation shape:", X_valid_lstm.shape)

    # ---------------------------------------------------------
    # 7. Train LSTM
    # ---------------------------------------------------------
    print("\n7. Training LSTM...")
    lstm_model, lstm_history, lstm_pred_norm, _ = train_lstm(
        X_train_lstm,
        y_train_lstm,
        X_valid_lstm,
        y_valid_lstm,
        sequence_length=SEQUENCE_LENGTH_LSTM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        learning_rate=LSTM_LEARNING_RATE,
    )

    lstm_pred = lstm_pred_norm * runoff_max
    lstm_actual = y_valid_lstm * runoff_max

    lstm_metrics = peak_nonpeak_rmse(
        lstm_actual,
        lstm_pred,
        threshold=PEAK_THRESHOLD,
    )

    lstm_model.save(RESULTS_DIR / "lstm_model.keras")

    pd.DataFrame({
        "date": valid_dates_lstm,
        "actual_runoff": lstm_actual,
        "predicted_runoff": lstm_pred,
    }).to_csv(RESULTS_DIR / "lstm_predictions.csv", index=False)

    save_actual_vs_predicted(
        valid_dates_lstm,
        lstm_actual,
        lstm_pred,
        RESULTS_DIR / "lstm_actual_vs_predicted.png",
        "LSTM - Actual vs Predicted Streamflow",
    )

    save_loss(
        lstm_history,
        RESULTS_DIR / "lstm_training_loss.png",
        "LSTM Training and Validation Loss",
    )

    save_model_comparison(
        ann_metrics["overall_rmse"],
        lstm_metrics["overall_rmse"],
        RESULTS_DIR / "ann_vs_lstm_rmse.png",
    )

    # ---------------------------------------------------------
    # 8. Save final metrics
    # ---------------------------------------------------------
    print("\n================ FINAL RESULTS ================")

    print("\nANN:")
    for key, value in ann_metrics.items():
        print(f"{key}: {value}")

    print("\nLSTM:")
    for key, value in lstm_metrics.items():
        print(f"{key}: {value}")

    summary = pd.DataFrame([
        {"model": "ANN", **ann_metrics},
        {"model": "LSTM", **lstm_metrics},
    ])

    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    print("\nAll results saved in:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
