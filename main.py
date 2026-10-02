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


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/streamflow.csv")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 200
BATCH_SIZE = 32


# ============================================================
# FINAL ANN PARAMETERS
# ============================================================

ANN_HIDDEN_SIZE = 4
ANN_LEARNING_RATE = 0.001

ANN_RAINFALL_LAG = 5
ANN_RUNOFF_LAG = 5


# ============================================================
# FINAL LSTM PARAMETERS
# ============================================================

LSTM_HIDDEN_SIZE = 4
LSTM_LEARNING_RATE = 0.0001

LSTM_SEQUENCE_LENGTH = 7


# Paper evaluation threshold
PEAK_THRESHOLD = 1500


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("       FINAL STREAMFLOW FORECASTING RUN")
    print("==============================================")
    print()

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    print(f"Total observations: {len(df)}")

    # ========================================================
    # 2. TRAIN / VALIDATION SPLIT
    # ========================================================

    train_rows, validation_rows = get_train_validation_rows(df)

    print(
        f"Training observations   : {len(train_rows)}"
    )

    print(
        f"Validation observations : {len(validation_rows)}"
    )

    # ========================================================
    # 3. SCALE DATA
    # ========================================================

    print()
    print("Scaling data...")

    (
        scaled_df,
        rainfall_max,
        runoff_max,
    ) = scale_full_series_using_train_max(
        df,
        train_rows,
    )

    print(f"Rainfall scaling value : {rainfall_max}")
    print(f"Runoff scaling value   : {runoff_max}")

    # ========================================================
    #                     ANN
    # ========================================================

    print()
    print("==============================================")
    print("                 TRAINING ANN")
    print("==============================================")
    print()

    # --------------------------------------------------------
    # 4. CREATE ANN SAMPLES
    # --------------------------------------------------------

    X_ann, y_ann, dates_ann = make_supervised_samples(
        scaled_df,
        rainfall_lag=ANN_RAINFALL_LAG,
        runoff_lag=ANN_RUNOFF_LAG,
    )

    X_ann_train, y_ann_train, _ = select_training_samples(
        X_ann,
        y_ann,
        dates_ann,
    )

    X_ann_valid, y_ann_valid, ann_valid_dates = (
        select_validation_samples(
            X_ann,
            y_ann,
            dates_ann,
        )
    )

    print(f"ANN training samples   : {len(X_ann_train)}")
    print(f"ANN validation samples : {len(X_ann_valid)}")

    # --------------------------------------------------------
    # 5. TRAIN ANN
    # --------------------------------------------------------

    (
        ann_model,
        ann_history,
        ann_predictions_norm,
        _,
    ) = train_ann(
        X_ann_train,
        y_ann_train,
        X_ann_valid,
        y_ann_valid,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=ANN_HIDDEN_SIZE,
        learning_rate=ANN_LEARNING_RATE,
    )

    # Convert normalized values back to original units
    ann_predictions = ann_predictions_norm * runoff_max
    ann_actual = y_ann_valid * runoff_max

    # --------------------------------------------------------
    # 6. SAVE ANN MODEL
    # --------------------------------------------------------

    ann_model_path = RESULTS_DIR / "final_ann_model.keras"

    ann_model.save(ann_model_path)

    # --------------------------------------------------------
    # 7. ANN EVALUATION
    # --------------------------------------------------------

    ann_metrics = peak_nonpeak_rmse(
        ann_actual,
        ann_predictions,
        PEAK_THRESHOLD,
    )

    print()
    print("==============================================")
    print("                FINAL ANN RESULTS")
    print("==============================================")

    print(
        f"Overall RMSE : "
        f"{ann_metrics['overall_rmse']:.6f}"
    )

    print(
        f"Peak RMSE    : "
        f"{ann_metrics['peak_rmse']}"
    )

    print(
        f"Non-peak RMSE: "
        f"{ann_metrics['nonpeak_rmse']}"
    )

    print(
        f"Peak samples : "
        f"{ann_metrics['peak_count']}"
    )

    print(
        f"Non-peak samples: "
        f"{ann_metrics['nonpeak_count']}"
    )

    # --------------------------------------------------------
    # 8. SAVE ANN PREDICTIONS
    # --------------------------------------------------------

    ann_predictions_df = pd.DataFrame(
        {
            "Date": ann_valid_dates,
            "Actual": ann_actual,
            "Predicted": ann_predictions,
        }
    )

    ann_predictions_df.to_csv(
        RESULTS_DIR / "final_ann_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 9. ANN PLOTS
    # --------------------------------------------------------

    save_actual_vs_predicted(
        ann_valid_dates,
        ann_actual,
        ann_predictions,
        RESULTS_DIR / "final_ann_actual_vs_predicted.png",
        "Final ANN: Actual vs Predicted Streamflow",
    )

    save_loss(
        ann_history,
        RESULTS_DIR / "final_ann_training_loss.png",
        "Final ANN Training Loss",
    )

    # ========================================================
    #                     LSTM
    # ========================================================

    print()
    print("==============================================")
    print("                TRAINING LSTM")
    print("==============================================")
    print()

    # --------------------------------------------------------
    # 10. CREATE LSTM SAMPLES
    # --------------------------------------------------------

    X_lstm, y_lstm, dates_lstm = make_lstm_samples(
        scaled_df,
        sequence_length=LSTM_SEQUENCE_LENGTH,
    )

    X_lstm_train, y_lstm_train, _ = (
        select_training_samples(
            X_lstm,
            y_lstm,
            dates_lstm,
        )
    )

    X_lstm_valid, y_lstm_valid, lstm_valid_dates = (
        select_validation_samples(
            X_lstm,
            y_lstm,
            dates_lstm,
        )
    )

    print(
        f"LSTM training samples   : "
        f"{len(X_lstm_train)}"
    )

    print(
        f"LSTM validation samples : "
        f"{len(X_lstm_valid)}"
    )

    # --------------------------------------------------------
    # 11. TRAIN LSTM
    # --------------------------------------------------------

    (
        lstm_model,
        lstm_history,
        lstm_predictions_norm,
        _,
    ) = train_lstm(
        X_lstm_train,
        y_lstm_train,
        X_lstm_valid,
        y_lstm_valid,
        sequence_length=LSTM_SEQUENCE_LENGTH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        learning_rate=LSTM_LEARNING_RATE,
    )

    # Convert back to original units
    lstm_predictions = (
        lstm_predictions_norm * runoff_max
    )

    lstm_actual = y_lstm_valid * runoff_max

    # --------------------------------------------------------
    # 12. SAVE LSTM MODEL
    # --------------------------------------------------------

    lstm_model_path = RESULTS_DIR / "final_lstm_model.keras"

    lstm_model.save(lstm_model_path)

    # --------------------------------------------------------
    # 13. LSTM EVALUATION
    # --------------------------------------------------------

    lstm_metrics = peak_nonpeak_rmse(
        lstm_actual,
        lstm_predictions,
        PEAK_THRESHOLD,
    )

    print()
    print("==============================================")
    print("               FINAL LSTM RESULTS")
    print("==============================================")

    print(
        f"Overall RMSE : "
        f"{lstm_metrics['overall_rmse']:.6f}"
    )

    print(
        f"Peak RMSE    : "
        f"{lstm_metrics['peak_rmse']}"
    )

    print(
        f"Non-peak RMSE: "
        f"{lstm_metrics['nonpeak_rmse']}"
    )

    print(
        f"Peak samples : "
        f"{lstm_metrics['peak_count']}"
    )

    print(
        f"Non-peak samples: "
        f"{lstm_metrics['nonpeak_count']}"
    )

    # --------------------------------------------------------
    # 14. SAVE LSTM PREDICTIONS
    # --------------------------------------------------------

    lstm_predictions_df = pd.DataFrame(
        {
            "Date": lstm_valid_dates,
            "Actual": lstm_actual,
            "Predicted": lstm_predictions,
        }
    )

    lstm_predictions_df.to_csv(
        RESULTS_DIR / "final_lstm_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 15. LSTM PLOTS
    # --------------------------------------------------------

    save_actual_vs_predicted(
        lstm_valid_dates,
        lstm_actual,
        lstm_predictions,
        RESULTS_DIR / "final_lstm_actual_vs_predicted.png",
        "Final LSTM: Actual vs Predicted Streamflow",
    )

    save_loss(
        lstm_history,
        RESULTS_DIR / "final_lstm_training_loss.png",
        "Final LSTM Training Loss",
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    ann_rmse = ann_metrics["overall_rmse"]
    lstm_rmse = lstm_metrics["overall_rmse"]

    print()
    print("==============================================")
    print("             FINAL MODEL COMPARISON")
    print("==============================================")

    print(f"ANN  RMSE : {ann_rmse:.6f}")
    print(f"LSTM RMSE : {lstm_rmse:.6f}")

    # --------------------------------------------------------
    # 16. SAVE FINAL METRICS
    # --------------------------------------------------------

    final_metrics = pd.DataFrame(
        [
            {
                "model": "ANN",
                "hidden_size": ANN_HIDDEN_SIZE,
                "learning_rate": ANN_LEARNING_RATE,
                "rainfall_lag": ANN_RAINFALL_LAG,
                "runoff_lag": ANN_RUNOFF_LAG,
                "sequence_length": "",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "overall_rmse": ann_metrics[
                    "overall_rmse"
                ],
                "peak_rmse": ann_metrics[
                    "peak_rmse"
                ],
                "nonpeak_rmse": ann_metrics[
                    "nonpeak_rmse"
                ],
                "peak_count": ann_metrics[
                    "peak_count"
                ],
                "nonpeak_count": ann_metrics[
                    "nonpeak_count"
                ],
            },
            {
                "model": "LSTM",
                "hidden_size": LSTM_HIDDEN_SIZE,
                "learning_rate": LSTM_LEARNING_RATE,
                "rainfall_lag": "",
                "runoff_lag": "",
                "sequence_length": LSTM_SEQUENCE_LENGTH,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "overall_rmse": lstm_metrics[
                    "overall_rmse"
                ],
                "peak_rmse": lstm_metrics[
                    "peak_rmse"
                ],
                "nonpeak_rmse": lstm_metrics[
                    "nonpeak_rmse"
                ],
                "peak_count": lstm_metrics[
                    "peak_count"
                ],
                "nonpeak_count": lstm_metrics[
                    "nonpeak_count"
                ],
            },
        ]
    )

    final_metrics.to_csv(
        RESULTS_DIR / "final_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 17. ANN VS LSTM GRAPH
    # --------------------------------------------------------

    save_model_comparison(
        ann_rmse,
        lstm_rmse,
        RESULTS_DIR / "final_ann_vs_lstm_rmse.png",
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("==============================================")
    print("             FINAL RUN COMPLETE")
    print("==============================================")
    print()
    print("Results saved in:")
    print("results/")
    print()
    print("Important final files:")
    print("  final_metrics.csv")
    print("  final_ann_predictions.csv")
    print("  final_lstm_predictions.csv")
    print("  final_ann_model.keras")
    print("  final_lstm_model.keras")
    print("  final_ann_actual_vs_predicted.png")
    print("  final_lstm_actual_vs_predicted.png")
    print("  final_ann_training_loss.png")
    print("  final_lstm_training_loss.png")
    print("  final_ann_vs_lstm_rmse.png")
    print()


if __name__ == "__main__":
    main()