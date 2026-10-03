import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.getcwd())

from evaluation.evaluate import peak_nonpeak_rmse  # noqa: E402
from preprocessing.preprocess import (  # noqa: E402
    get_train_validation_rows,
    load_data,
    make_lstm_samples,
    make_supervised_samples,
    scale_full_series_using_train_max,
    select_training_samples,
    select_validation_samples,
)

# Final settings, copied from main.py
ANN = dict(hidden=4, lr=0.001, rain_lag=5, runoff_lag=5)
LSTM = dict(hidden=4, lr=0.0001, seq=7)
EPOCHS, BATCH, PEAK = 200, 32, 1500

os.makedirs("results", exist_ok=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_baselines(scaled, runoff_max, train_rows):
    """Baselines use exactly the same samples/split as the final ANN."""
    X, y, d = make_supervised_samples(scaled, ANN["rain_lag"], ANN["runoff_lag"])
    Xt, yt, _ = select_training_samples(X, y, d)
    Xv, yv, _ = select_validation_samples(X, y, d)
    r = ANN["rain_lag"]  # features = [rain lags..., runoff lags...]

    preds = {
        "Train-mean (always predict average)": np.full_like(yv, yt.mean()),
        "Persistence (Q(t) = Q(t-1))": Xv[:, r],
        "Linear regression (rain + runoff lags)": LinearRegression().fit(Xt, yt).predict(Xv),
        "Linear regression (runoff lags only)": LinearRegression().fit(Xt[:, r:], yt).predict(Xv[:, r:]),
    }
    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "overall_rmse": rmse(yv * runoff_max, p * runoff_max)})
    out = pd.DataFrame(rows)
    out.to_csv("results/baseline_metrics.csv", index=False)
    print("\n=== Baselines (validation 2008-2012) ===")
    print(out.round(3).to_string(index=False))
    return out


def run_seeds(scaled, runoff_max, n_seeds):
    import tensorflow as tf
    from models.ann_model import build_ann
    from models.lstm_model import build_lstm

    Xa, ya, da = make_supervised_samples(scaled, ANN["rain_lag"], ANN["runoff_lag"])
    Xat, yat, _ = select_training_samples(Xa, ya, da)
    Xav, yav, _ = select_validation_samples(Xa, ya, da)

    Xl, yl, dl = make_lstm_samples(scaled, LSTM["seq"])
    Xlt, ylt, _ = select_training_samples(Xl, yl, dl)
    Xlv, ylv, _ = select_validation_samples(Xl, yl, dl)

    rows = []
    for seed in range(n_seeds):
        tf.keras.utils.set_random_seed(seed)
        m = build_ann(Xat.shape[1], ANN["hidden"], ANN["lr"])
        m.fit(Xat, yat, epochs=EPOCHS, batch_size=BATCH, verbose=0)
        p = m.predict(Xav, verbose=0).ravel()
        rows.append({"model": "ANN", "seed": seed, "overall_rmse": rmse(yav * runoff_max, p * runoff_max)})

        tf.keras.utils.set_random_seed(seed)
        m = build_lstm(LSTM["seq"], 2, LSTM["hidden"], LSTM["lr"])
        m.fit(Xlt, ylt, epochs=EPOCHS, batch_size=BATCH, verbose=0)
        p = m.predict(Xlv, verbose=0).ravel()
        rows.append({"model": "LSTM", "seed": seed, "overall_rmse": rmse(ylv * runoff_max, p * runoff_max)})
        print(f"seed {seed}: ANN {rows[-2]['overall_rmse']:.2f} | LSTM {rows[-1]['overall_rmse']:.2f}")

    res = pd.DataFrame(rows)
    res.to_csv("results/multi_seed_metrics.csv", index=False)
    summ = res.groupby("model")["overall_rmse"].agg(["mean", "std", "min", "max"]).round(3)
    print(f"\n=== {n_seeds} seeds, mean +/- std ===")
    print(summ)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-nn", action="store_true", help="baselines only")
    ap.add_argument("--data", default="data/streamflow.csv")
    a = ap.parse_args()

    df = load_data(a.data)
    train_rows, _ = get_train_validation_rows(df)
    scaled, _, runoff_max = scale_full_series_using_train_max(df, train_rows)

    run_baselines(scaled, runoff_max, train_rows)
    if not a.no_nn:
        run_seeds(scaled, runoff_max, a.seeds)


if __name__ == "__main__":
    main()