import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"date", "rainfall", "runoff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    df["runoff"] = pd.to_numeric(df["runoff"], errors="coerce")

    df = df.dropna(subset=["date", "rainfall", "runoff"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    return df


def get_train_validation_rows(df: pd.DataFrame):
    """Returns the original chronological rows for the paper-style year split."""
    train_mask = (df["date"].dt.year >= 2003) & (df["date"].dt.year <= 2007)
    valid_mask = (df["date"].dt.year >= 2008) & (df["date"].dt.year <= 2012)

    train = df.loc[train_mask].copy()
    valid = df.loc[valid_mask].copy()

    if train.empty or valid.empty:
        raise ValueError(
            "The dataset does not contain both 2003-2007 and 2008-2012. "
            "Update the year split for your selected dataset."
        )

    return train, valid


def scale_full_series_using_train_max(df, train_df):
    """
    Scale the complete chronological series using constants calculated
    ONLY from the training period.

    This lets validation samples use historical observations immediately
    before 2008 (e.g. late 2007) without leaking validation values into
    the scaling constants.
    """
    rain_max = train_df["rainfall"].max()
    runoff_max = train_df["runoff"].max()

    if rain_max == 0 or runoff_max == 0:
        raise ValueError("Maximum rainfall/runoff cannot be zero.")

    scaled = df.copy()
    scaled["rainfall"] = scaled["rainfall"] / rain_max
    scaled["runoff"] = scaled["runoff"] / runoff_max

    return scaled, rain_max, runoff_max


def make_supervised_samples(
    df: pd.DataFrame,
    rainfall_lag: int = 3,
    runoff_lag: int = 4,
):
    """
    For target time t:

      rainfall[t-1 ... t-rainfall_lag]
      runoff[t-1 ... t-runoff_lag]

    Target:
      runoff[t]

    Samples are created from the COMPLETE chronological series first.
    The caller then selects train/validation samples using target dates.
    This is important for time-series validation because the first
    validation prediction can legitimately use late training-period history.
    """
    max_lag = max(rainfall_lag, runoff_lag)

    X, y, dates = [], [], []

    rainfall = df["rainfall"].to_numpy(dtype=float)
    runoff = df["runoff"].to_numpy(dtype=float)
    date_values = df["date"].to_numpy()

    for t in range(max_lag, len(df)):
        features = []

        for lag in range(1, rainfall_lag + 1):
            features.append(rainfall[t - lag])

        for lag in range(1, runoff_lag + 1):
            features.append(runoff[t - lag])

        X.append(features)
        y.append(runoff[t])
        dates.append(date_values[t])

    return np.asarray(X), np.asarray(y), pd.to_datetime(dates)


def make_lstm_samples(df: pd.DataFrame, sequence_length: int = 5):
    """
    Creates sequences of [rainfall, runoff] for the previous
    sequence_length timesteps and predicts runoff at time t.
    """
    X, y, dates = [], [], []

    rainfall = df["rainfall"].to_numpy(dtype=float)
    runoff = df["runoff"].to_numpy(dtype=float)
    date_values = df["date"].to_numpy()

    for t in range(sequence_length, len(df)):
        sequence = np.column_stack(
            [
                rainfall[t - sequence_length:t],
                runoff[t - sequence_length:t],
            ]
        )

        X.append(sequence)
        y.append(runoff[t])
        dates.append(date_values[t])

    return np.asarray(X), np.asarray(y), pd.to_datetime(dates)


def select_training_samples(X, y, dates):
    years = pd.DatetimeIndex(dates).year
    mask = (years >= 2003) & (years <= 2007)
    return X[mask], y[mask], dates[mask]


def select_validation_samples(X, y, dates):
    years = pd.DatetimeIndex(dates).year
    mask = (years >= 2008) & (years <= 2012)
    return X[mask], y[mask], dates[mask]
