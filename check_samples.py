import pandas as pd

from preprocessing.preprocess import (
    load_data,
    get_train_validation_rows,
    scale_full_series_using_train_max,
    make_supervised_samples,
    select_training_samples,
    select_validation_samples,
)

DATA_PATH = "data/streamflow.csv"

RAINFALL_LAG = 3
RUNOFF_LAG = 4

# Load data
df = load_data(DATA_PATH)

# Find training and validation periods
train_rows, valid_rows = get_train_validation_rows(df)

# Normalize
scaled_df, rain_max, runoff_max = (
    scale_full_series_using_train_max(
        df,
        train_rows
    )
)

print("\n========== TRAIN / VALIDATION ==========")

print("Training rows:", len(train_rows))
print("Validation rows:", len(valid_rows))

print(
    "Training:",
    train_rows["date"].min(),
    "to",
    train_rows["date"].max()
)

print(
    "Validation:",
    valid_rows["date"].min(),
    "to",
    valid_rows["date"].max()
)

print("\nTraining rainfall maximum:", rain_max)
print("Training streamflow maximum:", runoff_max)

# Create lagged samples
X, y, dates = make_supervised_samples(
    scaled_df,
    rainfall_lag=RAINFALL_LAG,
    runoff_lag=RUNOFF_LAG
)

# Select training samples
X_train, y_train, train_dates = (
    select_training_samples(
        X,
        y,
        dates
    )
)

# Select validation samples
X_valid, y_valid, valid_dates = (
    select_validation_samples(
        X,
        y,
        dates
    )
)

print("\n========== ANN SAMPLE CHECK ==========")

print("X_train shape:")
print(X_train.shape)

print("y_train shape:")
print(y_train.shape)

print("X_valid shape:")
print(X_valid.shape)

print("y_valid shape:")
print(y_valid.shape)

print("\nFirst training sample:")
print(X_train[0])

print("\nFirst training target:")
print(y_train[0])

print("\nFirst training target date:")
print(train_dates[0])

print("\nFirst validation sample:")
print(X_valid[0])

print("\nFirst validation target:")
print(y_valid[0])

print("\nFirst validation target date:")
print(valid_dates[0])

print("\n========== SAMPLE CHECK COMPLETE ==========")
print("NO MODEL HAS BEEN TRAINED.")