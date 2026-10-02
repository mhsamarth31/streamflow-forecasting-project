from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/streamflow.csv")

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET CHECK ==========")

print("Columns:")
print(df.columns.tolist())

print("\nTotal rows:")
print(len(df))

# Convert columns
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
df["runoff"] = pd.to_numeric(df["runoff"], errors="coerce")

print("\nMissing values:")
print(df[["date", "rainfall", "runoff"]].isna().sum())

# Remove invalid rows
df = df.dropna(
    subset=["date", "rainfall", "runoff"]
)

# Sort chronologically
df = df.sort_values("date")

print("\nDate range:")
print(df["date"].min())
print(df["date"].max())

print("\nDuplicate dates:")
print(df["date"].duplicated().sum())

# Select paper-style period
project_df = df[
    (df["date"].dt.year >= 2003) &
    (df["date"].dt.year <= 2012)
].copy()

print("\n========== 2003-2012 ==========")

print("Rows:")
print(len(project_df))

print("\nStart:")
print(project_df["date"].min())

print("\nEnd:")
print(project_df["date"].max())

# Check daily gaps
expected_dates = pd.date_range(
    project_df["date"].min(),
    project_df["date"].max(),
    freq="D"
)

actual_dates = pd.DatetimeIndex(
    project_df["date"]
)

missing_dates = expected_dates.difference(
    actual_dates
)

print("\nMissing dates:")
print(len(missing_dates))

if len(missing_dates) > 0:
    print(missing_dates[:20])

print("\nRainfall statistics:")
print(project_df["rainfall"].describe())

print("\nStreamflow statistics:")
print(project_df["runoff"].describe())

print("\n========== CHECK COMPLETE ==========")
print("NO MODEL HAS BEEN TRAINED.")