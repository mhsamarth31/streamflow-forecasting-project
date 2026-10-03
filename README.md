# Streamflow Forecasting Using ANN and LSTM

College Mini Project based on the methodology in:

**Application of Artificial Neural Network in Streamflow Forecasting**  
Mian Xiao and Shanni You

## Project objective

Predict the next streamflow/runoff value using historical rainfall and streamflow values.

The project implements:
1. Feed-forward ANN baseline
2. LSTM sequence model
3. Overall RMSE evaluation
4. Peak vs non-peak RMSE evaluation
5. Hyperparameter/lag experiments

## Expected dataset

Put the dataset at:

`data/streamflow.csv`

The CSV must contain these columns:

```text
date,rainfall,runoff
2003-01-01,0.10,120.5
2003-01-02,0.00,115.2
...
```

- `date`: chronological date
- `rainfall`: rainfall value
- `runoff`: streamflow/runoff value

The original paper uses daily rainfall and runoff data from the Leaf River basin near Collins, Louisiana, for 2003-2012. Training is 2003-2007 and validation is 2008-2012.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

After placing `data/streamflow.csv`:

```bash
python main.py
```

Results are written to `results/`.
## Baselines and multi-seed evaluation

To check whether the ANN/LSTM errors are actually good, `baselines_and_seeds.py` compares them with simple models and repeats training with several random seeds. It uses the same train/validation samples and final settings as `main.py`.

```bash
python baselines_and_seeds.py --no-nn      # baselines only (takes seconds)
python baselines_and_seeds.py --seeds 5    # baselines + 5 seeds each for ANN and LSTM
```

Outputs: `results/baseline_metrics.csv` and `results/multi_seed_metrics.csv`.

| Model | Validation RMSE (2008-2012) |
|---|---|
| Train-mean predictor | 63.45 |
| Persistence (Q(t) = Q(t-1)) | 46.77 |
| Linear regression (rain + runoff lags) | 45.01 |
| Linear regression (runoff lags only) | 43.42 |
| ANN (5 seeds, mean ± std) | 44.76 ± 0.44 |
| LSTM (5 seeds, mean ± std) | 41.03 ± 0.94 |

The LSTM beat the ANN in all 5 seeds. There are no validation days above the 1500 peak threshold, so peak RMSE cannot be computed for this validation period.

## Output

The program produces:

- trained ANN model
- trained LSTM model
- prediction CSV files
- overall RMSE
- peak RMSE
- non-peak RMSE
- actual-vs-predicted plots
- training-loss plots
- comparison plots

## Important

Do not commit the dataset if its license/repository rules prohibit redistribution. Keep the dataset locally and document its official source in the final report.

## Project structure

```text
streamflow_forecasting_project/
├── data/
│   └── streamflow.csv
├── models/
│   ├── ann_model.py
│   └── lstm_model.py
├── preprocessing/
│   └── preprocess.py
├── training/
│   ├── train_ann.py
│   └── train_lstm.py
├── evaluation/
│   └── evaluate.py
├── visualization/
│   └── plots.py
├── results/
├── main.py
├── requirements.txt
└── README.md
```


## How training and validation work

The code does **not** randomly split the time series.

1. The complete chronological series is cleaned and sorted.
2. Training target dates are 2003-2007.
3. Validation target dates are 2008-2012.
4. Lag/sequence samples are created from the complete chronological series.
5. Therefore, an early validation prediction can use valid historical context immediately before it (for example, late-2007 runoff when predicting an early-2008 value).
6. Scaling constants are calculated from the training period only and then applied to the complete series.
7. The validation targets are never used for gradient updates.

This is important for time-series experiments because the validation period should be later in time, not randomly mixed with the training period.

## Configurations

### Reference paper settings (starting point)

- **ANN:** rainfall lag = 3, runoff lag = 4, hidden units = 12, sigmoid hidden activation, linear output, learning rate = 0.009, maximum epochs = 200
- **LSTM:** sequence length = 5, features per timestep = rainfall + runoff, hidden units = 12, learning rate = 0.001, maximum epochs = 200

### Final settings used in this project (`main.py`)

The paper's settings were used as a starting point. Hyperparameter experiments on our dataset (`quick_experiment.py`, results in `results/hyperparameter_results.csv`) led to the following final settings:

- **ANN:** rainfall lag = 5, runoff lag = 5, hidden units = 4, sigmoid hidden activation, linear output, learning rate = 0.001, epochs = 200, batch size = 32
- **LSTM:** sequence length = 7, features per timestep = rainfall + runoff, hidden units = 4, learning rate = 0.0001, epochs = 200, batch size = 32

Note: hyperparameters were chosen using the validation set (no separate test set was held out), so the reported errors may be slightly optimistic.

### Evaluation

- overall RMSE
- peak RMSE: actual runoff > 1500
- non-peak RMSE: actual runoff <= 1500

The threshold and configurations follow the paper's reported experimental setup. The scaling implementation in this repository uses training-period maxima to avoid validation leakage; if exact numerical reproduction of the paper is required, document the paper's original normalization convention separately.
