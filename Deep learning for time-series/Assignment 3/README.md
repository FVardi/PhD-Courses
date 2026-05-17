# Assignment 3 — RUL Prediction on CMAPSS

Remaining Useful Life (RUL) prediction on the NASA CMAPSS turbofan engine dataset using RNN, LSTM, TCN, and XGBoost. Two data pipelines are compared: a raw sliding-window approach and a feature-sequence approach. Results are reported for FD001 (single operating condition) and FD002 (six operating conditions).

## Requirements

Python 3.13. Install dependencies:

```
pip install torch scikit-learn xgboost optuna pandas numpy pyarrow pyyaml joblib tqdm matplotlib
```

Key versions used during development:

| Package | Version |
|---|---|
| torch | 2.11.0 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| optuna | 4.8.0 |
| pandas | 3.0.1 |
| numpy | 2.4.2 |

## Repository layout

```
src/
  config.yaml              # All hyperparameters and settings
  data/CMAPSSData/         # Raw CMAPSS data files (not tracked in git)
  results/
    selected_features.yaml # Chosen sensor channels
    splits_FD001/          # Preprocessed train/val/test splits for FD001
    splits_FD002/          # Preprocessed train/val/test splits for FD002
    checkpoints/           # Saved model weights (.pt)
    predictions/           # Per-engine test predictions (.parquet)
    figures/               # Generated plots
    all_results_FD001.csv  # Aggregated results table for FD001
    all_results_FD002.csv  # Aggregated results table for FD002
    lstm_study_val.csv     # Hyperparameter study validation results
    lstm_study_test.csv    # Hyperparameter study test results
dev/
  1.1_eda.py               # FD001 exploratory data analysis
  1.2_data_preparation.py  # FD001 preprocessing and splits
  1.3_datasets.py          # Dataset classes (SlidingWindow, FeatureSequence)
  1.4_models.py            # Model definitions (RNN, LSTM, TCN)
  1.5_train.py             # Training loop
  1.6_evaluate.py          # Test evaluation and prediction saving
  1.7_xgboost_tune.py      # Optuna hyperparameter tuning for XGBoost
  1.8_xgboost_baseline.py  # XGBoost training and evaluation
  1.9_run_all.py           # Run all (approach, model, seed) combinations
  1.10_aggregate_results.py# Aggregate results CSVs into summary tables
  1.11_lstm_hparam_study.py# LSTM hyperparameter grid search
  2.1_eda.py               # FD002 exploratory data analysis
  2.2_data_preparation.py  # FD002 preprocessing (clustering + per-cluster normalisation)
report/
  report.tex               # LaTeX report
```

## Reproduction

All scripts are run from the `dev/` directory. The dataset files must be present under `src/data/CMAPSSData/`.

### Step 1 — FD001 preprocessing and EDA

```bash
cd "Deep learning for time-series/Assignment 3/dev"
python 1.1_eda.py
python 1.2_data_preparation.py
```

`1.1_eda.py` computes sensor correlations and writes `src/results/selected_features.yaml`.
`1.2_data_preparation.py` produces the normalised train/val/test splits under `src/results/splits_FD001/`.

### Step 2 — FD002 preprocessing and EDA

```bash
python 2.1_eda.py
python 2.2_data_preparation.py
```

`2.2_data_preparation.py` fits KMeans ($k=6$) on FD002 operating settings, assigns cluster labels, applies per-cluster z-score normalisation, and saves splits to `src/results/splits_FD002/`.

### Step 3 — XGBoost tuning

```bash
python 1.7_xgboost_tune.py                  # FD001
python 1.7_xgboost_tune.py --dataset FD002  # FD002
```

Runs 100 Optuna trials and saves the best parameters to `src/results/xgboost_best_params.yaml` and `src/results/FD002_xgboost_best_params.yaml`.

### Step 4 — Train and evaluate all models

```bash
python 1.9_run_all.py                  # FD001
python 1.9_run_all.py --dataset FD002  # FD002
```

Trains every `(approach, model, seed)` combination — 6 deep learning configurations × 5 seeds plus XGBoost — and saves results to `src/results/all_results_{DATASET}.csv`. Completed runs are skipped automatically (resumable). Delete the corresponding `.parquet` file under `src/results/predictions/` to force a re-run for a specific configuration.

### Step 5 — Hyperparameter study

```bash
python 1.11_lstm_hparam_study.py
```

Runs a grid search over window size, hidden size, and learning rate for the LSTM model on FD001. Results are written to `src/results/lstm_study_val.csv` and `src/results/lstm_study_test.csv`.

## Configuration

All training settings are centralised in `src/config.yaml`. The feature-sequence approach uses a separate set of overrides to compensate for the small number of gradient updates per epoch that arises from engine-level batching:

| Parameter | Window | Sequence |
|---|---|---|
| Batch size | 64 | 8 |
| Learning rate | 1e-3 | 3e-4 |
| Gradient clip norm | 1.0 | 0.25 |
| Early stopping patience (epochs) | 10 | 50 |
| Max epochs | 200 | 500 |

RNN and LSTM hidden-to-hidden weights are initialised orthogonally. The LSTM forget gate bias is initialised to 1 to preserve gradient flow on long sequences.

## Key results

### FD001

| Approach | Model | RMSE (mean ± std) | NASA (mean ± std) |
|---|---|---|---|
| window | LSTM | 14.88 ± 0.36 | 444 ± 65 |
| window | RNN | 15.82 ± 0.57 | 502 ± 100 |
| window | TCN | 16.55 ± 0.32 | 477 ± 37 |
| sequence | XGBoost | 14.46 ± 0.05 | 386 ± 15 |
| sequence | TCN | 17.60 ± 0.24 | 508 ± 24 |
| sequence | RNN | 20.66 ± 3.72 | 646 ± 171 |

### FD002

| Approach | Model | RMSE (mean ± std) | NASA (mean ± std) |
|---|---|---|---|
| sequence | XGBoost | 14.10 ± 0.01 | 775 ± 7 |
| sequence | RNN | 14.59 ± 0.71 | 834 ± 62 |
| window | LSTM | 14.78 ± 0.60 | 1132 ± 169 |
| window | RNN | 15.44 ± 0.86 | 1120 ± 118 |
| sequence | TCN | 16.73 ± 0.53 | 1104 ± 40 |
| window | TCN | 16.45 ± 0.42 | 1217 ± 117 |

Sequence LSTM is excluded from the summary tables due to training instability (bimodal results: 1 of 5 seeds converges, 4 collapse to a constant predictor). See the report for full details.
