# Assignment 3 — Progress Tracker

## Part 1 — FD001: Single Condition, Single Fault Mode

- [x] 1.1 Problem understanding (report written)
- [x] 1.2 EDA — lifetime distribution, sensor trajectories, Spearman correlation, sensor shortlist
- [x] 1.3 Data preparation — RUL cap, engine-wise split, normalisation, fixed seed
- [x] 1.4 Approach 1 — raw sliding-window pipeline (RNN, LSTM, TCN)
- [x] 1.5 Approach 2 — feature-sequence pipeline (RNN, LSTM, TCN)
- [x] 1.6 Models — RNN, LSTM, TCN implemented for both approaches
- [x] 1.7 XGBoost baseline — Optuna tuning + evaluation on FD001

## Part 2 — Evaluation Protocol

- [x] 2.1 Metrics — RMSE and NASA score reported for all configurations
- [x] 2.2 Multiple seeds — 5 seeds, mean ± std reported in results table
- [x] 2.3 Fair comparison — same splits, RUL target, window size, normalisation across models

## Part 3 — Focused Hyperparameter Study

- [x] Training — LSTM grid search on FD001 (window × hidden × lr, 3 seeds each, 72 runs)
- [x] Training — LSTM grid search on FD002 (running on other machine)
- [ ] 3.1 Setup — write up in report: architecture choice (LSTM), grid axes, seed budget
- [ ] 3.2 Analysis — write up in report (see outline below)
- [ ] ⚠ Re-run sequence hyperparameter study for FD001 with corrected training protocol — the original grid (12 configs, 3 seeds) produced val RMSE 45–87 which reflects training failure, not hyperparameter effects; results are not interpretable until re-run with sequence-specific fixes applied
- [ ] ⚠ Run LSTM hyperparameter study for FD002 — planned but never completed; note that the sequence LSTM on FD002 has a stability problem (4/5 seeds collapse) that should be resolved before the sequence part of this study is attempted

## Part 4 — FD002 Extension

- [x] 4.1 EDA — op-settings distribution, clustering visualisation, raw vs normalised trajectories
- [x] 4.2 Clustering and normalisation — KMeans k=6, per-cluster z-score, no leakage
- [x] 4.3 Pipelines and models — both approaches, all three architectures on FD002
- [x] 4.4 XGBoost baseline on FD002

## ⚠ Point of Attention — Sequence RNN/LSTM Results

The sequence-approach RNN and LSTM results (RMSE ~43-44) are suspected to reflect undertrained models
rather than true model capability. Root cause: `batch_size=64` was applied to engine-level samples
(pool of ~80 engines), giving only **2 gradient updates per epoch** vs ~213 for the window approach.
With `early_stopping_patience=10`, the models stopped after as few as ~20-40 gradient updates total.

**Fix applied:** added `sequence_batch_size: 8` to `config.yaml` and updated `1.5_train.py` to use it
for the sequence DataLoader. This gives ~10 batches per epoch (80 engines / 8), making the training
dynamics more comparable to the window approach.

**Action required:**
- [ ] Re-run `1.9_run_all.py` (sequence approach only, all seeds) on both FD001 and FD002 with the new config
- [ ] Re-run `1.11_lstm_hparam_study.py` for both datasets with the new config
- [ ] Compare new sequence RNN/LSTM results to old — if RMSE drops significantly, the original results were undertrained
- [ ] Decide whether to replace the results tables in the report

---

## Part 5 — Comparative Analysis

- [ ] 5.1 Within-approach comparison
- [ ] 5.2 Between-approach comparison
- [ ] 5.3 Deep learning vs XGBoost
- [ ] 5.4 FD001 vs FD002
- [ ] 5.5 RMSE vs NASA score

## Deliverables

- [ ] README.md — reproduction instructions
- [ ] Report finalised (all sections complete)
- [ ] Oral presentation / slides

---

## Part 3.2 — Hyperparameter Study Analysis Outline

> Write after FD002 hparam study results are available. FD001 results ready now.

**Points to address:**

1. **Best configuration per approach** — report winning `(window_size, hidden_size, lr)` for window and sequence approaches on both datasets, with val RMSE mean ± std
2. **Sensitivity analysis** — which axis matters most?
   - Window size: does more context help, or does the model saturate early?
   - Hidden size: is 128 meaningfully better than 64, or is the model capacity not the bottleneck?
   - Learning rate: how much does 1e-4 vs 1e-3 change the outcome?
3. **Grid-edge check** — is the best config at the boundary of the search grid (e.g. largest window, smallest lr)? If so, the grid may be too narrow.
4. **Default config retrospective** — was the Part 1 default (w=30, h=128, lr=1e-3) a reasonable starting point? Should a different default have been used?

---

## Part 5 — Comparative Analysis Outline

### 5.1 Within-approach comparison

*For each approach, rank RNN vs LSTM vs TCN on FD001 and FD002.*

**Raw-window approach:**
- LSTM wins on RMSE on both datasets (FD001: 14.88, FD002: 14.78)
- RNN second, TCN third — differences are small (~1-2 RMSE points)
- Ask: are window/LSTM vs window/RNN differences significant given seed std?

**Feature-sequence approach:**
- TCN wins decisively (FD001: 17.60, FD002: 16.73)
- RNN and LSTM collapse to RMSE ~43-44 on both datasets — effectively not learning
- This collapse is the key finding: discuss why RNN/LSTM fail under the sequence-to-sequence setup (vanishing gradient across long padded sequences, masking not fully compensating)

**Cross-approach ranking consistency:**
- Rankings are not the same: LSTM leads in window, TCN leads in sequence
- TCN's local convolutions are robust to sequence length; RNN/LSTM are not

### 5.2 Between-approach comparison

*For each architecture, compare window vs sequence.*

- **RNN/LSTM**: window dramatically better than sequence (~15 vs ~44 RMSE) — feature-sequence hurts badly for recurrent models
- **TCN**: approaches are roughly comparable (window 16.55 vs sequence 17.60 on FD001) — TCN handles both paradigms similarly well
- Conclusion: the feature-sequence benefit depends entirely on architecture; it does not universally help
- The full-history argument for sequence: the advantage of seeing the entire engine trajectory is not materialising for RNN/LSTM, possibly due to the difficulty of training long masked sequences
- FD001 vs FD002 consistency: same pattern holds on both datasets

### 5.3 Deep learning vs XGBoost

- XGBoost is best overall: FD001 RMSE 14.46, FD002 RMSE 14.10
- Beats all DL models including best window-LSTM (14.88 FD001, 14.78 FD002)
- Margin is small and within seed variability for RMSE, but NASA score gap is larger (386 vs 444)
- Interpretation: XGBoost treats each window independently (IID) yet wins — suggests the temporal structure across windows is not being exploited effectively by the DL models, or that the handcrafted features already summarise the relevant signal sufficiently
- This is a finding, not a failure: the problem may not require long-range temporal modelling

### 5.4 FD001 vs FD002

- Window models: RMSE roughly stable across datasets (FD001 ~15-17, FD002 ~15-16) — regime-aware normalisation largely closes the gap
- Sequence RNN/LSTM: collapse on both datasets equally — multi-condition does not make it worse because the baseline is already broken
- Sequence TCN: improves slightly from FD001 (17.60) to FD002 (16.73) — possibly the op_condition feature adds useful signal
- NASA score degrades substantially on FD002 for window models (e.g. window-RNN: 502 → 1120) despite similar RMSE — models become more biased (systematically late predictions) under multi-condition data
- XGBoost is surprisingly stable or better on FD002 (14.46 → 14.10 RMSE; 386 → 775 NASA) — NASA score degradation suggests more late predictions despite better RMSE

### 5.5 RMSE vs NASA score

- Rankings broadly agree: configurations that win on RMSE also tend to win on NASA score
- Key exception: window-LSTM has better RMSE than window-TCN on FD001 (14.88 vs 16.55) but similar NASA scores (444 vs 477) — small gap, not clearly significant
- Sequence RNN/LSTM: RMSE ~43-44 but NASA score is extreme (5827-5847 FD001, ~51k FD002) — the NASA penalty amplifies the already-high errors exponentially, consistent with systematically late predictions
- FD002 NASA inflation: even models with reasonable RMSE (~15-17) see NASA scores double or triple vs FD001, suggesting they predict RUL too high (too conservative / late) on multi-condition data
- Practical implication: a model with better RMSE is not guaranteed to be safer — NASA score captures the asymmetric cost of overestimating remaining life