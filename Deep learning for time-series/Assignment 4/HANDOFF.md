# Handoff — Assignment 4, Self-Supervised Representation Learning for Time Series

## GOAL

Evaluate whether frozen self-supervised representations (T-Loss, TS2Vec) beat raw-signal
baselines for time-series classification at 100% and 10% label budgets, then test robustness
to contiguous sensor dropout, a TS2Vec masking ablation, and a TF-C FD-A→FD-B reproduction.
Deliverable is a LaTeX write-up in `solution/solution.tex` plus machine-readable results.

## CONTEXT

- **Root:** `C:/Users/au808956/Documents/Repos/PhD-Courses/Deep learning for time-series/Assignment 4`
- **Python 3.13.11**, venv at `PhD-Courses/.venv` (shared across courses).
  `torch 2.11.0+cu128` (CUDA build — was CPU-only, reinstalled), sklearn 1.8, numpy 2.4,
  pandas 3.0, matplotlib 3.10. GPU: NVIDIA RTX 4500 Ada, 25.8 GB, `cuda.is_available()` True.
- **External checkouts** at `C:/Users/au808956/Documents/Repos/Time series repos/`
  (this folder was renamed mid-project from `Time series research`; `checkouts.root` in
  `src/config.yaml` points at the current name):
  - `UnsupervisedScalableRepresentationLearningTimeSeries` @ `4aff592` — this is T-Loss
  - `ts2vec` @ `b0088e1`
  - `TFC-pretraining` @ `9667582`
  All three verified clean (`git status` shows no modified `.py`). **They must not be
  modified**; all compatibility code lives in `src/methods/`.
- **Datasets** in `data/raw/` (gitignored, 194 MB): FordA, FordB, ArticularyWordRecognition
  (AWR), Epilepsy. UCR/UEA official train/test partitions. Part B–E use FordA, AWR,
  Epilepsy; FordB is optional and **skipped**.
- **Array convention:** `(n_series, n_channels, length)` throughout. TS2Vec wants
  `(N, T, C)`; the transpose happens only in `src/methods/ts2vec.py`.

### Repo layout

```
src/utils/config.py        load_config(), ${...} interpolation, results_dir()
src/utils/seeding.py       set_seed(); torch imported lazily
src/dataio/ts.py           .ts parser (only format all 4 datasets ship)
src/dataio/splits.py       official splits + .npy cache + shape validation
src/dataio/subsets.py      stratified label subsets, save/load
src/dataio/scaling.py      per-channel fit/apply
src/representations/raw.py flatten — the baseline "encoder"
src/methods/_checkout.py   activate(): stops the two checkouts colliding
src/methods/tloss.py       T-Loss adapter + Python 3.12 import shim
src/methods/ts2vec.py      TS2Vec adapter + (N,C,T)->(N,T,C)
src/methods/__init__.py    load_pretrained(), encode(), checkpoint_path()
src/probes/probes.py       both probes, grids, fold rule, a priori defaults
src/evaluation/metrics.py  accuracy, macro-F1, mean_sd
src/corruption/dropout.py  Part D contiguous dropout
dev/0_..10_                pipeline drivers (thin; logic lives in src/)
eda/                       4 dataset notebooks + plots.py (exploration only)
```

## DECISIONS MADE

- **Scaling:** per-channel, fitted on the **full TRAIN split** (label-free, so not leakage),
  applied before flattening/encoding. Only Epilepsy is materially affected — FordA, FordB and
  AWR ship already per-series z-normalised.
- **Representation scaling:** `StandardScaler` on the 320 encoder dims, fit on clean TRAIN,
  in Part C and D. Measured per-dim sd spread 7.8× (T-Loss) / 10.8× (TS2Vec) vs 1.1–2.0× for
  raw features. Part B deliberately gets no equivalent step — adding one moves its numbers by
  ≤0.0145 (measured). Deviates from TS2Vec's own protocol, which scales before its logistic
  regression but not its SVM.
- **18 pretraining runs** (3 datasets × 3 seeds × 2 methods), not 6 — captures encoder
  training variance.
- **Published defaults unchanged** for both methods; only `in_channels`/`input_dims` (from
  data) and device are set. T-Loss `nb_steps=1500`; TS2Vec `n_iters` by its own rule = 600
  FordA/AWR, **200 Epilepsy**.
- **`encoding_window='full_series'`** for TS2Vec (its own classification protocol).
- **Determinism: do nothing.** No cuDNN flags, no strict mode. Write-up should state
  pretraining is seeded but not bitwise reproducible on GPU.
- **Untunable cells use a priori defaults** from `config.yaml`
  (`{C:1}` / `{C:1, gamma:scale}`, sklearn defaults), **not** inheritance from the 100% cell.
  User's decision, commit `df091bc`. Reason: inheritance leaks a choice made under a larger
  label budget, and that choice is itself a near-arbitrary tie-break because AWR@100% is
  saturated (all logistic-regression `C` within 1% of best).
- **Fold rule:** `k = min(configured_folds, min_class_count)`, floor 2; if
  `min_class_count < 2` the cell is untunable. Gives k=5 FordA, k=3 Epilepsy@10%, untunable
  AWR@10%.
- **100% cell runs per seed** in Parts B and C — only the CV fold shuffle varies in Part B,
  but that changes the selected hyperparameter and hence the score. Running once reported a
  fabricated sd of 0.
- **Corruption design (Part D), fixed before any result was inspected**, in `config.yaml`
  with inline justification: 20% of each series, 1 contiguous interval, uniform random start
  per series, **all channels share the interval**, fill = train channel mean, test-only,
  seeded by dataset seed.
- **`5_` stays clean-only**; `7_` owns corruption and does its own encoding of corrupted test
  data.
- **Both probes** reported for SSL rows in Part D, not just the principal one.

## CURRENT STATE

### Working (verified)

- **Parts A–C complete.** 18/18 checkpoints, 36/36 representation arrays, Part B 36 rows,
  Part C 72 rows.
- `dev/0_` fetch+verify (parses `.ts` headers, checks counts against `datasets.meta`),
  `1_` prepare, `2_` baselines, `3_`/`4_` pretrain, `5_` extract, `6_` probe eval — all run
  clean end to end.
- Verified: seeds reproduce encoder init exactly and differ across seeds; label subsets are
  byte-reproducible and barely overlap across seeds; header and row labels match in all 8
  splits; 1-NN Euclidean on FordA reproduces the published ~0.66 reference (0.6652), which is
  what confirmed FordA's near-chance logistic-regression result is real and not a bug.
- T-Loss pretraining: 65 min total for the 7 recorded runs (2 FordA runs finished before
  record-writing was made incremental, so they have checkpoints but no runtime row).
  TS2Vec: 161 s for all 9.

### In progress / not started

- `dev/7_corruption_eval.py` is **written and compiles**, but has only been smoke-tested on
  Epilepsy. `results/metrics/corruption.csv` currently holds that partial run (18 rows) and
  must be regenerated with a full run.
- `dev/8_ts2vec_mask_ablation.py`, `dev/9_tfc_fda_fdb.py`, `dev/10_aggregate_results.py`
  are still 4-line stubs.
- `solution/solution.tex`: every results table is empty and every `\TODOfill` unfilled.

### Uncommitted

HEAD is `df091bc "Ready for part c"`. Modified: `dev/2_`–`7_`, `src/config.yaml`,
`src/methods/__init__.py`, `eda/ford_dataset.ipynb`, `results/metrics/baselines_raw.csv`.
Untracked: `src/corruption/dropout.py`, `src/methods/{_checkout,tloss,ts2vec}.py`,
`eda/{awr,epilepsy,fordb}_dataset.ipynb`, `eda/plots.py`, all `results/metrics/*.csv` and
`results/runs/*.csv`.

## KEY ARTIFACTS

### Commands

```bash
python dev/0_fetch_data.py [--verify-only]
python dev/1_prepare_data.py
python dev/2_baselines_raw.py
python -u dev/3_pretrain_tloss.py      # ~11 min/run; run in a real terminal, not background
python -u dev/4_pretrain_ts2vec.py     # ~3 min total
python dev/5_extract_representations.py
python dev/6_probe_eval.py
python dev/7_corruption_eval.py        # not yet run in full
```

All are resumable/idempotent: they skip existing checkpoints or outputs. `--force` retrains.
Smoke runs (`--nb-steps` / `--n-iters`) write `_smoke<N>` filenames so a short encoder can
never be mistaken for a protocol run.

### Results (test accuracy, mean ± sd over seeds 42/123/456)

```
dataset      probe   labels  raw              tloss            ts2vec
FordA        logreg  100%    0.4922±0.0034    0.9230±0.0102    0.9273±0.0077
FordA        logreg   10%    0.5058±0.0083    0.9098±0.0055    0.9121±0.0053
FordA        rbfsvm  100%    0.8280±0.0000    0.9293±0.0031    0.9348±0.0072
FordA        rbfsvm   10%    0.6689±0.0131    0.9152±0.0085    0.9169±0.0052
AWR          logreg  100%    0.9733±0.0000    0.9656±0.0077    0.9833±0.0033
AWR          logreg   10%    0.8278±0.0222    0.8144±0.0506    0.8378±0.0271
AWR          rbfsvm  100%    0.9800±0.0000    0.9700±0.0058    0.9833±0.0058
AWR          rbfsvm   10%    0.2300±0.1305    0.1800±0.0416    0.1578±0.0453   <- see below
Epilepsy     logreg  100%    0.6329±0.0167    0.9686±0.0084    0.9662±0.0084
Epilepsy     logreg   10%    0.4396±0.0715    0.7440±0.0233    0.7367±0.0493
Epilepsy     rbfsvm  100%    0.8478±0.0000    0.9734±0.0042    0.9565±0.0000
Epilepsy     rbfsvm   10%    0.5145±0.1007    0.7005±0.0633    0.6546±0.0834
```

Machine-readable: `results/metrics/baselines_raw.csv` (Part B),
`results/metrics/ssl_probe.csv` (Part C). Both carry per-seed rows with
`params`, `tuned`, `folds`, `tuning_note`, `fit_seconds`.

### Findings that shape the write-up

- **FordA is the headline.** Raw logistic regression is *below chance* (0.4922 vs a 0.5159
  majority-class baseline) and both encoders lift it to ~0.92. Mechanism established by an
  EDA experiment: randomly circular-shifting each series destroys AWR (0.9733 → 0.1367) but
  leaves FordA unchanged (0.4886 → 0.4856), i.e. FordA has arbitrary phase, so no fixed
  weight vector exists. Convolutional encoders supply the missing shift-invariance.
- **AWR is where SSL does not help** — both encoders sit at or below the raw baseline at
  100%. AWR is temporally aligned (segmented word utterances) and already near ceiling.
- **AWR @ 10% RBF-SVM is near-chance for every representation** because the cell is
  untunable and the a priori default `{C:1, gamma:'scale'}` overfits at n=27 (it reproduces
  all 27 training labels but generalises to ~0.15). This measures the fallback rule, not the
  representations — say so explicitly.
- **AWR @ 10% has ~1 labelled example per class** (27 samples, 25 classes). The label regime
  is defined as 10% of the *total*, which hands FordA 180 examples/class and AWR 1 — worth a
  sentence, since it makes the 10% column a harder problem for many-class datasets.
- **T-Loss loaders compute normalisation statistics on train+test combined** (both UCR and
  UEA). We bypass their loaders and use a train-only scaler, so we are *more* rigorous than
  the original — state it, so a small shortfall vs published numbers isn't read as a failed
  reproduction.
- **Epilepsy trains a third as long as the others under TS2Vec** (200 vs 600 iters, by
  TS2Vec's own rule). The training budget is not one number.

## DEAD ENDS

- **`timeout: 600000` on a background Bash run makes it *shorter*, not longer** (10 min vs
  the ~30 min default cap). Long pretraining must run in the user's own terminal with
  `python -u`; otherwise stdout buffering also loses all output when killed.
- `losses/__init__.py` and `networks/__init__.py` in the T-Loss checkout call
  `loader.find_module()`, removed in Python 3.12. Do **not** edit the checkout —
  `src/methods/tloss.py:install_checkout()` pre-builds both packages into `sys.modules`.
- Both checkouts ship a top-level `utils.py`; importing one poisons the other.
  `src/methods/_checkout.py:activate()` fixes this. Verified by interleaving
  T-Loss → TS2Vec → T-Loss → TS2Vec in one process.
- T-Loss `fit()` also trains an internal SVM, and `save()` pickles it via
  `sklearn.externals.joblib` (removed from modern sklearn). Use `fit_encoder()` and
  `save_encoder()` only.
- A within/between-class distance ratio on representations was tried as a quality proxy —
  it is a poor predictor (wrong space, wrong geometry for a hyperplane, dimensionality not
  comparable). Do not put it in the write-up.

## OPEN ISSUES

- `results/metrics/corruption.csv` is from an Epilepsy-only smoke run; needs a full run.
- Two T-Loss FordA encoders (seeds 42, 123) have no runtime row. Re-running them with
  `--force` costs ~21 min and yields identical encoders (seeding is verified reproducible);
  the only gain is a complete training-time table. **User has not decided.**
- `probes.logistic_regression.C` and `probes.rbf_svm` grids still carry a `TODO confirm`
  in `config.yaml`. Results were produced with them as written.
- How to present the AWR @ 10% RBF-SVM cell in the tables (report as-is with a note, or as a
  range). *Suggested, not confirmed:* report as-is and explain it measures the fallback.
- Part D's table in `solution.tex` lists T-Loss/TS2Vec as single rows, implying one probe,
  but `7_` computes both. Table layout not yet reconciled.

## NEXT STEPS

1. Run `python dev/7_corruption_eval.py` in full (all 3 datasets, 3 representations, 2
   probes, 3 seeds = 54 rows) and check the clean column reproduces Parts B and C.
2. Write `dev/8_ts2vec_mask_ablation.py` — FordA, TS2Vec `binomial` (default) vs
   `continuous` masking, evaluated clean and corrupted. `mask_mode` is a `TSEncoder`
   constructor argument in the checkout; `src/methods/ts2vec.py:build()` currently does not
   expose it.
3. Fill `solution/solution.tex` Tables 1–2 from `baselines_raw.csv` and `ssl_probe.csv`, and
   write the Part A–C `\TODOfill` sections.

## WORKING PREFERENCES

- **Never create README or documentation files unless explicitly asked.** Findings go in the
  reply, not into new `.md` files. (Also stored in Claude's memory.)
- The user is learning the material and wants to be involved: explain mechanisms, lay out
  decisions with trade-offs, and wait for their call on protocol choices rather than deciding
  unilaterally.
- They check understanding by restating things — correct errors directly and precisely.
- Verify claims by running code rather than asserting them; they have repeatedly asked for
  things to be checked before being believed.
- Code style: thin `dev/` drivers, all reusable logic in `src/`, docstrings that explain
  *why* a choice was made (especially protocol deviations), no unexplained magic constants.
