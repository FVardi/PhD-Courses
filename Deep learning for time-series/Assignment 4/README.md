# Assignment 4 — Self-Supervised Representation Learning for Time Series Classification

Frozen-representation evaluation of **T-Loss** and **TS2Vec** on FordA, ArticularyWordRecognition
and Epilepsy against raw-signal baselines, under 100% and 10% label regimes, plus a
contiguous-dropout robustness test, a TS2Vec masking ablation, and an exploratory TF-C
FD-A → FD-B bearing-fault transfer run.

Status: **structure only — no implementation yet.**

## External checkouts

The three official repositories are kept **outside** this repo and referenced by path from
[src/config.yaml](src/config.yaml). They are official checkouts at verified commits and must not be
modified; all compatibility and integration code lives under `src/methods/` and must not silently
alter a method's learning objective.

| Method | Commit | Location |
|---|---|---|
| T-Loss | `4aff592` | `Time series research/T-Loss` — **not yet cloned** |
| TS2Vec | `b0088e1` | `Time series research/ts2vec` — **not yet cloned** |
| TF-C   | `9667582` | `Time series research/TFC-pretraining` — present, includes FD-A/FD-B |

## Repository layout

```
src/
  config.yaml            Paths, seeds, label regimes, probe grids, corruption convention
  dataio/                Official split loading, stratified 10% subsets, train-only scaling
  corruption/            Contiguous sensor dropout (Part D)
  methods/               Adapters wrapping the external T-Loss / TS2Vec / TF-C checkouts
  representations/       Fixed-length representation extraction and caching
  probes/                Logistic regression and RBF-SVM probes, train-only tuning
  evaluation/            Accuracy / macro-F1, per-seed aggregation to mean +/- sd
  utils/                 Seeding, config loading, runtime and hardware recording
dev/
  0_fetch_data.py                 Fetch UCR/UEA datasets into data/raw/
  1_prepare_data.py               Official splits + save 10% label-subset indices
  2_baselines_raw.py              Part B  raw-signal baselines
  3_pretrain_tloss.py             Part C  T-Loss pretraining (no labels)
  4_pretrain_ts2vec.py            Part C  TS2Vec pretraining (no labels)
  5_extract_representations.py    Part C  frozen representation extraction
  6_probe_eval.py                 Part C  probe fitting + single test evaluation
  7_corruption_eval.py            Part D  clean vs contiguous dropout
  8_ts2vec_mask_ablation.py       Part E  FordA binomial vs continuous masking
  9_tfc_fda_fdb.py                Part F  unchanged TF-C FD-A -> FD-B run
  10_aggregate_results.py         Part G  result tables and figures
data/                    Not tracked; see data/README.md
results/
  label_subsets/         Saved 10% indices per (dataset, seed)
  checkpoints/           Pretrained encoders (not tracked)
  representations/       Cached encodings (not tracked)
  metrics/               Machine-readable per-seed results
  tables/                Aggregated mean +/- sd tables
  figures/               Plots for the presentation
  runs/                  Resolved configs, software versions, runtime, hardware
  tfc/                   Part F execution record
tests/                   Checks for label subsets, corruption and probe wiring
solution/                LaTeX write-up (Part A paper-to-code map)
```

## Protocol constants

Three fixed seeds (`42, 123, 456`). The 10% stratified subset a seed produces is reused by
**every** method, and its indices are saved. Paired comparisons share seeds, label subsets,
corruptions and probe settings. Tuning uses a training-only split or CV — never test performance.
In Parts C–E the encoder stays frozen and representation extraction is separate from probe
training; Part F follows TF-C's own downstream procedure, and the write-up states whether that
fine-tunes the encoder.

Both accuracy and macro-F1 are reported, and retained even where they support different
conclusions.

## Reproducing each result

To be filled in as each part lands: one command per result table or figure.

| Result | Command |
|---|---|
| _TBD_ | _TBD_ |

## Environment

To be recorded (Python version, package versions, hardware) once the first runs execute.
