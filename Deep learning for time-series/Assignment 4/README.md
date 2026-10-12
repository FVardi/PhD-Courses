# Assignment 4 — Self-Supervised Representation Learning for Time Series Classification

Frozen-representation evaluation of **T-Loss** and **TS2Vec** on FordA, ArticularyWordRecognition
(AWR) and Epilepsy against raw-signal baselines, with 100% and 10% of the training labels. Also a
contiguous-dropout robustness test, a TS2Vec masking ablation, and an exploratory run of the
released TF-C code for FD-A → FD-B bearing-fault transfer.

The write-up is [solution/solution.pdf](solution/solution.pdf). Every number in its tables is
read from the per-seed CSV files in `results/`, as listed under
[Reproducing each result](#reproducing-each-result).

## Environment

The results were produced with:

| Component | Version |
|---|---|
| OS | Windows 11 |
| Python | 3.13.11 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| NumPy | 2.4.2 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.1 |
| SciPy | 1.17.1 |
| PyYAML | 6.0.3 |
| matplotlib | 3.10.8 (figures only) |
| GPU | NVIDIA RTX 4500 Ada Generation, 25.8 GB |

Setup:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scikit-learn pandas scipy pyyaml matplotlib
```

A CUDA GPU is required. The TF-C code creates its augmentation masks with
`torch.cuda.FloatTensor` and fails without one (`RuntimeError: No CUDA GPUs are available`).
T-Loss and TS2Vec run on CPU with `--cpu`, but T-Loss is then very slow.

All commands below are run from this folder (`Assignment 4/`).

## External checkouts

The three official repositories are kept **outside** this repository, at the verified commits,
and are never modified. All compatibility code is in `src/methods/`.

| Method | Repository | Commit |
|---|---|---|
| T-Loss | `github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries` | `4aff592` |
| TS2Vec | `github.com/zhihanyue/ts2vec` | `b0088e1` |
| TF-C | `github.com/mims-harvard/TFC-pretraining` | `9667582` |

```bash
git clone https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
git -C UnsupervisedScalableRepresentationLearningTimeSeries checkout 4aff592
git clone https://github.com/zhihanyue/ts2vec
git -C ts2vec checkout b0088e1
git clone https://github.com/mims-harvard/TFC-pretraining
git -C TFC-pretraining checkout 9667582
```

Clone all three into one folder and point `checkouts.root` in
[src/config.yaml](src/config.yaml) at it. On another machine, put the override in an untracked
`src/config.local.yaml` instead of editing the tracked file:

```yaml
checkouts:
  root: "D:/path/to/the/folder/with/the/three/checkouts"
```

## Data

Nothing in `data/` is tracked; [data/README.md](data/README.md) has the dataset sizes and
caveats. Expected layout:

```
data/raw/FordA/  ArticularyWordRecognition/  Epilepsy/  FordB/   (UCR/UEA, official splits)
data/raw/FD-A/   train.pt  val.pt  test.pt                       (TF-C processed data)
data/raw/FD-B/   train.pt  val.pt  test.pt
```

- **UCR/UEA datasets:** `python dev/0_fetch_data.py` downloads and verifies them
  (`--verify-only` checks a copied-in course folder instead). FordB is fetched but not used.
- **FD-A and FD-B:** the processed files from the TF-C authors on Figshare
  (FD-A article `19930205`, FD-B article `19930226`). The article-level download link is behind a
  bot check, so download the three files of each article individually; the Figshare API
  (`https://api.figshare.com/v2/articles/<id>`) lists each file's URL and MD5 checksum.
  `dev/9_tfc_fda_fdb.py` then links them into the TF-C checkout as `datasets/FD_A` and
  `datasets/FD_B`, where TF-C's `main.py` expects them.

## Reproducing each result

Run the steps in this order. Steps 3 and 4 must run before 5–8, since checkpoints and
representations are not tracked.

| # | Command | Writes | Runtime (RTX 4500) |
|---|---|---|---|
| 1 | `python dev/0_fetch_data.py` | `data/raw/` | a few minutes |
| 2 | `python dev/1_prepare_data.py` | `data/processed/` (training-split scalers), `results/label_subsets/` (10% indices) | about a minute |
| 3 | `python -u dev/3_pretrain_tloss.py` | `results/checkpoints/tloss_*.pth`, `results/runs/tloss_pretrain.csv` | about 10 min per run, 9 runs |
| 4 | `python -u dev/4_pretrain_ts2vec.py` | `results/checkpoints/ts2vec_*.pth`, `results/runs/ts2vec_pretrain.csv` | under a minute per run, 9 runs |
| 5 | `python dev/5_extract_representations.py` | `results/representations/`, `results/runs/extract.csv` | a few minutes |

Then each result:

| Result in the report | Command | Output |
|---|---|---|
| Part B: raw-signal baselines (`tab:baselines`) | `python dev/2_baselines_raw.py` | `results/metrics/baselines_raw.csv` |
| Part C: frozen-representation results (`tab:ssl-clean`) | `python dev/6_probe_eval.py` | `results/metrics/ssl_probe.csv` |
| Part C: training time and representation size (`tab:cost`) | steps 3 and 4 above | `results/runs/tloss_pretrain.csv`, `results/runs/ts2vec_pretrain.csv` |
| Part D: clean versus dropout (`tab:dropout`, `tab:dropout-f1`) | `python -u dev/7_corruption_eval.py` | `results/metrics/corruption.csv` |
| Part E: masking ablation (`tab:masking-ablation`) | `python -u dev/8_ts2vec_mask_ablation.py` | `results/metrics/mask_ablation.csv`, `results/runs/ts2vec_mask_pretrain.csv` |
| Part E: exploratory dropout sweep (not in the report) | `python -u dev/8b_ts2vec_mask_sweep.py` | `results/metrics/mask_sweep.csv` |
| Part F: TF-C execution record and result (`tab:tfc-record`, `tab:tfc-result`) | `python -u dev/9_tfc_fda_fdb.py` | `results/tfc/execution_record.json`, with the stage logs `pre_train.log` and `fine_tune_test.log` beside it (in the hand-in; not tracked in git) |
| Per-seed table (`tab:per-seed`) | from the CSVs above | `results/metrics/*.csv` |
| Waveform figures (presentation) | `python eda/slide_waveforms.py` | `results/figures/waveforms_*.png` |

The report's tables give mean ± sample standard deviation over the three seeds of the rows in
these CSV files; each row is one (method, dataset, probe, label regime, seed). Every row also
records the selected hyperparameters, the number of CV folds and whether the cell could be
tuned.

Most scripts take `--datasets`, `--seeds` or `--methods` to run a subset, and `--help` lists
the options. Pretraining scripts skip checkpoints that already exist unless `--force` is given.

## Reproducibility notes

- **Seeds:** 42, 123 and 456. Within a seed, every method shares the 10% label subset, the CV
  folds, the encoder initialisation and the corrupted test set.
- **Not bitwise reproducible:** training is seeded, but GPU arithmetic differs between devices.
  Retraining on a different GPU gives slightly different encoders, and numbers that differ in the
  third decimal. The conclusions were unchanged when the encoders were retrained on a second
  machine.
- **Test sets** are used once, for the final score, and never for pretraining, scaling or
  tuning. Part F is the exception: it runs TF-C's own procedure, which evaluates the test set
  after every epoch and reports the best epoch.
- **Departures from the official code** are listed in the report's compliance section. The TF-C
  checkout runs unchanged through `src/methods/_tfc_launch.py`, which only restores the
  `np.float` alias removed in NumPy 1.24.

## Repository layout

```
src/
  config.yaml            Paths, seeds, label regimes, probe grids, corruption settings
  dataio/                Official split loading, stratified 10% subsets, train-only scaling
  corruption/            Contiguous sensor dropout (Part D)
  methods/               Adapters for the external T-Loss, TS2Vec and TF-C checkouts
  representations/       Raw-signal flattening
  probes/                Logistic regression and RBF-SVM probes, train-only tuning
  evaluation/            Accuracy, macro-F1, clean-versus-corrupted scoring
  utils/                 Seeding and config loading
dev/                     Numbered scripts, one per step (see above)
eda/                     Dataset notebooks and the presentation figures
data/                    Not tracked; see above
results/
  label_subsets/         Saved 10% indices per (dataset, seed)
  metrics/               Per-seed results
  runs/                  Runtime and device of each pretraining and extraction run
  tfc/                   Part F execution record
  figures/               Presentation figures
  checkpoints/           Pretrained encoders (not tracked)
  representations/       Cached encodings (not tracked)
solution/                LaTeX write-up and PDF
```
