# Data

Nothing in `raw/` or `processed/` is tracked in git. Populate `raw/` with `dev/0_fetch_data.py`
(or by copying in the course-supplied `data` folder), then `dev/1_prepare_data.py` writes `processed/`.

Use the **official** training and test partitions throughout. Every official test set stays
held out: no test input is used for SSL pretraining, and normalisation/imputation/feature
transforms are fit on training data only.

## Expected layout

```
raw/
  FordA/                      3,601 train / 1,320 test,  500 x 1,  2 classes
  ArticularyWordRecognition/    275 train /   300 test,  144 x 9, 25 classes
  Epilepsy/                     137 train /   138 test,  206 x 3,  4 classes
  FordB/                      3,636 train /   810 test,  500 x 1,  2 classes  (fetched, not used)
  FD-A/                       train.pt / val.pt / test.pt: 8,184 / 2,728 / 2,728 x 5,120 x 1
  FD-B/                       train.pt / val.pt / test.pt:    60 /    21 / 13,559 x 5,120 x 1
processed/
```

Sources: UCR/UEA Time Series Classification Archive for the first four; the TF-C authors'
processed files on Figshare for FD-A and FD-B.

## FD-A / FD-B

Used only by the TF-C experiment (Part F), and not fetched by `dev/0_fetch_data.py`. Download
the TF-C authors' processed files from Figshare (FD-A article `19930205`, FD-B article
`19930226`) into `raw/FD-A/` and `raw/FD-B/`. The article-level download link is behind a bot
check, so download the three files of each article individually; the Figshare API
(`https://api.figshare.com/v2/articles/<id>`) lists each file's URL and MD5 checksum.

TF-C's unchanged `main.py` reads its data from `datasets/FD_A` and `datasets/FD_B` inside its
own checkout, a path it does not let you change. `dev/9_tfc_fda_fdb.py` therefore creates
those two entries as directory links (junctions on Windows) pointing at `raw/FD-A/` and
`raw/FD-B/`; the checkout's `.gitignore` already ignores `datasets/`. See `tfc_data` in
[../src/config.yaml](../src/config.yaml). FD-A labels are not used during pretraining.

## Caveats to respect in the write-up

- `Epilepsy` here is the UEA motion dataset: simulated seizure-like movement by healthy
  participants, **not** a clinical seizure-detection dataset. It is also distinct from the
  `Epilepsy` folder inside the TF-C checkout.
- `ArticularyWordRecognition` carries no speaker identifiers, so its test score must not be
  described as verified speaker-independent generalisation.
- Do not concatenate FordA and FordB or treat them as paired instances.
