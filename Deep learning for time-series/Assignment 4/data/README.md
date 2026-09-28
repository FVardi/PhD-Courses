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
  FordB/                      3,636 train /   810 test,  500 x 1,  2 classes  (optional extension)
processed/
```

Sources: UCR/UEA Time Series Classification Archive.

## FD-A / FD-B

Not stored here. The TF-C experiment (Part F) reads its processed FD-A and FD-B directly from
`datasets/` inside the external, unchanged TF-C checkout — see `checkouts.tfc` in
[../src/config.yaml](../src/config.yaml). FD-A labels must not be used during pretraining.

## Caveats to respect in the write-up

- `Epilepsy` here is the UEA motion dataset: simulated seizure-like movement by healthy
  participants, **not** a clinical seizure-detection dataset. It is also distinct from the
  `Epilepsy` folder inside the TF-C checkout.
- `ArticularyWordRecognition` carries no speaker identifiers, so its test score must not be
  described as verified speaker-independent generalisation.
- Do not concatenate FordA and FordB or treat them as paired instances.
