# Experimental protocol: notes for the write-up

Written 6 October 2026 as input for the protocol section of `solution/solution.tex`. Every
point was checked against the code and the result files: `src/config.yaml`,
`src/probes/probes.py`, `src/dataio/subsets.py`, `src/evaluation/metrics.py` and
`results/`.

Section 1 corrects the first draft of the protocol, section 2 lists what that draft left
out, and section 3 is a cleaned-up version to adapt.

## 1. Corrections to the first draft

- **"5-fold cross-validation":** it is *stratified* 5-fold CV on the training labels only,
  used only to choose probe hyperparameters, and the number of folds shrinks when a class
  is small. Epilepsy at 10% uses 3 folds. AWR at 10% is not tuned at all, since it has
  about one label per class, so it uses fixed defaults (C = 1, γ = scale).
- **"Three seeds for CV and 10% label sets":** the seed controls more than that. Each seed
  (42, 123, 456) also sets:
  - the SSL encoder's initialisation and training randomness, so there are three
    separately pretrained encoders per method and dataset;
  - the corruption pattern, which is identical for every method under a given seed;
  - the probe's own random state.

  Within a seed, every method uses the same label subset, folds and corrupted test set.
  That is what makes the comparisons paired.
- **"No test set leakage or reporting only best result":** true for Parts B–E, where each
  test set is used once, for the final score. Not true for Part F: TF-C's supplied
  procedure evaluates the test set every epoch and reports the best one. It was used as
  shipped, so state it as the exception.
- **"Mean and std of the latter":** both accuracy and macro-F1 are reported as mean ±
  standard deviation, and it is macro-F1, not plain F1. The standard deviation is the
  sample standard deviation over the three seeds (`ddof=1`).
- **"Masking ablation for robustness on corrupted data":** it compares clean *and*
  corrupted performance. It runs on FordA only, with 100% labels and logistic regression.
- **"Transfer learning with TF-C":** this is not part of the project's own protocol. It is
  an exploratory run of the released TF-C code with its own procedure, one seed (42), and
  an encoder that is fine-tuned, not frozen.

## 2. Missing from the first draft

- **Preprocessing:** per-channel standardisation, fitted on the full training split
  without labels. The raw signal is then flattened.
- **SSL pretraining:**
  - pretrained on all training inputs without labels, never on test inputs;
  - published default settings (T-Loss 1,500 steps; TS2Vec 600 iterations, 200 on
    Epilepsy);
  - encoder frozen after pretraining, giving one 320-dimensional vector per series
    (TS2Vec with `full_series` max-pooling).
- **Representation scaling:** the representations are standardised, fitted on the training
  representations, before the probe.
- **Probes:** logistic regression is the principal probe (linear); the RBF-SVM is reported
  separately. The grids are C ∈ {0.01, 0.1, 1, 10, 100} for logistic regression, and
  C ∈ {0.1, 1, 10, 100} with γ ∈ {scale, 0.001, 0.01, 0.1} for the RBF-SVM.
- **10% subsets:** stratified, made once per seed, reused by every method, indices saved in
  `results/label_subsets/`.
- **Corruption:** fixed before any result was seen.
  - 20% of each series removed in one contiguous interval;
  - the same interval on all channels;
  - filled with the training channel mean;
  - test set only;
  - the probe is fitted on clean training data, then scored on both clean and corrupted
    test sets.

## 3. Suggested protocol

### Pipelines

1. **Raw baseline (Part B):** standardised, flattened raw signal → probe, with 100% and
   10% of the training labels.
2. **SSL representations (Part C):** encoder pretrained on all training inputs without
   labels → frozen → one 320-dimensional vector per series → standardised → probe, with
   100% and 10% of the labels.
3. **Robustness (Part D):** the 100%-label models from pipelines 1 and 2, scored on the
   clean test set and on a corrupted copy (one contiguous 20% interval per series, all
   channels, filled with the training mean).
4. **Masking ablation (Part E):** TS2Vec on FordA pretrained with binomial (default) and
   with continuous masking. Everything else is matched (encoder, iterations, seeds, probe,
   100% labels). Both are scored on clean and corrupted test sets.
5. **TF-C (Part F):** the released TF-C code run unchanged, pretraining on FD-A and
   fine-tuning and testing on FD-B. One seed, with its own procedure: encoder fine-tuned,
   best test epoch reported.

### Common to pipelines 1–4

- **Seeds:** 42, 123 and 456. Within a seed, all methods share the label subset, CV folds,
  encoder initialisation and corrupted test set.
- **Probes:** logistic regression (principal) and RBF-SVM, reported separately.
- **Tuning:** stratified 5-fold CV on the training labels, with fewer folds when the
  smallest class has fewer than five examples, and fixed defaults when a class has one
  example.
- **Test sets:** used once, for the final score. Never used for pretraining, scaling or
  tuning.
- **Metrics:** accuracy and macro-F1, both as mean ± sample standard deviation over the
  three seeds. Per-seed values are in `results/metrics/`.
