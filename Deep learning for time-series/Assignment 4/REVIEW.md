# Review of `solution/solution.tex`

Reviewed 2 October 2026 on the RTX 4500 machine. Claims were checked against the result
CSVs in `results/` and against the three checkouts. Nothing in the document was edited.

**Summary:** the tables are all correct and consistent with the CSVs. The problems are in
Part A (a few factual errors and stale statements), in two Part G conclusions, and in
sections that are still empty.

Line numbers refer to `solution/solution.tex` as it was at the time of the review. They
will drift once the file is edited, so fix from the bottom of each table upwards, or search
for the quoted text.

## 1. Factual errors

| Line | What it says | What is true |
|---|---|---|
| 423 | All TF-C datasets are reduced to one channel and length 178 | One channel, yes. The length is `TSlength_aligned`, which is 5120 in the FD-A config; 178 is the SleepEEG value. It contradicts Part F. |
| 410 | "I cannot check the number of samples without loading the files" | Stale. Part F now has them: 640 of 8,184 FD-A, 60 FD-B train, 13,559 FD-B test. |
| 180 | `_eval_with_pooling` is called by `TS2Vec.fit()` | It is only called by `TS2Vec.encode()`. |
| 216 | `temporal_contrastive_loss` ↔ Equation (2) | Line 258 says Equation (1), and instance ↔ (2). The two places disagree; line 258 is the consistent one. |
| 300 | The FFT "is called in `TFC.forward()`" | The FFT is computed once in `Load_Dataset.__init__`; `forward` only receives the result. |
| 119 | `CausalCNNEncoder.forward` "wraps the whole classification pipeline" | It is the encoder (convolutions, pooling, linear layer). No classification happens there. |
| 316 | `__init__` defines "the four encoders" | Two encoders and two projectors. Item (i) there also describes the code where the paper location should be. |
| 341 | `model_finetune` "does the same for the test datasets" | It trains on the FD-B training set; `model_test` handles the test set. |
| 42, 47 | T-Loss series are zero-padded "if they are too short" | The causal padding is applied in every convolution, whatever the length. |
| 480 | FordA: "same timestamps in channels" | FordA has one channel. The point is the same timestamps across series. |

Smaller imprecisions in Part A:

- **Line 46:** the T-Loss objective is a logistic loss on dot products, not a comparison of
  distances, and it is a "triplet" loss.
- **Line 58:** "same view, no masking" is wrong, since both views are masked.
- **Line 61:** "dilation of 2" should be max-pooling with kernel 2.
- **Line 78:** the released TF-C code truncates to the configured length; it does not
  zero-pad.
- **Line 155:** "the anchor itself can contain a negative sample" presumably means a
  negative can be drawn from the anchor's own series.
- **Line 249:** the sentence about the `while` loop and Algorithm 1 belongs under
  `hierarchical_contrastive_loss`, not under the mask generator.
- **Lines 311 and 374** describe the frequency augmentation differently ("amplifies others"
  and "halved"). Both follow from the code: the two augmented copies are summed, so kept
  bins are doubled and "removed" bins stay at 1×. One consistent description would be
  better.

## 2. Conclusions and interpretations

### "The effect of SSL is deemed clearer with only 10% labels" (line 806)

Only half supported. It holds for the RBF-SVM, and it does not hold for logistic
regression, which the document names as the principal probe.

| Gain over raw (accuracy) | 100% labels | 10% labels |
|---|---|---|
| FordA, logistic regression | +0.43 | +0.40 |
| FordA, RBF-SVM | +0.10 | +0.25 |
| Epilepsy, logistic regression | +0.33 | +0.30 |
| Epilepsy, RBF-SVM | +0.11 to +0.13 | +0.14 to +0.19 |
| AWR, logistic regression | −0.01 to +0.01 | −0.01 to +0.01 |

It also sits awkwardly beside Part C (line 556), which says the drop to 10% is no smaller
than for the raw baseline. The statement the numbers support is "clearer for the nonlinear
probe, unchanged for the linear probe".

### Other points

- **"AWR: representations worsen the results" (line 802) is too strong.** With logistic
  regression T-Loss is 0.013 lower and TS2Vec 0.010 higher, both inside the seed spread
  (0.02–0.05). Only the RBF-SVM cell is lower (0.16–0.18 against 0.23), and that cell is
  untuned for all three representations.
- **"Epilepsy is between FordA and AWR" (line 812) does not match Part C.** Epilepsy has
  the largest drop to 10% (0.23 against 0.15 for AWR) and similar seed spread. It has the
  fewest labels (13), though more per class than AWR (about 3 against 1). The same
  paragraph uses series length as an explanation without a mechanism, and it does not
  address dimensionality (1, 9 and 3 channels), which the heading asks about.
- **Epilepsy baseline comments (line 484) are stated as fact.** "Beats chance only because
  the mean level differs" and "classes differ in amplitude" were not tested in anything in
  the repo. The FordA explanation is backed by the circular-shift experiment; these need a
  "likely" or a check.
- **The AWR baseline comment (line 482) skips the most striking number in the table.**
  "Same as logistic regression" is true at 100% labels, but at 10% the RBF-SVM scores 0.23
  against 0.83. The reason (untuned defaults with about one label per class) is in the
  protocol and should be in the comment.
- **Part D, "several short gaps would be a much easier problem" (line 590) is contradicted
  by the sweep.** At the same total, 5 blocks hurt TS2Vec on FordA more than 1 block
  (−0.040 against −0.028 at 20%; −0.096 against −0.057 at 40%). Present it as the
  assumption made beforehand, or drop it.
- **Part D discussion (lines 655–662) is thin on three points.**
  - "Classifier: not shown in the table, but there is no difference" rests on numbers the
    reader cannot see. The RBF rows are in `results/metrics/corruption.csv` and could go in
    the appendix.
  - The FordA logistic-regression increase (+0.014) is noise around chance and should be
    labelled so.
  - The SSL representations keep a much higher absolute accuracy after corruption, and the
    differences in Δ are within the seed spread. Both are worth stating.
- **Part G, first subsection (line 797) answers only half the heading.** It does not say
  where raw nonlinear classification stays competitive: on AWR, where the raw RBF-SVM
  reaches 0.98.
- **The stability subsection (line 809)** is fair, but should mention that three seeds is
  few, and that the spread at 10% mixes two sources (label subset and encoder seed). At
  100% the raw rows have spread 0 because nothing in them depends on the seed.

## 3. Fidelity to the process

- **Part E setup (line 669) says too little for the design to be judged.** Missing are:
  - the mask parameters (binomial p = 0.5; continuous 5 runs of 10% of the crop);
  - that seeds, initial weights, 600 iterations and the probe are matched;
  - that the binomial arm is the Part C encoders;
  - the two confounds (about 41% against 50% coverage; hidden-layer masking against
    input-level corruption).
- **The sweep and the second-machine replication are not mentioned.**
  `results/metrics/mask_sweep.csv` is in the repo, but the word "sweep" does not appear in
  the document. The RTX 5060 run retrained every encoder and gave the same Part E
  conclusion (Δ −0.030 for both arms). Both strengthen "neither".
- **Part F's collapse numbers have no stated source.** The sentence saying they come from a
  separate diagnostic script is gone, and the scripts exist only in a temporary folder on
  the RTX 4500 machine, not in the repo. As it stands a grader cannot reproduce the
  1.27 → 0.002 figures or the seed-0 result.
- **Part F's command row** shows `python main.py …`. The actual run went through the
  launcher (`src/methods/_tfc_launch.py`) with an absolute log path. The text below the
  table explains it, so this is minor.
- **Parts B and C never state the pretraining settings.** T-Loss 1,500 steps; TS2Vec
  600/600/200 iterations; pretraining on the full unlabelled training set; `full_series`
  pooling; and "seeded but not bitwise reproducible", which now has a concrete example from
  the second machine.
- **The pretraining-cost table (lines 562–579) is empty, and the data for it is
  incomplete.** TS2Vec is fully logged (about 7 to 37 s per run). T-Loss has only 7 rows
  from the RTX 4500 (about 9 to 11 min each), with FordA seeds 42 and 123 missing. All nine
  T-Loss runs were logged on the RTX 5060 (about 21 to 27 min each); those rows are in git
  history at commit `1790f90`. Both methods give 320 dimensions.
- **Leftover template prompts** remain as body text at lines 85–86, 406–408 and 586–587.
- **The discrepancy list at lines 414–424** is nested under the "test influences the best
  results" bullet. It needs its own lead-in.
- **"Following one TF-C batch" (lines 354–362)** lists six generic steps with no tensor
  shapes.

## 4. Still empty

- "Implications for future multichannel industrial sensor data" (lines 823–826) has an
  empty bullet. One relevant fact from this work: the TF-C code keeps only the first
  channel of every dataset.
- All nine compliance notes, "Limitations and conclusions", the per-seed appendix, and the
  two optional extensions (which should be removed or marked as not done).
- `\listoftodos` is still printed at the top.

## 5. Typos and grammar

| Line | Fix |
|---|---|
| 42 | "positiv" → positive; "which sub-series" → while |
| 46 | "triple loss" → triplet |
| 47 | "dot product are" → is |
| 48 | "avalable" |
| 55 | "to produced"; "After and input" → an |
| 59 | "both view" → views |
| 74, 269 | "agumented" |
| 79 | "ocnnected" |
| 107 | "where is is called" |
| 108 | "achitecture" |
| 138 | "to sue" → use; "calls is" → it |
| 142 | "can be training" → trained |
| 151 | "wether" |
| 198, 211, 219 | "reprensentation(s)" |
| 258 | "hierarchical\_contastive\_loss" |
| 269 | "two version" |
| 279 | "if run" → If |
| 280, 341 | "mode\_finetune", "mode\_test", "mode\_pretrain" → model\_ |
| 306 | "Sectio" |
| 316 | "augmentments"; "is training during" → trained |
| 329 | "postive" |
| 349 | "pre-training of fine-tuning" → or |
| 373 | "multiple different of which a radnom" |
| 374 | "The two augmentation are" |
| 385 | "transofmr" |
| 401, 402 | "repesentations" |
| 411 | "train.py" → train.pt |
| 419 | "aguments" |
| 421 | "Pertubations" |
| 428 | "torch.cude" → cuda |
| 436 | "appied" |
| 480 | "phse" |
| 694 | "difference binomial and continuous" → between |
| 770 | stray comma after the commit hash |
| 802 | "Representations worsens"; "atrributed" |
| 815 | "Neither are" → is |
