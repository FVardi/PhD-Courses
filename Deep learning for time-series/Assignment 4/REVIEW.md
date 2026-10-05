# Submission status check

Checked 5 October 2026 against the assignment PDF (§5 tasks, §6 experimental rules,
§8 submission, §9 expected result tables), with the slide presentation excluded. Claims
were checked against the result CSVs, the label-subset files and the three checkouts. This
replaces the review of 2 October, most of whose points have since been fixed.

Line numbers refer to `solution/solution.tex` at commit `f4ae2bd`.

**Summary:** the written answers are complete; the reproducibility deliverables are not.
Every Part A–G question has an answer, every number in the tables matches the CSVs
(including the per-seed table), and the experimental rules are followed. Still missing:
the README is the original placeholder, the training-time table is empty, and
`dev/10_aggregate_results.py`, `tests/` and `results/figures/` are empty.

## §8 Submission

| Item | Status |
|---|---|
| 1. Presentation (slides excluded): method comparison, paper-to-code map, protocol, tables, interpretation, limitations, conclusions | Done in `solution.tex`, apart from plots: §8.1 asks for "principal result tables and plots", and `results/figures/` is empty. They may belong in the slides. |
| 2. Reproducible code for baselines, pretraining, extraction, probes, corruption, ablation | Done: `dev/2_` to `dev/8_`, plus `8b_` and `9_`. |
| 3. README with environment setup and a command per result table or figure | **Not done.** See below. |
| 4. Machine-readable per-seed results, configurations, label-subset indices | Done. All metric CSVs are tracked, `src/config.yaml` holds the configuration, and the 9 label-subset JSON files hold the indices and class counts. The two `*_smoke20.csv` files are also tracked; they are not results and could be removed. |

What is wrong with `README.md`:

- It still says "Status: structure only — no implementation yet".
- It lists the checkouts as "not yet cloned", under the old folder name
  (`Time series research`).
- The command table is `_TBD_`, and the environment section says "to be recorded".
- It describes `tests/`, `results/tables/` and `results/figures/` as holding content, but
  they are empty, and `dev/10_aggregate_results.py` is still a stub.

## §9 Expected result tables

| Required | Status |
|---|---|
| Clean-test results for every method and label regime | Tables `tab:baselines` and `tab:ssl-clean` |
| Clean-versus-dropout table for the 100%-label models | Tables `tab:dropout` and `tab:dropout-f1` |
| FordA standard-versus-continuous ablation table | Table `tab:masking-ablation` |
| FD-A → FD-B execution record and classification result | Tables `tab:tfc-record` and `tab:tfc-result` |
| Brief record of training time and representation dimensionality | **Missing.** `tab:cost` is empty, with a todo note at line 577. See below. |
| Per-seed results in supplementary material, mean ± sd in the main text | Done: per-seed table and CSVs |

Data available for `tab:cost`, from `results/runs/` (RTX 4500):

- TS2Vec: about 24–37 s per run on FordA, 18–19 s on AWR, 7 s on Epilepsy.
- T-Loss: about 525–590 s per run on AWR, 540–560 s on Epilepsy, 646 s on FordA seed 456.
  FordA seeds 42 and 123 have no timing row on this machine.
- All nine T-Loss runs were timed on the RTX 5060 (about 21–27 min each); those rows are
  in git history at commit `1790f90`.
- Both methods give 320-dimensional representations.

## §6 Experimental rules

All nine are followed, and the compliance section says so accurately.

- **Test sets held out:** yes. The one exception is Part F, where the supplied code
  evaluates the test set every epoch; the write-up states it.
- **Training-only normalisation:** yes. The claim about the T-Loss loaders is right:
  `ucr.py` lines 102–103 compute mean and variance over train and test together, and they
  are not used.
- **Stratified 10% subsets with saved indices:** yes. For example, AWR seed 42 has 27
  indices covering all 25 classes.
- **Same seeds, subsets, corruptions and probe settings:** yes. The clean columns matched
  across Parts B–E with zero difference.
- **Recording software versions is only partly met.** Versions are recorded only in
  `results/tfc/execution_record.json`. Parts B–E list device and runtime per run but no
  versions. Saying in the README or the compliance note that one environment was used
  throughout (Python 3.13.11, PyTorch 2.11.0 + CUDA 12.8, NumPy 2.4.2, scikit-learn 1.8.0)
  would close this.

## Parts A–G: gaps

- **Part A, the comparison asks "what information the objective encourages the encoder to
  preserve".** The "Objective" bullets describe the losses rather than what is preserved,
  and nothing compares the three methods side by side. A sentence per method, or a small
  table, would answer it.
- **Part A, the CPU-only question asks to "explain any failure observed".** The answer says
  the script "will crash". Reproduced on the RTX 4500 machine with the GPU hidden
  (`CUDA_VISIBLE_DEVICES=-1`): `torch.cuda.FloatTensor` raises
  `RuntimeError: No CUDA GPUs are available`. Quoting that makes it an observation rather
  than a prediction.
- **Part A, the call sites:** the assignment asks for the call site when a `forward` is
  invoked through `module(...)` or `nn.Sequential`.
  - `CausalConvolutionBlock.forward` runs through the `nn.Sequential` inside `CausalCNN`,
    so name that.
  - `CausalCNNEncoder.forward` is also called from inside `TripletLoss.forward`, not only
    from `encode`.
- **Part B's protocol does not mention the seeds or the hyperparameter grids.** Both are in
  Part C; a forward reference would do.
- **Part C says "only logistic regression … is discussed",** but the assignment asks for a
  comparison with *both* raw baselines. The comparison with the raw RBF-SVM is in Part G;
  one bullet in Part C would cover it.
- **Part F, the pretraining loss of 8.18 cannot be checked from tracked files.** The stage
  logs are gitignored, and `execution_record.json` does not hold the loss curve. The
  command row also shows `--logs_save_dir results/tfc`, but the run used an absolute path;
  run from `code/TFC`, the relative path would point inside the checkout.
- **Part G, "Implications for future multichannel industrial sensor data" is thin on the
  multichannel part.** Two points from the results bear on it: AWR is the only
  multichannel dataset, and the TF-C code drops every channel but the first.

## Small items

- Line 67: missing full stop after "domains".
- Line 665: the old sentence "Mask configurations are the standards…" now duplicates the
  bullets below it.
- `\listoftodos` is still printed at the top.
- "Per-seed results" no longer sits under `\appendix`, so it is numbered as a main section.
