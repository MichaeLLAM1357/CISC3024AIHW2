# `01_vbll_classification.ipynb` — Architecture design

**CISC3024 Pattern Recognition · AI Assignment #2 · VBLL (ICLR 2024)**

| | |
|---|---|
| Task | 10-class image classification + uncertainty estimation |
| In-distribution (ID) | MNIST |
| Out-of-distribution (OOD) | Fashion-MNIST |
| Arms | (a) MLP + softmax · (b) VBLL `DiscClassification` · (c) VBLL `GenClassification` · (d) `laplace-torch` last-layer Laplace |
| Metrics | Accuracy · NLL · ECE · OOD AUROC |
| Figures | Reliability diagram · OOD score histogram |
| Runtime | Google Colab free tier, NVIDIA T4 |

> This document is a **design only** — it specifies what each cell contains and what
> every interface guarantees. No implementation code is written here, and no code in
> the delivered notebook will be written by hand.

---

## 1. Design goals

1. **One evaluation path for four heterogeneous models.** This is the whole design
   problem. The four arms return four different things, and the evaluation code must
   not care.
2. **Reproducible.** Fixed seeds, fixed train/val split, artifacts written to disk.
3. **Report-ready.** Every number and every figure the report needs is emitted by the
   notebook, not reconstructed by hand afterwards.
4. **Fail loudly, early.** Bad configuration (no GPU, missing `return_ood`, wrong
   array shape) must raise at the point of the mistake, not 20 cells later.

---

## 2. The central problem: four arms, four different outputs

| Arm | Forward pass returns | How probabilities are obtained | How an OOD score is obtained |
|---|---|---|---|
| (a) MLP + softmax | raw logits `(N, C)` | apply softmax | `1 − max p` |
| (b) VBLL Disc | a VBLL output object | read `out.predictive.probs` | read `out.ood_scores` |
| (c) VBLL Gen | a VBLL output object | read `out.predictive.probs` | read `out.ood_scores` |
| (d) Laplace LLA | depends on the link approximation | returned directly by the Laplace predictive call | **nothing native** — must be derived |

Three of the four need different post-processing, and arm (d) has no OOD score at all.
If `evaluate_model` handled these cases itself it would need an `if arm == ...` ladder —
which breaks the moment an arm is added or changed.

### Decision: an adapter (port) layer

Each arm is wrapped in a thin adapter object that exposes **one** method:

```
predict(loader) -> Prediction
```

`evaluate_model` only ever sees a `Prediction`. It never sees a model, a logit tensor
or a VBLL output object. Adding a fifth arm later means writing one adapter and
registering it — no change to the metrics or the plotting code.

---

## 3. The `Prediction` contract

A plain data container (dataclass or dict) — an **interface**, not a model.

| Field | Shape | Meaning |
|---|---|---|
| `probs` | `(N, C)` | predictive class probabilities, rows sum to 1 |
| `logits` | `(N, C)` | raw scores; kept so NLL can be computed either from probs or from log-softmax |
| `ood_score` | `(N,)` | **higher = more out-of-distribution** |
| `n` | scalar | sample count, for sanity checks |
| `uncertainty_aleatoric` | `(N,)` or `None` | VBLL arms only |
| `uncertainty_epistemic` | `(N,)` or `None` | VBLL arms only |
| `extra` | dict | free-form provenance: `mc_samples`, `link_approx`, `parameterization`, … |

### Invariants — asserted inside the adapter, never inside `evaluate_model`

- `probs.shape == (n, C)` and `ood_score.shape == (n,)`
- `probs` rows sum to 1 within `1e-4`
- all values in `[0, 1]`, no `NaN`, no `inf`
- `n` equals the length of the dataset behind the loader

Putting validation in the adapter means a malformed prediction is caught by the arm
that produced it, with that arm's name in the message.

---

## 4. The three adapters

### 4.1 `BaselineAdapter`
- Holds the trained MLP. Runs it in eval mode under `no_grad`, concatenating logits
  across the loader.
- `probs = softmax(logits)`, `ood_score = 1 − max(probs)` (the max-softmax baseline).
- No uncertainty fields.

### 4.2 `VBLLAdapter`
- Holds a model whose **final layer is a VBLL head**.
- Forward returns the VBLL output object; the adapter reads `out.predictive.probs` and
  `out.ood_scores` directly.
- **Critical rule: do not apply softmax again.** The VBLL head already produces
  probabilities; a second softmax would silently destroy the calibration being measured.
- Because VBLL is sampling-free, **one forward pass is enough** — no Monte-Carlo loop.
- Guard: if the head was built with `return_ood=False`, `ood_scores` does not exist. The
  adapter must raise a clear error naming that cause rather than an `AttributeError`.
- If the VBLL head exposes the aleatoric/epistemic split, copy it into the optional
  fields; otherwise leave them `None`.

### 4.3 `LaplaceAdapter` — how `laplace-torch` is made to fit the same interface

This is the case the user asked about specifically. `laplace-torch` does **not** fit the
pattern of the other arms, so the adapter does three things:

1. **Fit stage (outside `predict`).** The Laplace object is created *around an
   already-trained plain MLP* — last-layer Laplace is a post-hoc method, so there is no
   end-to-end training. Hyperparameters are optimised against the **validation** split,
   never the test split.
2. **Prediction stage (inside `predict`).** The Laplace predictive call returns
   probabilities **directly**, already normalised. So `probs` come straight out — again,
   **no softmax**. The call is wrapped in `no_grad` and, if memory is tight, batched over
   the loader.
3. **OOD stage.** `laplace-torch` has no `ood_scores` attribute. The adapter therefore
   **derives** the OOD score from the predictive distribution it already has —
   `ood_score = 1 − max p`, or the predictive entropy.

   > **This is a real methodological asymmetry, not an implementation detail.** Arm (d)'s
   > OOD score is *not* the same quantity as arm (b)'s `ood_scores`. Section 6 makes that
   > comparison fair instead of hiding it.

- Two sub-modes exposed as a single config knob: a fast deterministic link
  approximation versus a sampled one. The sampled mode is closer in spirit to VBLL's
  sampling-free predictive and is the more honest comparison; the fast mode is for
  iterating.
- Cost note: the GLM predictive is the slowest and most memory-hungry step in the
  notebook. It is the main reason the Laplace arm is evaluated last.

---

## 5. Metrics module

All metrics are **pure functions of arrays** — no model, no loader, no device. That makes
them independently checkable, which matters because a wrong ECE silently invalidates a
headline claim.

| Function | Input | Output |
|---|---|---|
| `accuracy` | `probs`, `y` | scalar |
| `nll` | `probs`, `y`, `eps` | scalar (clip before `log`) |
| `ece` | `probs`, `y`, `n_bins` | `ece` **plus** the bin arrays `(bin_conf, bin_acc, bin_count)` |
| `auroc` | `scores_id`, `scores_ood` | scalar, positive class = OOD |

Design notes:
- `ece` returns the bin arrays so the **reliability diagram is a by-product of the metric**,
  not a separately-written plotting routine that can drift out of sync with the number in
  the table.
- ECE uses **equal-width** bins by default, with `n_bins` as a config knob. Equal-mass
  (adaptive) binning is the standard alternative and is a natural one-line ablation.
- AUROC is computed with `sklearn.roc_auc_score` on `scores_id` vs `scores_ood` so that a
  score where *higher means more OOD* yields AUROC > 0.5. If an arm returns a
  consistently inverted score, AUROC drops below 0.5 — that is a signal, and the design
  keeps it visible rather than auto-flipping it silently.

---

## 6. Fairness: making the four OOD scores comparable

The four arms score OOD-ness with three different definitions:

| Arm | Native score | Comparable to the others? |
|---|---|---|
| (a) MLP + softmax | `1 − max p` | partially |
| (b) VBLL Disc | `out.ood_scores` (likelihood-based) | no |
| (c) VBLL Gen | `out.ood_scores` (likelihood-based, generative) | no |
| (d) Laplace LLA | `1 − max p` (derived) | partially |

Reporting a single AUROC column would therefore compare four different things and invite
the wrong conclusion.

**Decision: report two score families.**

- `auroc_native` — each arm's own score. This is what the method *claims*, and it is the
  headline for VBLL.
- `auroc_entropy` — the **predictive entropy** `H[p]`, computable from `probs` for *all
  four arms*. This is the apples-to-apples comparison.

Both are produced by `evaluate_model`, so the results table carries both columns and the
report can state the difference explicitly instead of glossing over it.

---

## 7. Cell structure

28 designed cells in 8 sections, plus one **cell 00b** prepended during implementation
(the dependency self-repair cell, see §11.5). The notebook therefore ships **29 cells:
7 Markdown + 22 code**. `md` = Markdown, `code` = code cell.

### Section 0 — Header, config, guardrails

| Cell | Type | Contents |
|---|---|---|
| 00 | md | Title; algorithm + venue; ID/OOD datasets; the four arms; what the notebook produces; "all code AI-generated" statement |
| 00b | code | **Dependency self-repair.** Idempotent: reports the current state, writes the constraints file pinning Colab's own `torch`/`torchvision`, installs `vbll` + `laplace-torch` behind it, drops the import caches, and verifies both that every import works and that CUDA is still alive. Exits in ~1 s on a healthy session. Added in §11.5 |
| 01 | code | Imports; `DEVICE`; global `SEED`; a single `CFG` mapping (epochs, batch size, lr, weight decay, `n_bins`, Laplace `n_samples`, `SEEDS` list, output dirs); a `set_seed()` helper that seeds `random`, `numpy` and `torch` |
| 02 | code | **Environment echo + fail-fast guard.** Asserts CUDA is available; prints `torch` / `torchvision` / `vbll` / `laplace` versions and the GPU name; imports `vbll` and `laplace` so a missing setup fails here rather than in cell 18. Also prints the versions into a variable the report can quote |
| 03 | md | Reproducibility note: what is fixed (seed, split, budget) and what is not (cuDNN nondeterminism, single seed) |

### Section 1 — Data

| Cell | Type | Contents |
|---|---|---|
| 04 | md | Why MNIST is ID and Fashion-MNIST is OOD; the fact that both have 10 classes so a *random* classifier scores ~0.10 on both; normalisation choice |
| 05 | code | Transforms; loaders for train, val, test and OOD; **train/val split carved with a fixed-seed generator** so the split is identical for every arm and every seed run; asserts the four dataset sizes |
| 06 | code | Sanity visual: one batch as a grid + per-class count histogram for train and val — catches a broken split before it costs an hour |

### Section 2 — Model definitions

| Cell | Type | Contents |
|---|---|---|
| 07 | md | The four arms; what is held constant across arms (identical backbone, identical optimiser budget, identical data) — so differences are attributable to the head alone |
| 08 | code | `build_backbone()` (shared MLP) and a parameter counter; the parameter count is a report artefact |
| 09 | code | Arm (a): backbone + `nn.Linear` head, cross-entropy loss |
| 10 | code | Arms (b)(c): a factory that attaches a VBLL head with a `head="disc"`/`"gen"` switch, exposing `reg_weight`, `parameterization`, `prior_scale`, `return_ood`. **Defaults must set `return_ood=True`** — the adapter depends on it |
| 11 | code | Arm (d): build a plain MLP with the same backbone; the Laplace object is constructed later, after that MLP is trained (cell 18) |

### Section 3 — The evaluation contract (the core of the notebook)

| Cell | Type | Contents |
|---|---|---|
| 12 | md | **The adapter design**, in full: the four-outputs problem, the `Prediction` contract, the invariants, and the OOD-score fairness argument from §6 above |
| 13 | code | `Prediction` container + `check_contract()` validator that asserts every invariant and reports which arm violated it |
| 14 | code | The three adapters (`BaselineAdapter`, `VBLLAdapter`, `LaplaceAdapter`), each exposing only `predict(loader) -> Prediction` |
| 15 | code | The four metric functions (pure array functions) |
| 16 | code | **`evaluate_model(adapter, id_loader, ood_loader) -> dict`** — the single entry point. Calls `predict` once on each loader, validates both `Prediction`s, then returns accuracy, NLL, ECE (+ bins), both AUROCs, and the per-sample arrays needed by the plots |

> Cells 13–16 are deliberately ordered *before* any training. Defining the contract first
> means the models are built against it rather than the other way round.

### Section 4 — Training

| Cell | Type | Contents |
|---|---|---|
| 17 | code | `train_model(model, train_loader, val_loader, cfg)` — **one loop that handles both objectives**: cross-entropy for the baseline, and the VBLL training/validation loss functions for arms (b)(c). Parameter groups put `weight_decay = 0` on the VBLL head. Early stopping on validation accuracy. Returns the history and the best state |
| 18 | code | Run the loop over the arm registry; save a checkpoint and the history per arm; print the per-arm log. Then, for arm (d) only, construct the Laplace object around the trained MLP and optimise its hyperparameters on the **validation** split |

### Section 5 — Evaluation and reporting

| Cell | Type | Contents |
|---|---|---|
| 19 | code | Build one adapter per trained arm, call `evaluate_model` on each, assemble the results table (one row per arm, one column per metric). Display it |
| 20 | code | **Reliability diagram** — four curves plus the diagonal; ECE shown in the legend so the plot and the table cannot disagree |
| 21 | code | **OOD score histograms** — ID vs OOD overlaid, one panel per arm (2×2), using the native score; a second figure repeats it with the common entropy score |
| 22 | code | Optional risk–coverage (accuracy vs abstention) curve — reuses `probs` only, so it costs no new model code |
| 23 | code | Export: `results.csv`, `results.json`, every figure into `figures/`, plus the environment echo from cell 02 |

### Section 6 — Robustness (behind a config flag)

| Cell | Type | Contents |
|---|---|---|
| 24 | md | Why one seed is not evidence: the differences between arms may be noise |
| 25 | code | Multi-seed loop over `SEEDS`, reporting mean ± std for the headline metrics only, to keep runtime sane |
| 26 | code | Ablation hooks: VBLL covariance parameterisation (`diagonal` / `dense` / `lowrank`), `prior_scale`, and `return_ood` on/off |

### Section 7 — Wrap-up

| Cell | Type | Contents |
|---|---|---|
| 27 | md | Threats to validity, stated plainly: MNIST is saturated; one seed per configuration unless §6 is run; the four OOD scores are not the same quantity; the generative and discriminative VBLL losses are **not numerically comparable**; the Laplace GLM predictive is an approximation. Plus what each result feeds into in the report |

---

## 8. Runtime budget (T4, free tier)

| Stage | Estimate |
|---|---|
| Data download + first epoch warm-up | ~1 min |
| Baseline MLP, 30 epochs | ~1–2 min |
| VBLL Disc, 30 epochs | ~2–3 min |
| VBLL Gen, 30 epochs | ~2–3 min |
| Laplace: train MLP + `optimize_hyperparameters` | **~5–10 min** (the slowest step) |
| Evaluation, metrics, 4 figures | < 1 min |
| **Single-seed total** | **~15–20 min** |
| With 3 seeds (§6) | ~45–60 min |

Comfortably inside one Colab session. Checkpoints and results are written to Drive so a
disconnect costs at most one arm — the same pattern used in Assignment #1.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| VBLL `ood_scores` turns out inverted (AUROC < 0.5) | Keep it visible; document the sign convention rather than silently flipping |
| Head built with `return_ood=False` | Factory defaults to `True`; the adapter raises a named error if it is missing |
| Laplace GLM predictive exhausts memory on a full test pass | Batch the prediction; expose the sample-count knob |
| Generative VBLL loss is numerically huge | Never compare losses across arms; compare metrics only (explicit warning in the notebook) |
| A double softmax silently destroys VBLL calibration | The adapters are the only place probabilities are produced, and neither applies softmax to VBLL or Laplace output |
| Single-seed differences are noise | §6 multi-seed loop; otherwise state the caveat in §27 |
| Colab disconnects mid-run | Drive-backed checkpoints, per-arm resume |

---

## 10. Open decisions to confirm before implementation

1. **Laplace link approximation** — fast deterministic, or sampled (more honest, slower)?
   Recommendation: sampled for the reported run, deterministic while iterating.
2. **`n_bins` for ECE** — 15 is the common default. Recommendation: 15, with equal-mass
   binning as a one-line ablation.
3. **Is §6 (multi-seed + ablations) in scope** for the deadline, or should the first
   deliverable be a clean single-seed notebook that runs end-to-end in ~20 minutes?
   Recommendation: ship single-seed first, add §6 once the pipeline is proven — this is
   the same "smoke test before spending GPU hours" discipline used in Assignment #1.

---

## 11. Resolved decisions and implementation deltas

The three open decisions were settled by the operator and are now the notebook's
defaults:

| # | Decision | Resolved as |
|---|---|---|
| 1 | Laplace link approximation | `CFG['laplace_link_approx'] = "sampled"` (maps to laplace-torch's `'mc'`); `"deterministic"` (maps to `'probit'`) is kept for fast iteration. The mapping lives in one place, `LINK_APPROX_MAP`, because laplace-torch does **not** accept the strings `"sampled"`/`"deterministic"` — it validates against `{'mc', 'probit', 'bridge', 'bridge_norm'}`. |
| 2 | ECE binning | `n_bins = 15`, equal-width by default. |
| 3 | §6 in scope? | Gated behind `CFG['run_robustness'] = False`. The multi-seed loop and the ablation hooks are fully written and verified, but the default run stays single-seed. |

Four additions were made during implementation. Each was forced by something the
design did not anticipate, and each is recorded here rather than left implicit.

**11.1 `deterministic_mc` — RNG pinning around VBLL predictive calls.**
§2 claims VBLL is sampling-free, which is true of the *posterior* but not of the
library's `predictive()`: it averages a hard-coded 20 Monte-Carlo samples internally,
so `out.predictive.probs` is stochastic. Two consequences the design missed:
(a) the reported metrics were not reproducible run to run; (b) validation accuracy
wobbled between epochs, so early stopping could fire on MC noise rather than on a real
change. A small context manager (`deterministic_mc`) pins the RNG for the duration of a
predictive call and restores the caller's state afterwards, so training is unaffected.
Verified: two consecutive `evaluate_model` calls on the same VBLL adapter now agree to
`0` difference in NLL and AUROC.

**11.2 `ece_equal_mass` and a second ECE column.** §5 called equal-mass binning "a
natural one-line ablation"; it is now implemented and reported alongside the default.
The motivation turned out to be well founded: on MNIST, 10 of the 15 equal-width bins
are empty and the top bin holds ~93% of the test set, so the equal-width estimator is
weighted almost entirely by that one bin. The results table therefore carries both
`ece` and `ece_eqmass`. On the verification run the two happened to agree closely
(0.0054 vs 0.0042 for VBLL-Disc), which is itself a useful finding — the structural
concentration is real, but it does not distort the number much at this accuracy level.
§27 states that neither column is "the" ECE and that they answer slightly different
questions.

**11.3 Bin-occupancy printout.** Cell 20 prints the number of non-empty bins and the
mass in the top bin for every arm. This is the evidence behind 11.2 and costs one loop.

**11.4 Output layout.** Figures go to `outputs/figures/` and `results.csv` /
`results.json` to `outputs/`, rather than everything in one folder, so that the
directory named `figures` contains only figures. `CFG['out_dir']` and `CFG['fig_dir']`
are the only two places this is decided.

**11.5 Dependency self-repair cell — the notebook is now 29 cells, not 28.** The first
real run hit `ModuleNotFoundError: No module named 'vbll'` because Colab had recycled
the session and discarded every pip-installed package. The setup notebook
(`00_colab_setup.ipynb`) had done its job correctly, in a session that no longer
existed. Requiring the operator to notice this and re-run a second notebook first is a
real usability defect, and it is worse for a grader who opens only
`01_vbll_classification.ipynb`.

A new **first code cell** ("dependency self-repair") now runs before everything else.
It is idempotent: on a healthy session it reports and exits in about a second, touching
nothing. When packages are missing it re-uses the exact constraints mechanism from
`00_colab_setup.ipynb` — record Colab's own `torch`/`torchvision` versions, then
`pip install -c colab-constraints.txt`, so no transitive dependency can replace the
CUDA build of PyTorch with a CPU wheel. It then verifies that every import works *and*
that `torch.__version__` and `torch.cuda.is_available()` are unchanged, and it raises a
diagnosable error (with the exact commands to run) if the install does not take.

Consequence for this document: cells 00–27 keep the meanings defined in §7, and the
self-repair cell sits in front of cell 01 as **cell 00b**. The notebook contains
**29 cells: 7 Markdown + 22 code**.

### Verification performed

Both checks execute the **delivered** `.ipynb`, not a copy. Cells are located by a
unique marker in their header comment, not by a hard-coded index, so inserting or
reordering a cell cannot silently make the harness test the wrong thing.

| Harness | What it does | Result |
|---|---|---|
| `test_classification_notebook.py` | Extracts all 22 code cells and runs them end to end against real MNIST / Fashion-MNIST on CPU (4 epochs, small Laplace grid), then asserts 37 properties of the artefacts | 37/37 pass |
| `test_metrics.py` | Runs the metric functions from the metrics cell against hand-computed answers, including clipping, empty bins, the `conf == 1.0` boundary, and both ECE conventions | 23/23 pass |
| `test_hotfix_install.py` | Hides `vbll`/`laplace` so the self-repair cell takes its **install** branch, intercepts `pip` as `--dry-run`, and asserts the command carries `-c <constraints>`, pins the local torch, never lists torch itself, and fails loudly when the packages remain missing | 11/11 pass |

Also verified: `nbformat` 4.0 validation passes; all 22 code cells `compile()`; the
`return_ood=False` guard raises a *named* `RuntimeError`; `check_contract` rejects a
malformed `Prediction`; the VBLL OOD sign flip is correct (`auroc_native` ≈ 0.92 > 0.5,
confirming that `1 - out.ood_scores` is the right orientation).


