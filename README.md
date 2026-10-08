# CISC3024 Pattern Recognition — AI Assignment #2

**Variational Bayesian Last Layers (ICLR 2024) for calibrated classification and
out-of-distribution detection, against a last-layer Laplace baseline**

Four arms share one MLP backbone and one evaluation contract on MNIST (in-distribution) and
Fashion-MNIST (out-of-distribution). The Bayesian heads **halve the calibration error** at no
accuracy cost, and two of them raise OOD AUROC from **0.845 to 0.956** — but only one of the
two VBLL heads actually helps, and the honest reading is that the plain MLP's confidence is
the thing being fixed, not its accuracy.

> Every line of code in this repository was generated with AI assistance, as the assignment
> brief requires. The report documents what the AI got right, what it got wrong, and how each
> error was caught.
>
> * [`CISC3024_AI2_AI_Workflow_Log.md`](CISC3024_AI2_AI_Workflow_Log.md) — the chronological
>   record: every prompt, every error, every correction.
> * [`CISC3024_AI2_Prompt_Log.md`](CISC3024_AI2_Prompt_Log.md) — the same prompts grouped by
>   stage, in Chinese (the language they were actually typed in).

---

## Results

MNIST in-distribution, Fashion-MNIST out-of-distribution, 30 epochs, seed 0. Every arm uses
the same backbone, the same optimiser and the same train/validation split. Lower NLL/ECE is
better; higher AUROC is better.

| Arm | Accuracy | NLL | ECE | ECE (equal-mass) | OOD AUROC (native) | OOD AUROC (entropy) |
|---|---|---|---|---|---|---|
| `a_mlp_softmax` | 0.9792 | 0.0818 | 0.0105 | 0.0102 | 0.8453 | 0.8446 |
| `b_vbll_disc` | 0.9773 | 0.0750 | **0.0047** | 0.0031 | 0.9519 | 0.9552 |
| `c_vbll_gen` | 0.9734 | 0.1118 | 0.0050 | 0.0027 | 0.8261 | 0.8246 |
| `d_laplace_lla` | **0.9794** | **0.0707** | 0.0048 | 0.0035 | **0.9560** | **0.9595** |

### Four findings

1. **The calibration win is the real result.** ECE falls from 0.0105 to ≈0.0048 — roughly a
   factor of two — for both Bayesian heads, and it is free: no arm loses accuracy.
2. **OOD detection improves for two arms and degrades for one.** Laplace (0.9595) and
   VBLL-Disc (0.9552) clearly beat the plain MLP (0.8446). **`c_vbll_gen` is *worse* than the
   plain MLP (0.8246)** — the generative head models a class-conditional density over
   *inputs* and inverts it by Bayes, and that inversion is the weak link here. Saying "VBLL
   beats the MLP" without naming the head would be wrong.
3. **Laplace and VBLL-Disc are a near-tie, not an ordering.** 0.9595 against 0.9552 is a
   difference of 0.004, well inside the run-to-run variation measured below.
4. **A reduced-budget run cannot predict this ranking.** A 4-epoch smoke run put VBLL-Disc
   first; the 30-epoch run reverses the top two and drops the baseline from second to third.

Figures: `figures/sanity_data.png`, `figures/reliability_diagram.png`,
`figures/ood_histogram_native.png`, `figures/ood_histogram_entropy.png`,
`figures/risk_coverage.png`. Written up in report §4.2–§4.3.

---

## Ablation study

Run separately as `notebooks/02_vbll_ablations.ipynb`; raw rows in
`data/results_ablations.csv`. Seed 0, same backbone and budget as the main run.

| Arm | Changed from default | Accuracy | NLL | ECE | OOD AUROC |
|---|---|---|---|---|---|
| `disc_diagonal` | — (reference) | 0.9839 | 0.0571 | 0.0061 | 0.9208 |
| `disc_lowrank` | `parameterization='lowrank'` | 0.9838 | 0.0615 | 0.0059 | 0.9390 |
| `disc_dense` | `parameterization='dense'` | 0.9835 | 0.0584 | 0.0056 | 0.9253 |
| `disc_prior_0.1` | `prior_scale=0.1` | 0.9833 | 0.0574 | 0.0040 | 0.9103 |
| `disc_prior_10` | `prior_scale=10.0` | 0.9832 | 0.0610 | 0.0077 | 0.9145 |
| `disc_no_ood` | `return_ood=False` | 0.9850 (val) | — | — | — |

1. **Accuracy does not care** about the covariance shape or the prior scale — the whole
   sweep spans 0.0007, which is *smaller* than the run-to-run variation measured below.
2. **`dense` is not worth it.** It carries 660,500 head parameters against 5,140 for
   `diagonal`, measured ≈375× slower per step, and buys the *smallest* OOD gain of the three.
3. **Calibration moves monotonically with the prior scale** — ECE 0.0040 at `prior_scale=0.1`
   rising to 0.0077 at `10.0`. At ≈2.6× the run-to-run reference, this is the most defensible
   finding in the sweep.
4. **`disc_no_ood` is a contract test, not a performance comparison.** With `return_ood=False`
   the head carries no `ood_scores`, and the adapter raised a *named* `RuntimeError`
   identifying the cause — exactly as §3.2 designed. Its recorded accuracy is the validation
   figure, so it must not be ranked against the other rows.

### The multi-seed loop was not run

Every number above is a **single-seed observation with no error bar**, and so is every number
in the main table. The reason is a time budget: the sweep is already six independent 30-epoch
training runs on top of a 25–40 minute main run, and repeating it per seed multiplies that.

Rather than leave it as a caveat, the report derives a rough **empirical reference for
run-to-run variation** from two identically-configured runs that differ only in RNG trajectory
(`b_vbll_disc` in notebook 01 and `disc_diagonal` in notebook 02 — the two notebooks seed
differently):

| | Accuracy | ECE | OOD AUROC |
|---|---|---|---|
| `b_vbll_disc` (main run) | 0.9773 | 0.0047 | 0.9519 |
| `disc_diagonal` (ablation run) | 0.9839 | 0.0061 | 0.9208 |
| **difference** | **0.0066** | **0.0014** | **0.0311** |

Re-reading both tables against that reference: the main table's 0.11 AUROC lead is ≈3.5× the
reference and survives; the Laplace/VBLL-Disc near-tie (0.004) and the `lowrank` advantage
(0.018) are both *inside* it and are **not** treated as orderings anywhere in the report. This
is the most useful thing the ablation produced for the main table.

---

## The evaluation contract

The core engineering decision, and the reason the four arms are comparable at all: every arm
is adapted to one interface, `predict(loader) -> Prediction`, where `Prediction` carries
`probs`, `logits` and an `ood_score` (higher = more OOD) and **validates its own shapes**.
Adding an arm means writing an adapter, not touching the metrics.

Two consequences worth knowing:

* **Two OOD score families are reported.** Each arm has its *native* score, but only
  predictive entropy is defined for all four, so `auroc_entropy` is the only fair cross-arm
  comparison. Both are in the tables.
* **The contract fails loudly.** With `return_ood=False` the adapter raises a named
  `RuntimeError` at the boundary where the failure is interpretable, rather than a bare
  `AttributeError` deep inside the metrics code. The ablation verifies this.

---

## Repository contents

```
.
├── CISC3024_AI2_Report.md               the full report (6 required sections)
├── CISC3024_AI2_Report.docx             the same report in Word
├── CISC3024_AI2_Report.pdf              the same report in PDF
├── CISC3024_AI2_AI_Workflow_Log.md      every prompt, error and correction
├── CISC3024_AI2_Prompt_Log.md           the prompt log, in Chinese
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 00_colab_setup.ipynb             environment probe and install (optional)
│   ├── 01_vbll_classification.ipynb     the main experiment — four arms, all figures
│   └── 02_vbll_ablations.ipynb          the ablation sweep (§4.4), resumable
├── figures/                             the five figures used in the report
├── data/
│   ├── results.csv                      the main table
│   ├── results.json                     configuration, environment, per-arm provenance
│   └── results_ablations.csv            the six ablation rows
└── docs/
    ├── 01_algorithm_survey.md           how the algorithm was found and screened
    ├── 02_notebook_architecture.md      the notebook design specification
    ├── colab_setup.md                   Colab install runbook
    ├── RUN_ON_COLAB.md                  how to run the main notebook + handback checklist
    ├── RUN_ABLATIONS.md                 how to run the ablation sweep
    └── PANDOC_CONVERSION.md             Markdown → Word/PDF with Pandoc
```

---

## How to reproduce

1. **Open a GPU runtime.** In Colab, *Runtime → Change runtime type → T4 GPU*. This matters:
   the notebook sets `CFG['require_cuda'] = True` and stops immediately without it. To
   deliberately run on CPU, set that flag to `False` — but the Laplace arm is slow on CPU.
2. **The main run.** Open `notebooks/01_vbll_classification.ipynb` and *Run all*. Its first
   cell is a **dependency self-repair** step: Colab discards pip-installed packages when a
   session is recycled, so the cell re-installs `vbll` and `laplace-torch` *behind a
   constraints file that pins Colab's own torch* — guaranteeing the CUDA build is never
   swapped for a CPU wheel. It is idempotent; on a healthy session it only reports. Expect
   25–40 minutes, and ~0.97–0.98 accuracy.
3. **The ablations.** Open `notebooks/02_vbll_ablations.ipynb` and *Run all*. It is
   self-contained (it re-runs nothing from notebook 01) and **resumable**: each finished arm
   is written to `data/results_ablations.csv` at once, so if the runtime is recycled, run the
   same notebook again and it continues from where it stopped.

### Environment

| | |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| vbll | 0.4.9 |
| laplace-torch | 0.2.3 |
| Hardware | Google Colab free tier, NVIDIA Tesla T4 |
| Seed | 0 |

---

## A note on two device bugs

Both were found by *running* the code on a GPU, not by reading it, and both are documented in
report §3.4.

1. **`laplace-torch`'s `prior_precision` setter moves the tensor to the model's device.**
   `np.asarray` on it therefore raises `TypeError: can't convert cuda:0 device type tensor to
   numpy` on a GPU and is perfectly correct on CPU. A 37-assertion CPU-only harness passed
   while the notebook died on the GPU — **a test suite that runs on one device silently
   declares every other device out of scope.**
2. **`vbll` 0.4.9 allocates two helper identity matrices with `torch.eye(...)` and no
   `device=`.** They land on the CPU while the model is on the GPU. The `diagonal`
   parameterization never touches either line, so the default configuration runs clean and
   only the `lowrank` ablation arm dies: **the defect is conditional on a configuration
   value, and re-running the default could never expose it.** Both notebooks install a
   device-safety shim at import time that replaces the two functions with numerically
   identical, device-correct equivalents (verified: max absolute difference `0.0`, gradient
   path intact). Editing `site-packages` would not survive a Colab session recycle.

---

## Reference

Harrison, J., Willes, J., Snoek, J. *Variational Bayesian Last Layers.* ICLR 2024.
[arXiv:2404.11599](https://arxiv.org/abs/2404.11599)

Daxberger, E., Kristiadi, A., Immer, A., et al. *Laplace Redux — Effortless Bayesian Deep
Learning.* NeurIPS 2021.

Implementations: [`VectorInstitute/vbll`](https://github.com/VectorInstitute/vbll),
[`AlexImmer/Laplace`](https://github.com/AlexImmer/Laplace).
