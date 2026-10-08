# CISC3024 Pattern Recognition — AI Assignment #2
## Stage 1 — Algorithm survey, screening and selection

> Working document (English), produced entirely by AI. Feeds §1 and §2 of the final
> report. Every repository and number below was checked against the live source on
> 2026-10-08; nothing here is estimated.

---

## 0. What the brief asks for

| # | Requirement |
|---|---|
| R1 | Find a **recent** classification/recognition algorithm **based on probability or Bayes theory** |
| R2 | Application in **computer vision and pattern recognition** |
| R3 | The **AI must do everything** — searching, programming, report writing |
| R4 | Report (Word or PDF) with 6 required parts, submitted via UMMoodle, through Turnitin |
| R5 | **"You are NOT allowed to write any code by yourself!"** |

Constraints added by the student, which bound the search as much as the brief does:

| Constraint | Consequence |
|---|---|
| Must run on the **free tier of Google Colab** (T4, ~15 GB VRAM) | Excludes MCMC-at-scale, large foundation models, anything needing many GPU-hours |
| Must have **official / authoritative PyTorch code** | Excludes methods released only as TensorFlow/JAX, and excludes re-implementation (which R5 forbids anyway) |
| Must be **reproducible inside one week** | Penalises repositories with no install instructions or no runnable entry point |

---

## 1. How the AI searched

Search was driven by five independent keyword angles so that the candidate pool would
not be biased toward a single sub-field. Queries issued (verbatim) and what each
surfaced:

| # | Query | Yield |
|---|---|---|
| Q1 | `Variational Bayesian Last Layer VBLL ICLR 2024 classification github` | VBLL — ICLR 2024, official repo `VectorInstitute/vbll` |
| Q2 | `evidential deep learning Dirichlet classification 2024 2025 github official code` | EDL — original NeurIPS 2018, PyTorch re-implementation (2023), 2024–25 variants |
| Q3 | `Laplace approximation Bayesian deep learning classification 2024 2025 library` | Laplace / `laplace-torch`, ELLA (NeurIPS 2022), `laplax` (JAX, 2025) |
| Q4 | `Bayesian neural network image classification 2025 new method github pytorch` | BNN via VI / SG-MCMC (`JavierAntoran/Bayesian-Neural-Networks`) |
| Q5 | `Gaussian process classification deep kernel learning 2024 2025 computer vision pytorch` | GP classification / Deep Kernel Learning — `GPyTorch` |
| Q6 | `Bayesian Nonnegative Decision Layer BNDL ICLR 2025 github code` | **BNDL** — ICLR 2025, `XYHu122/BNDL` |
| Q7 | `Bayesian uncertainty quantification classification 2025 paper code computer vision` | A 2024 review of Bayesian UQ; confirmed the taxonomy used below |

The seventh query was added after the first six, specifically to check that no
obvious 2025–2026 method had been missed. It surfaced only surveys and
application papers, not a better-engineered candidate, so the search was closed.

**Two corrections the AI made to its own first pass.**

1. The first pass returned "Bayes by Backprop" (2015) and "MC Dropout" (2016) as
   leading candidates. Both are foundational but **not recent**, and the brief asks
   for a *recent* algorithm — they were demoted to the "classical baseline" row
   rather than the candidate list.
2. `ELLA` looked recent in search results but its paper is **NeurIPS 2022**; the
   search engine surfaced it alongside 2025 work. It fails SC1 and was excluded,
   leaving `laplace-torch` (maintained through 2025) as the Laplace-family
   representative.

---

## 2. Screening criteria

Six filters. Five are pass/fail; the sixth ranks the survivors. These mirror the
criteria used in AI Assignment #1 so the two reports are methodologically
consistent.

| ID | Criterion | Type | Pass condition |
|----|-----------|------|----------------|
| SC1 | Recency | Hard | Primary publication **2023–2026** |
| SC2 | Probabilistic / Bayesian core | Hard | The *core contribution* is a probabilistic model — an explicit prior, likelihood and posterior (or a Dirichlet/GP formulation). A network that merely outputs a softmax does **not** qualify |
| SC3 | CV / PR application | Hard | Evaluated on a recognisable vision classification/recognition task |
| SC4 | Official PyTorch code | Hard | Author-released or de-facto-authoritative repo, installable and runnable |
| SC5 | Colab free-tier feasible | Hard | Fits 16 GB VRAM; full run completes inside one session |
| SC6 | MVP friction | Soft (1–5) | Glue code, dependency weight and debugging between `pip install` and a reported number |

---

## 3. Candidates

| Algorithm | Venue / year | Probabilistic core | Official PyTorch | Reported outcome |
|---|---|---|---|---|
| **VBLL** (Variational Bayesian Last Layers) | **ICLR 2024** | Deterministic **variational inference** over the last-layer weights; ELBO; closed-form predictive | `VectorInstitute/vbll` — `pip install vbll`, docs + tests + official Colab tutorials | Improves accuracy, **calibration** and **OOD detection** vs baselines (classification & regression) |
| **BNDL** (Bayesian Non-negative Decision Layer) | **ICLR 2025** | Conditional Bayesian **non-negative factor analysis**; Weibull variational posterior | `XYHu122/BNDL` — bare upload (2 commits, no README/requirements/run instructions) | Accuracy + uncertainty + interpretability |
| **Laplace approximation** (`laplace-torch`) | Laplace Redux, NeurIPS 2021; library maintained into 2025 | Gaussian **posterior** around a MAP estimate via the GGN/Hessian; last-layer variant (LLLA) | `AlexImmer/Laplace` — `pip install laplace-torch` | Post-hoc Bayesian uncertainty for any trained net |
| **EDL** (Evidential Deep Learning) | NeurIPS 2018 (+ 2024–25 variants) | **Dirichlet** prior over class probabilities (subjective logic); evidence → uncertainty | `clabrugere/evidential-deeplearning` (PyTorch, 2023); original is TensorFlow | Quantified classification uncertainty |
| **GP classification / Deep Kernel Learning** | Classic (1998 / 2016); active 2023–25 | **Gaussian process** prior over a latent function; variational ELBO (SVGP) | `cornellius-gp/gpytorch` — `pip install gpytorch` | Calibrated probabilistic classification |
| **BNN via VI / SG-MCMC** | Bayes by Backprop 2015; cyclic SGHMC 2020 | **Posterior over all weights** | `JavierAntoran/Bayesian-Neural-Networks` | Uncertainty, but expensive and non-deterministic |

### 3.1 The two serious contenders

**VBLL** and **BNDL** are both "Bayesian last layer" methods, published one year
apart, and they solve the same problem in different ways — VBLL by a deterministic
variational objective on a Gaussian last layer, BNDL by reformulating the network as
a Bayesian non-negative factor model with a Weibull posterior. On paper BNDL is the
more recent and the more ambitious.

**Why BNDL was rejected despite being newer.** Its repository contains only two
folders (`ResNet-Sparse`, `ViT-Sparse`) and **no README, no requirements file and no
run instructions**; the last commit is a bulk "Add files via upload". Getting a
number out of it would mean reverse-engineering the authors' setup from the paper —
which is (a) exactly the re-implementation that R5 forbids and (b) an open-ended
time sink with a one-week deadline. This is the same reasoning that selected
MobileNetV4 over higher-scoring alternatives in Assignment #1: reproducibility beats
a marginal recency gain.

**Why `laplace-torch` was rejected.** It is genuinely excellent and easy, but its
defining paper (Laplace Redux) is 2021, so it fails SC1 as a *recent algorithm*. It
is retained in the experiment design below as a **comparison baseline**, which is
where it adds the most value.

---

## 4. Scoring and selection

Weighted score out of 5. Weights reflect that reproducibility and finishing on time
matter more than a marginal recency gain.

| Algorithm | Recency ×0.15 | Prob./Bayes fit ×0.20 | Official code ×0.20 | Colab fit ×0.25 | Report value ×0.20 | **Total** |
|---|---|---|---|---|---|---|
| **VBLL** | 4.5 | 5 | 5 | 5 | 5 | **4.93** |
| Laplace (`laplace-torch`) | 3.5 | 5 | 5 | 5 | 4.5 | 4.68 |
| EDL (Dirichlet) | 3.0 | 5 | 4 | 5 | 4.5 | 4.40 |
| GP classification / DKL | 3.0 | 5 | 5 | 4 | 4 | 4.25 |
| BNDL | 5.0 | 5 | 2.5 | 3 | 4.5 | 3.90 |
| BNN via VI / SG-MCMC | 2.5 | 5 | 4 | 4 | 3.5 | 3.88 |

**Selected: VBLL — Variational Bayesian Last Layers (Harrison, Willes & Snoek, ICLR 2024).**

### 4.1 Justification

The honest argument is not that VBLL is the newest — BNDL is a year newer — but that
it is the best fit for this assignment's constraints:

- **It is unambiguously Bayesian.** The method is *deterministic variational
  inference*: a prior is placed on the last-layer weights, a variational posterior
  is optimised against an ELBO, and prediction integrates the posterior in closed
  form. That is the exact prior → likelihood → posterior chain the course's
  Bayes Decision Theory lecture builds on, which makes §2 of the report easy to
  write well.
- **It is one `pip install` away.** `pip install vbll`, a documented API, a test
  suite, and two official Colab tutorials (classification and regression) — so the
  risk of spending the week on plumbing instead of experiments is low.
- **It is nearly free to run.** VBLL is quadratic in last-layer width, so on Colab's
  free T4 a full comparison across several model variants fits comfortably inside
  one session.
- **It gives the report a real spine.** VBLL targets accuracy, *calibration* and
  *out-of-distribution detection* — three quantities that a probabilistic
  classifier should get right and a plain softmax does not. That yields a genuine
  experimental story rather than "we trained a model and it reached X %".

---

## 5. Proposed MVP (to be confirmed in Stage 2–3)

| Element | Proposal |
|---|---|
| Task | Image classification + uncertainty estimation |
| In-distribution data | **MNIST** (fast, matches the official tutorial) — optionally extended to CIFAR-10 |
| OOD data | **Fashion-MNIST** (tutorial default) or SVHN/CIFAR-100 |
| Backbone | Small MLP (tutorial) → optionally a small CNN |
| Comparison arms | (a) plain MLP + softmax + cross-entropy, (b) **VBLL discriminative** head, (c) **VBLL generative** head, (d) `laplace-torch` last-layer Laplace |
| Metrics | Test accuracy, NLL, **ECE / reliability diagram**, **OOD AUROC** |
| Ablations | covariance parameterisation (`diagonal` / `dense` / `lowrank`), `prior_scale`, `return_ood` on/off, backbone size |

**Verified API facts** (from the official docs and tutorial, so Stage 3 does not have
to guess):

```python
import vbll
# discriminative Bayesian head
head = vbll.DiscClassification(in_features, out_features, reg_weight,
                               parameterization='diagonal',
                               return_ood=True, prior_scale=1.0)
# generative Bayesian head
head = vbll.GenClassification(in_features, out_features, reg_weight, ...)

out   = model(x)
loss  = out.train_loss_fn(y)      # training objective (ELBO-derived)
vloss = out.val_loss_fn(y)        # validation objective
probs = out.predictive.probs      # predictive class probabilities
ood   = out.ood_scores            # OOD score, when return_ood=True
```

Reference hyper-parameters from the official classification tutorial: AdamW, batch
512, lr 3e-3, 30 epochs, `reg_weight = 1/N_train`, weight decay 0 on the VBLL head.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| MNIST is saturated, so accuracy differences may be noise | Report uncertainty metrics (ECE, OOD AUROC) as the primary result, not accuracy alone |
| Generative vs discriminative VBLL losses are **not comparable** numerically (tutorial warning) | Compare them on accuracy and calibration, never on loss value |
| Colab free tier disconnects | Drive-backed checkpoints + resume, as in Assignment #1 |
| Single seed | Report ≥3 seeds for the headline comparison; state the caveat otherwise |

---

## 7. References (Stage 1)

1. Harrison, J., Willes, J., Snoek, J. *Variational Bayesian Last Layers.* ICLR 2024. arXiv:2404.11599.
2. Hu, X., Duan, Z., Chen, B., Zhou, M. *Enhancing Uncertainty Estimation and Interpretability via Bayesian Non-negative Decision Layer.* ICLR 2025. arXiv:2505.22199.
3. Daxberger, E., Kristiadi, A., Immer, A., et al. *Laplace Redux — Effortless Bayesian Deep Learning.* NeurIPS 2021.
4. Sensoy, M., Kaplan, L., Kandemir, M. *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS 2018.
5. Wilson, A. G., Hu, Z., Salakhutdinov, R., Xing, E. P. *Deep Kernel Learning.* AISTATS 2016.
6. Blundell, C., Cornebise, J., Kavukcuoglu, K., Wierstra, D. *Weight Uncertainty in Neural Networks.* ICML 2015.
7. `VectorInstitute/vbll` — https://github.com/VectorInstitute/vbll
8. `XYHu122/BNDL` — https://github.com/XYHu122/BNDL
9. `AlexImmer/Laplace` (laplace-torch) — https://github.com/AlexImmer/Laplace
10. `cornellius-gp/gpytorch` — https://github.com/cornellius-gp/gpytorch
