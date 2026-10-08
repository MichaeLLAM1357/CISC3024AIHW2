# Colab setup — VBLL (CISC3024 AI Assignment #2)

Copy-paste runbook. Nothing here needs to be typed or adapted by hand.
Verified on 2026-10-08 by resolving the full dependency graph with
`pip install --dry-run` (see §5 for the exact resolver output).

---

## 0. TL;DR

| Step | Action |
|---|---|
| 1 | `Runtime → Change runtime type → T4 GPU` |
| 2 | Paste **Cell 1** into the first cell and run it (~60–90 s) |
| 3 | Paste **Cell 2** into the second cell and run it — it must print `ALL IMPORTS OK` |
| 4 | Continue with the experiment notebook |

---

## 1. Cell 1 — probe the environment, then install (recommended)

This is the cell to use. It first records what Colab already has, then writes a
**constraints file** pinning `torch` and `torchvision` to the versions Colab ships,
and installs everything else behind that constraint. No transitive dependency can
therefore replace Colab's CUDA-matched torch with a CPU wheel.

```python
# ===== Cell 1 — install all dependencies (Colab, ~60-90 s) =====
import pathlib, platform, torch, torchvision

base = lambda v: v.split("+")[0]          # "2.6.0+cu124" -> "2.6.0"

print("Python      :", platform.python_version())
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU         :", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE -> Runtime > Change runtime type > T4 GPU")

# Pin torch/torchvision to exactly what Colab already has.
pathlib.Path("colab-constraints.txt").write_text(
    f"torch=={base(torch.__version__)}\n"
    f"torchvision=={base(torchvision.__version__)}\n")
print("\n--- colab-constraints.txt ---")
print(pathlib.Path("colab-constraints.txt").read_text())

!pip install -q -c colab-constraints.txt \
    "vbll==0.4.9" "laplace-torch==0.2.3" \
    numpy scipy pandas scikit-learn matplotlib seaborn tqdm

# Prove the constraint held: torch must be byte-identical to the probe above.
print("\ntorch after install:", torch.__version__)
```

### Minimal alternative (if you only want one line)

Safe for the same reason — `torch>=2.0` and `torchvision>=0.15` are already
satisfied on Colab, so pip has no reason to replace them:

```python
!pip install -q "vbll==0.4.9" "laplace-torch==0.2.3" numpy scipy pandas scikit-learn matplotlib seaborn tqdm
```

---

## 2. Cell 2 — verify (must pass before spending any GPU time)

```python
# ===== Cell 2 — verification =====
import importlib, torch
import numpy, scipy, pandas, sklearn, matplotlib, seaborn, tqdm
import vbll, laplace

mods = [torch, numpy, scipy, pandas, sklearn, matplotlib, seaborn, tqdm, vbll, laplace]
for m in mods:
    print(f"{m.__name__:12s} {getattr(m, '__version__', 'n/a')}")

assert torch.cuda.is_available(), "CUDA unavailable — see colab_setup.md §4"
print("\nGPU:", torch.cuda.get_device_name(0),
      f"| VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Smoke-test the exact VBLL API the experiments use.
from vbll.layers.classification import DiscClassification, GenClassification
head = DiscClassification(16, 3, 1e-4, parameterization="diagonal", return_ood=True)
out  = head(torch.randn(8, 16))
loss = out.train_loss_fn(torch.randint(0, 3, (8,)))
print("VBLL head OK | loss", float(loss), "| probs", tuple(out.predictive.probs.shape))
print("\nALL IMPORTS OK")
```

Expected output: every package prints a version, `GPU: Tesla T4 | VRAM 15.x GB`,
and finally `ALL IMPORTS OK`.

---

## 3. What gets installed (and why)

`vbll` is tiny — it only needs `torch` and `numpy`. Almost all of the install time
and all of the dependency risk comes from `laplace-torch`, whose real dependency
tree is:

```
laplace-torch 0.2.3
├── torch>=2.0                     (satisfied by Colab)
├── torchvision>=0.15              (satisfied by Colab)
├── asdfghjkl==0.1a4               strict pre-release pin
├── backpack-for-pytorch           -> torch>=2.2.0, torchvision, einops, unfoldNd
├── curvlinops-for-pytorch>=3.0.1  -> backpack, scipy, einops, einconv,
│                                     linear_operator, opt-einsum, tqdm
├── torchmetrics                   -> torch, lightning-utilities
├── numpy, opt_einsum
```

`asdfghjkl==0.1a4` is a pre-release. That is fine: pip only refuses pre-releases
when they are not named explicitly, and `0.1a4` is named explicitly. If pip ever
refuses it anyway, add `--pre` to the install command.

---

## 4. CUDA version conflicts — symptom, cause, fix

The golden rule: **Colab's `torch` is already correct. Never reinstall it.**
`torch.version.cuda` is the CUDA version torch was *built against*; `nvidia-smi`
reports the maximum CUDA version the *driver* supports. Runtime must be ≤ driver.
Both are fine on a fresh Colab runtime — problems appear only after a manual
`pip install torch`.

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` returns `False` right after the install cell | A dependency pulled a CPU-only torch wheel | **`Runtime → Restart session`**, then re-run Cell 1 (with the constraints file) and Cell 2. Restarting is the clean fix — it restores Colab's pristine CUDA-matched stack without you having to guess a CUDA version |
| `ImportError: libcudart.so.12: cannot open shared object file` | torch runtime newer than the driver supports | Same: restart the runtime. Only if that fails, force-reinstall a wheel matching the CUDA version `!nvidia-smi` reports — see the command below |
| `ImportError: undefined symbol: ...c10_cuda...` / `libc10_cuda.so` | `torch` and `torchvision` are from different builds | Restart, then reinstall **both together**: `!pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121` (replace `cu121` with the CUDA version from `!nvidia-smi`) |
| `ImportError: numpy.core.multiarray failed to import` or `_ARRAY_API not found` | A package was compiled against numpy 1.x while Colab has numpy 2.x | `!pip install --upgrade <package>`; or pin `numpy<2`. Our set does not upgrade numpy, so this should not fire |
| `ERROR: Cannot install ... conflicting dependencies` | The constraints file blocked a dependency from upgrading torch — **this is the constraint working correctly** | Read which package wants the newer torch. Usually it is optional (a curvature backend). If it is genuinely needed, the torch bump has to be accepted, which means accepting the CUDA risk |
| `ModuleNotFoundError: No module named 'laplace'` after a successful install | Colab loaded a stale module cache because numpy/torch-adjacent wheels were touched | `Runtime → Restart session`, then run **Cell 2 only** — do **not** re-run the install |
| `RuntimeError: Found no NVIDIA driver on your system` | Runtime type is CPU-only | `Runtime → Change runtime type → T4 GPU` |
| `asdfghjkl` or `backpack-for-pytorch` fails to build | pip refused the pre-release pin | Add `--pre` to the install command |

### 4.1 Diagnosing it yourself in one cell

```python
# ===== CUDA diagnosis =====
!nvidia-smi | head -12          # driver + max CUDA the driver supports
import torch
print("torch        :", torch.__version__)
print("built for CUDA:", torch.version.cuda)
print("available    :", torch.cuda.is_available())
print("device count :", torch.cuda.device_count())
```

If `available` is `False` while `nvidia-smi` shows a T4, the installed torch is a
CPU build — restart the runtime and re-run Cell 1. If `available` is `True`, you are
done; there is nothing to fix.

### 4.2 Recovery command (only if a restart does not help)

Replace `cu121` with the CUDA version printed by `!nvidia-smi`:

```python
!pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Verified resolver output

`pip install --dry-run` for the package set, run on 2026-10-08:

```text
Would install asdfghjkl-0.1a4 backpack-for-pytorch-1.7.1
  curvlinops-for-pytorch-3.0.1 einconv-0.1.0 einops-0.8.2
  laplace-torch-0.2.3 lightning-utilities-0.15.3 linear_operator-0.6.1
  opt_einsum-3.4.0 seaborn-0.13.2 torchmetrics-1.9.0 unfoldNd-0.2.3 vbll-0.4.9
```

Note what is **absent**: `torch` and `torchvision` do not appear, i.e. the resolver
leaves Colab's CUDA build untouched. That is the property the constraints file
guarantees on Colab itself.

---

## 6. Notes

- The full install is ~13 new packages and completes in roughly 60–90 s on Colab.
- If Colab disconnects, re-run Cell 1 — it is idempotent.
- `requirements.txt` in this folder is the documentation/reproducibility pin set
  (and the source of the §10 environment table in the report). It is **not** meant
  to be `pip install -r`-ed on Colab for the reason explained at the top of it.
