# 補跑 §4.4 消融實驗 —— 操作清單

三個可複製的區塊，外加兩個**先講清楚才不會白等**的發現。

> **2026-10-09 更新 —— 優先用獨立 notebook。**
> 現在有 `02_vbll_ablations.ipynb`（23 格，self-contained）。它**不需要**先跑 notebook 01
> 的任何一格，裡面已經把 CFG、資料、backbone、adapters、訓練迴圈都用同一份原始碼帶上，
> 而且**可續跑**（每跑完一個 ablation 就寫檔，重跑會跳過已完成的）。直接在 Colab 開它、
> Run all 就好。
>
> 下面三個區塊**保留**，作為「萬一你想留在 notebook 01 裡面手動操作」的參考；
> 若用獨立 notebook，就不需要區塊 1 與區塊 2。

---

## ⚠️ 2026-10-09：`disc_lowrank` 在 GPU 上報 device 錯誤 —— 已修

**症狀**（跑消融時出現，`disc_diagonal` 卻沒事）：

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

出在 **vbll 套件自己**的 `vbll/utils/distributions.py`，
`LowRankNormal.logdet_covariance` 這一行：

```python
term2 = torch.linalg.det(arg1 + torch.eye(arg1.shape[-1])).log()
```

`torch.eye(n)` 沒帶 `device=`，所以建立在 **CPU**；而 `arg1` 由 head 的 `cov_factor`／
`cov_diag` 組成，模型已經 `.to(DEVICE)`，所以它在 **cuda:0**。

**為什麼 `diagonal` 沒事**：`diagonal` 對應 vbll 的 `Normal`，它的 `logdet_covariance`
只是 `2 * log(scale).sum(-1)`，**根本不會碰到 `torch.eye`**。
所以這個 bug 只在 `lowrank`（以及 precision 類參數化）才會出現 ——
**它是「取決於某個設定值」的，重跑預設設定永遠看不到。**

**修法**：不改 site-packages（Colab 每次回收都會清掉），改成在 notebook 裡覆寫這兩個函式。
`01` 與 `02` 兩份 notebook 的「Section 2 - arms (b) and (c): VBLL heads」那格**已經內含**這段 shim，
所以重新上傳就會生效。

### 如果你現在這個 session 不想重新上傳 —— 直接貼這一格

在**跑消融那一格之前**先執行下面這格即可（貼上、執行一次、之後照舊跑消融格）：

```python
# =============================================================================
# Device-safety shim for vbll 0.4.9 -- upstream bug, not our code
# =============================================================================
# vbll 建立兩個輔助 identity matrix 時沒有帶 device=，所以它們落在 CPU，
# 而模型在 GPU。第 327 行只有 'lowrank' 參數化會走到（'diagonal' 不會，所以主實驗沒事），
# 第 27 行則由 precision 類參數化經 cholesky_inverse() 走到。
# 唯一差別：identity matrix 繼承 device 與 dtype。數學與原版完全相同。
import torch
import vbll.utils.distributions as _vbll_dist


def _lowrank_logdet_covariance(self):
    term1 = torch.log(self.cov_diag).sum(-1)
    arg1 = _vbll_dist.tp(self.cov_factor) @ (self.cov_factor / self.cov_diag.unsqueeze(-1))
    eye = torch.eye(arg1.shape[-1], device=arg1.device, dtype=arg1.dtype)
    return term1 + torch.linalg.det(arg1 + eye).log()


def _cholesky_inverse(u, upper=False):
    if u.dim() == 2 and not u.requires_grad:
        return torch.cholesky_inverse(u, upper=upper)
    eye = torch.eye(u.size(-1), device=u.device, dtype=u.dtype).expand(u.size())
    return torch.cholesky_solve(eye, u, upper=upper)


_vbll_dist.LowRankNormal.logdet_covariance = property(_lowrank_logdet_covariance)
_vbll_dist.cholesky_inverse = _cholesky_inverse

print("vbll device-safety shim installed")
```

**已驗證**：與原版**數值完全相同**（最大絕對誤差 `0.0`），且梯度可正常回傳
（ELBO 依賴這條路徑）。

> 注意：`disc_diagonal` 已經在 CSV 裡了，重跑消融格會 `skip disc_diagonal`，
> 從 `disc_lowrank` 接著做 —— 不會白跑。

---

## 先講結論：三件事

### 1. 「只跑單一 seed」其實不需要改 `CFG['seeds']`

消融那格（Section 6 - ablation hooks）**完全不讀 `CFG['seeds']`**。多種子循環是**另一格**
（Section 6 - multi-seed robustness）。所以：

> **只要你「不跑」多種子那一格，消融自然就是單 seed。**

`CFG['seeds'] = [0]` 只是**保險** —— 萬一你不小心跑到多種子那格，它只花 1 個 seed 而不是 3 個。

### 2. `disc_dense` 不會爆顯存，但**極慢** —— 這才是真正的問題

我在本機實測了三種協方差參數化（`vbll.DiscClassification(256, 10, ...)`，batch 512）：

| parameterization | head 參數量 | `W_offdiag` 形狀 | 單步 batch-512 訓練（CPU） | 相對速度 |
|---|---|---|---|---|
| `diagonal` | 5,140 | — | 0.005 s | 1× |
| `lowrank` | 12,820 | `(10, 256, 3)` | 0.024 s | ~5× 慢 |
| **`dense`** | **660,500** | `(10, 256, 256)` | **1.875 s** | **~375× 慢** |

**顯存完全不是問題**：`dense` 多出來的參數只有 655,360 個 ≈ **2.6 MB**（加梯度和 AdamW 動量也才約 10 MB）。
中間張量最大也就 ~5 MB。T4 有 15 GB，**不可能 OOM**。

**問題是計算量**：`dense` 的 Jensen 下界要對每個類別做 256×256 的 Cholesky 與三角求解，
所以單步慢 **375 倍**（CPU 實測）。粗估 30 epochs 的 `disc_dense`：

- 本機 CPU：**約 100 分鐘**（其他五個加起來不到 10 分鐘）
- T4：會快很多（GPU 對這種小矩陣批次運算有利），但**仍然是壓倒性的主導成本**，可能數十分鐘
- 而且 vbll 預設 `cov_rank=3`，所以 **`lowrank` 幾乎免費**，不用擔心它

**建議**：把 `disc_dense` 排在最後單獨跑，或直接用它換取時間（見下面的「兩段式」做法）。

### 3. `disc_no_ood` 會如設計般報錯 —— 確認過了

會。那段程式碼是：

```python
try:
    VBLLAdapter(model_a, tag).predict(test_loader)
    note = "adapter did NOT raise - unexpected"
except RuntimeError as err:
    note = "adapter raised as designed: " + str(err)[:70]
```

而 adapter 在 `out.ood_scores` 不存在時會拋**具名** `RuntimeError`
（訊息開頭是 `[disc_no_ood] the VBLL head was built with return_ood=False, so ...`）。
這個路徑在驗證階段已經被斷言測過（37/37 裡包含這一項），所以它會走 `except` 分支，
`note` 會是 `adapter raised as designed: ...`。

> **注意一個不一致**：`disc_no_ood` 那一列的 `accuracy` 記的是**驗證集**準確率
> （`best_a["val_acc"]`），而其他五列記的是**測試集**準確率（`r_a["accuracy"]`）。
> 這是原始消融程式碼的小瑕疵 —— **不要把 `disc_no_ood` 的 accuracy 跟其他列直接比**。
> 它的意義只在於「有沒有如期報錯」，所以報告裡我會把那一列的數字留空、只寫 note。

---

## 區塊 1：啟用消融（複製到一格，跑一次）

在**已經跑過 cell 2（CFG 那一格）之後**執行。不需要去改 cell 2 的內容：

```python
# =============================================================================
# Enable Section 6 for a SINGLE-SEED ablation run
# =============================================================================
CFG["run_robustness"] = True   # 打開 Section 6 的閘門
CFG["seeds"] = [0]             # 保險：就算誤跑多種子那格，也只花 1 個 seed

print("run_robustness :", CFG["run_robustness"])
print("seeds          :", CFG["seeds"])
print()
print("下一步：只跑 'Section 6 - ablation hooks' 那一格。")
print("千萬不要跑 'Section 6 - multi-seed robustness' 那一格。")
```

**然後只跑 `Section 6 - ablation hooks` 那一格。** 不要 Run all —— 否則前面 30 epochs × 4 臂會重跑一遍。

> 前提：kernel 還活著、cell 1–14 與 16–20 的定義都還在記憶體裡。
> 消融那格**不需要** cell 15 的 `trained_models`，它自己會建模型、自己訓練。
> 但如果 Colab 把 session 回收了（你已經遇到過兩次），就必須從 cell 1 重跑到 cell 14
> （**跳過 cell 15 的 30 epochs 訓練**），才能跑消融。

---

## 區塊 2（建議）：可續跑的消融格

**為什麼建議用它**：原本的消融格是一口氣跑完六個模型，任何中斷（Colab 回收、逾時）就**全部歸零**。
考量到 `disc_dense` 可能要跑很久，這個風險很實際 —— 你的 session 已經被回收過兩次。

把 `Section 6 - ablation hooks` 那一格**整格替換**成下面這版。它每跑完一個就立刻寫檔，
重跑時會**跳過已完成的部分**：

```python
# =============================================================================
# Section 6 - ablations, RESUMABLE
# =============================================================================
# 每完成一個 ablation 就立刻寫進 results_ablations.csv，並在重跑時跳過已完成的，
# 所以中斷之後再跑一次就能接上，不會從頭來。
import os
import numpy as np
import pandas as pd

ABLATIONS = [
    ("disc_diagonal",  dict(kind="disc", parameterization="diagonal")),
    ("disc_lowrank",   dict(kind="disc", parameterization="lowrank")),
    ("disc_prior_0.1", dict(kind="disc", prior_scale=0.1)),
    ("disc_prior_10",  dict(kind="disc", prior_scale=10.0)),
    ("disc_no_ood",    dict(kind="disc", return_ood=False)),
    ("disc_dense",     dict(kind="disc", parameterization="dense")),   # 最貴，排最後
]

ABL_CSV = os.path.join(CFG["out_dir"], "results_ablations.csv")
COLS = ["ablation", "accuracy", "nll", "ece", "auroc_native", "note"]

done = {}
if os.path.exists(ABL_CSV):
    for _, r in pd.read_csv(ABL_CSV).iterrows():
        done[r["ablation"]] = r.to_dict()
    print("resuming - already finished:", sorted(done))
else:
    print("starting fresh")


def _save_abl():
    pd.DataFrame([done[t] for t, _ in ABLATIONS if t in done],
                 columns=COLS).to_csv(ABL_CSV, index=False)


for tag, kw in ABLATIONS:
    if tag in done:
        print("skip {} (already done)".format(tag))
        continue

    print("=" * 74)
    print("ablation: {}".format(tag))
    print("=" * 74)

    set_seed(CFG["seed"])
    model_a = build_vbll_model(**kw)
    _, best_a = train_model(model_a, train_loader, val_loader, CFG, tag=tag)

    if kw.get("return_ood", True) is False:
        # 這個 ablation 的重點就是：adapter 必須「大聲且具名」地失敗
        try:
            VBLLAdapter(model_a, tag).predict(test_loader)
            note = "adapter did NOT raise - unexpected"
        except RuntimeError as err:
            note = "adapter raised as designed: " + str(err)[:70]
        row = dict(ablation=tag, accuracy=round(best_a["val_acc"], 4),
                   nll=np.nan, ece=np.nan, auroc_native=np.nan, note=note)
    else:
        r_a = evaluate_model(VBLLAdapter(model_a, tag), test_loader, ood_loader,
                             n_bins=CFG["n_bins"])
        row = dict(ablation=tag, accuracy=round(r_a["accuracy"], 4),
                   nll=round(r_a["nll"], 4), ece=round(r_a["ece"], 4),
                   auroc_native=round(r_a["auroc_native"], 4), note="")

    done[tag] = row
    _save_abl()                                   # <-- 中斷也保得住
    print("  saved:", {k: v for k, v in row.items() if k != "note"})

print()
print(pd.DataFrame([done[t] for t, _ in ABLATIONS if t in done],
                   columns=COLS).to_string(index=False))
```

用這一版就**不需要區塊 1**（它不看 `run_robustness`）。中途斷了，就再跑同一格，
它會從 `skip ... (already done)` 接著做。

---

## 區塊 3：把 CSV 轉成 Markdown 表格（跑完後執行）

```python
# =============================================================================
# results_ablations.csv -> Markdown table (直接貼進報告)
# =============================================================================
import os
import pandas as pd

df = pd.read_csv(os.path.join(CFG["out_dir"], "results_ablations.csv"))


def _fmt(v):
    if pd.isna(v) or (isinstance(v, str) and not v.strip()):
        return "—"
    if isinstance(v, float):
        return "%.4f" % v
    return str(v)


cols = list(df.columns)
print("| " + " | ".join(cols) + " |")
print("|" + "|".join(["---"] * len(cols)) + "|")
for _, row in df.iterrows():
    print("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
```

把印出來的整段貼回給我，我會整理進報告 §4.4，並加上該有的注意事項
（單 seed、`disc_dense` 的成本、`disc_no_ood` 的 accuracy 是驗證集這三點）。

---

## 時間預算

| 項目 | 本機 CPU 實測 / 估計 | T4（估計） |
|---|---|---|
| `disc_diagonal` | < 1 分鐘 | < 1 分鐘 |
| `disc_lowrank` | ~2 分鐘 | < 1 分鐘 |
| `disc_prior_0.1` | ~2 分鐘 | < 1 分鐘 |
| `disc_prior_10` | ~2 分鐘 | < 1 分鐘 |
| `disc_no_ood` | ~2 分鐘（只訓練 + 觸發錯誤） | < 1 分鐘 |
| **`disc_dense`** | **~100 分鐘** | 數十分鐘（不確定，但明顯最貴） |

如果 `disc_dense` 跑太久，告訴我，我可以：
- 在本機 CPU 背景跑它（約 100 分鐘），或
- 把 §4.4 寫成「五軸已跑 + dense 因成本未跑」，並誠實說明理由。
