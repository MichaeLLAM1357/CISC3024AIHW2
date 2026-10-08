# 在 Colab 上跑 HW2 並把結果交回（操作清單）

這份清單的目的是：**你在 Colab 上跑一次，把指定幾樣東西貼回給我，報告 §4.3 / §4.4 就能填完。**
不需要你寫任何一行程式碼 —— 下面每一格都是複製貼上。

---

## 0. 前置：開啟 GPU

1. Colab 開一個新 notebook（或直接用你現有的）。
2. 選單 **Runtime → Change runtime type → Hardware accelerator = T4 GPU**，按 Save。
3. 確認右上角資源顯示是 GPU。**這一步不做，notebook 會在第一個 guard 就停下來** ——
   這是刻意設計的（`CFG['require_cuda'] = True`），寧可早停也不要偷偷用 CPU 跑 30 epochs。

---

## 1. 上傳兩個 notebook

把這兩個檔案上傳到 Colab（左側資料夾圖示 → 上傳）：

| 檔案 | 用途 |
|---|---|
| `01_vbll_classification.ipynb` | **主要實驗**，含依賴自修復，可以單獨跑 |
| `00_colab_setup.ipynb` | 環境探測與依賴鎖定（可選，但建議跑，用來記錄 Colab 版本號） |

> `01` 的最前面已經有「依賴自修復」cell，所以**即使不跑 `00`，`01` 也能自己裝好套件**。
> 這是上次 `ModuleNotFoundError: No module named 'vbll'` 之後加的。

---

## 2. 執行

**建議順序：**

1. 先跑 `00_colab_setup.ipynb`（**Run all**）。它會探測環境、寫出 `colab-constraints.txt`、
   安裝依賴並驗證。約 2–4 分鐘。
2. 再開 `01_vbll_classification.ipynb`，**Runtime → Run all**。

**時間預算（T4）：**

| 階段 | 預估 |
|---|---|
| 依賴自修復 | 健康 session 約 1 秒；要重裝約 1–2 分鐘 |
| 資料下載 | 約 30 秒 |
| 四個臂各訓練 30 epochs | 每個臂約 3–5 分鐘 |
| Laplace 後驗擬合 + `gridsearch`（`grid_size=30`） | 約 5–10 分鐘（最慢的一段） |
| 四個臂評估 + 出圖 | 約 1–2 分鐘 |
| **合計** | **約 25–40 分鐘** |

> Colab 免費層會因為閒置而斷線。跑之前把這個分頁留在前景，不要在跑的時候切走太久。

---

## 3. 跑完之後：請把這四樣貼回給我

### (1) 主結果表 —— **最重要**

找 notebook 裡這一格：

```
Section 5 - evaluate every arm through the same entry point
```

把它**整格的輸出**貼回（就是 `results_df.to_string()` 印出來的那張表），
長相是這樣：

```
                        accuracy     nll     ece  ece_eqmass  auroc_native  auroc_entropy
arm
a_mlp_softmax              ...     ...     ...        ...           ...            ...
b_vbll_disc                ...     ...     ...        ...           ...            ...
c_vbll_gen                 ...     ...     ...        ...           ...            ...
d_laplace_lla              ...     ...     ...        ...           ...            ...
```

### (2) 分箱佔用率報告

**同一格的輸出下半部**，長相是：

```
equal-width bin occupancy on the ID test set (15 bins):
  a_mlp_softmax     non-empty bins .. /15   mass in top bin ....
  ...
  -> compare the 'ece' and 'ece_eqmass' columns before quoting either.
```

這段是 §4.3 的 Table 4.3.2，也是「等寬 ECE 其實被單一箱主導」這個發現的證據。

### (3) 五張圖

在 Colab 左側 `outputs/figures/` 底下，五個檔案：

| 檔名 | 報告中的位置 |
|---|---|
| `sanity_data.png` | 資料健全性檢查（不放進報告正文，留作附錄備查） |
| `reliability_diagram.png` | §4.3 圖一 |
| `ood_histogram_native.png` | §4.3 圖二 |
| `ood_histogram_entropy.png` | §4.3 圖三 |
| `risk_coverage.png` | §4.3 圖四 |

**最省事的做法**：在 Colab 加一格（下面直接複製），它會把整個 `outputs/` 打包成
`hw2_outputs.zip`，你從左側檔案面板下載即可，一次拿到圖 + CSV + JSON：

```python
import shutil, os
os.chdir("/content")
shutil.make_archive("hw2_outputs", "zip", "outputs")
print("下載 /content/hw2_outputs.zip")
from google.colab import files
files.download("hw2_outputs.zip")
```

### (4) 環境資訊（順手，但很有用）

`results.json` 裡的 `environment` 欄位，或 `00_colab_setup.ipynb` 印出的環境表。
我要用它把報告裡「torch 版本 / CUDA 版本 / GPU 型號」寫成真實值，而不是「Colab 預設」。

---

## 4. 如果你也想跑 §4.4 消融（可選）

消融是**關閉**的，因為它會把執行時間拉長。要跑的話：

1. 在 `01` 裡找到 `CFG` 那一格（`Section 0 - imports, global configuration, seeding`）。
2. 把這一行：

   ```python
   "run_robustness": False,
   ```

   改成：

   ```python
   "run_robustness": True,
   ```

3. 然後只重跑 **Section 6** 的兩格（`Section 6 - multi-seed robustness (gated)` 與
   `Section 6 - ablations (gated)`）。**不要 Run all**，否則前面 30 epochs 會白跑一遍。

   多種子那段是 3 個 seed × 4 個臂，**每個 seed 約 15–20 分鐘**，三個 seed 就是 45–60 分鐘。
   如果你不想花這個時間，就告訴我，§4.4 會誠實寫明「hook 已備妥、因算力未執行」。

跑完把 `results_ablations.csv` 與 `results_multiseed.csv` 的內容貼回。

---

## 5. 如果出錯

| 症狀 | 原因 | 對策 |
|---|---|---|
| `RuntimeError: This notebook requires a GPU` | 沒開 GPU runtime | Runtime → Change runtime type → T4 GPU，然後 **Restart session** 再 Run all |
| `ModuleNotFoundError: No module named 'vbll'` | Session 被回收，pip 套件沒了 | 直接重跑第一個 cell（依賴自修復），它會自己重裝 |
| `torch.cuda.is_available()` 變成 `False` | 有人把 CUDA 版 torch 換成 CPU 版 | 依賴自修復 cell 已用 constraints 檔擋住這件事；若還是發生，**重啟 session** 再跑第一個 cell |
| CUDA out of memory（Laplace 那段） | GLM 預測一次吃太多 | 把 `CFG['laplace_batch_size']` 從 `512` 調成 `256` 或 `128` |
| Session 斷線、跑到一半沒了 | Colab 免費層閒置回收 | 重跑；若常發生，告訴我，改成「存 checkpoint + 分段跑」 |
| `RuntimeError: hotfix incomplete: ...` | 依賴真的裝不上 | 錯誤訊息裡有可直接複製的修復指令，照著貼 |

---

## 6. 交回之後我會做什麼

你貼回上面四樣之後，我會：

1. 把 §4.3 的 Table 4.3.1、Table 4.3.2 填成真實數字，並把四張圖放進 `figures/`；
2. 把 §4.4 依 `results_ablations.csv` 寫完（或誠實標明未執行）；
3. 用真實的環境版本更新 §4.1 的設定表；
4. 更新 §3.6 的驗證表，把「本機 CPU 冒煙測試」與「Colab 正式跑」分開列；
5. 產出 `CISC3024_AI2_Report.docx` 與 `.pdf`（Times New Roman 12pt 正文 / Arial-Bold 標題，與 HW1 一致）；
6. §6 的源碼連結等你決定託管方式後補上。

> **附帶的交叉檢查。** 我這邊同時在跑一次本機 CPU 的全量 30-epoch 版本（僅供對照）。
> 如果你的 Colab 數字和我這邊的形狀差很多（例如某個臂的 AUROC 掉到 0.5 以下），
> 那就是有東西不對，先別急著寫進報告 —— 告訴我，我們一起查。
