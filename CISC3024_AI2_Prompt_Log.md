# CISC3024 AI Assignment #2 — 提示詞全集（Prompt Log）

**演算法：** Variational Bayesian Last Layers (VBLL) — ICLR 2024 ｜
**應用：** 影像分類 + 校準化不確定性 + 分布外（OOD）偵測
**資料：** MNIST（分布內）+ Fashion-MNIST（分布外）
**最終結果：** *（待 Colab 正式跑完填入）*

---

## 這份文件是什麼

作業要求報告必須交代「**how you ask AI tools to find the algorithm**」，並且要寫出
「**detailed steps**」。這份檔案把整個專案用過的每一條提示詞按階段記錄下來，
包含當時的意圖、以及實際產出了什麼。

> 提示詞是**中文原文**，因為那是我實際輸入的語言。報告正文
> （`CISC3024_AI2_Report.md`）是英文，其中的 **Appendix C** 是本檔案的英文註解版。

**標記慣例**（重要，為了不讓讀者誤以為所有引用都是逐字）：

| 標記 | 意思 |
|---|---|
| **逐字** | 我當時實際輸入的原文，一字未改 |
| **要點** | 依當階段需求重建的摘要，不是逐字引用 |

**貫穿全程的四條原則：**

| 原則 | 具體做法 |
|---|---|
| **不叫 AI 下結論** | 提示詞只索取「程式碼 / 計畫 / 解釋 / 批評」，演算法選型與數字解讀由我自己下判斷 |
| **能查證的 AI 說法一律查證** | 有 **6 處 API 事實與名字給人的直覺相反**，全部是打開已安裝套件原始碼才查出來的，記錄在報告 §3.2–§3.4 與本檔「階段 3c」 |
| **貼真實錯誤，不貼描述** | 提示詞直接引用真實報錯字串，修復才落在真正的缺陷上（階段 3d 是最典型的例子） |
| **要求「不要猜測」** | 開場第一條提示詞就寫明「有問題請提出，不要做猜測」，這是整個專案分成六個可審查階段的原因 |

---

## 階段 0 — 讀懂作業

### 0.1 開場提示詞（逐字）

> 我需要在做 CISC3024 Pattern Recognition 的 AI Assignment #2。要求：找一个**近期**的、
> **基于概率或 Bayes 理论**的分类/识别算法，用于计算机视觉与模式识别。整个作业必须由
> AI 完成（找算法、编程、写报告）。报告用 Word 或 PDF，下周经 UMMoodle 提交，会过
> Turnitin 查重。报告须含 6 部分：(1) AI 如何找到算法 (2) 算法描述 (3) AI 如何实现
> (4) 实验设置与结果 (5) 学到什么 (6) 源码的网页链接。**"You are NOT allowed to write
> any code by yourself!"** 格式或风格可参考 HW1 的报告。你先按一步步不同階段來做，
> 有問題請提出，不要做猜測。

**這條提示詞有五處是刻意設計的：**

1. **「不要做猜測」** —— 這句是整個專案形狀的來源。它把「一個大而不可審查的交付物」
   變成「六個可審查的階段」，也讓演算法選型變成「先提案、後實作」而不是先做再說。
2. **點名 style 要對齊 HW1**，不只是「要有一份報告」。沒有這句，報告的結構會與
   老師手上已有的參照點不一樣。
3. **逐字引用禁令。** 作業最關鍵的一條規則決定「誰做什麼」，所以寫在提示詞裡而不是轉述。
4. **自己加的限制比作業本身更綁手。** 我在後續對話中補上兩條：
   必須跑在 **Colab 免費層（T4, ~15 GB）**、必須有**官方 PyTorch 程式碼**。
   第一條排除了大規模 MCMC 與貝葉斯集成；第二條排除了「從論文重寫」——
   而那正是作業禁止的事。
5. **「下周提交」** 是唯一沒有被釘死的資訊，我在最後確認階段回頭問了確切日期。

**產出**：六階段計畫（調研選型 → 數學描述 → Colab 環境 → notebook 架構 → notebook 實作
→ 報告與交付），以及第一批待確認問題（選題方式、實驗環境、託管方式、報告語言、截止日）。

---

## 階段 1 — 找演算法

### 1.1 核心檢索提示詞（要點）

> 依作業要求找「近期 + 機率/Bayes 內核 + 電腦視覺 + 官方 PyTorch 程式碼 + Colab 免費層可跑」
> 的分類演算法；**要 5–6 個候選而不是 1 個**，每個候選都要核對官方倉庫與論文；
> 最後給出加權評分與一個 MVP 建議。

**為什麼要「多個候選」**：一個推薦無從評估；六個可以互相對照 —— 這正是報告 §1.3
那張篩選標準表（SC1–SC6）與 §1.4 評分表的來源。

**產出**：7 組檢索（Q1–Q7）、SC1–SC6 篩選標準、6 個候選、加權評分：
**VBLL 4.93** > laplace-torch 4.68 > EDL 4.40 > GP/DKL 4.25 > BNDL 3.90 > BNN 3.88。

**過程中 AI 自我更正了兩處**（這也是要「核對論文」而不是「相信摘要」的理由）：
- Bayes by Backprop (2015) 與 MC Dropout (2016) 不夠「近期」，降級為基線而非候選；
- **ELLA 論文實為 NeurIPS 2022**，不滿足 SC1，因此從候選名單剔除。

**兩個「不加這句就找不到」的事實：**
- VBLL 的官方實作是 `VectorInstitute/vbll`，`pip install vbll` 即可，且有官方 Colab 教學；
- **BNDL 雖然更新（ICLR 2025），但被否**：倉庫只有 `ResNet-Sparse` / `ViT-Sparse`
  兩個資料夾，沒有 README、沒有 requirements、沒有執行說明，只有 2 次 commit。
  復現風險高，而「從論文重寫」等於違反「禁止手寫程式碼」。理由與 HW1 選
  MobileNetV4 的邏輯一致：**可復現性優先於新穎性**。

**交付**：`outputs/01_algorithm_survey.md`。

### 1.2 確認階段 1 成果（要點）

> 這是階段 1（選題與方案設計）的成果，之後怎麼做？

---

## 階段 2 — 演算法數學描述

### 2.1 要求推導而不是概述（要點）

> 把 VBLL 的數學寫清楚：問題設定、先驗、變分後驗、ELBO、Jensen 下界、
> 兩種 head 的差別、以及為什麼它適合本作業。**要能對應到課程的
> `4-BayesDecisionTheory.pdf` 的先驗–似然–後驗鏈條。**

**產出**：報告 §2。核心是 Jensen 下界

$$\mathbb{E}_{q}[\log \mathrm{softmax}(z)_y] \;\ge\; \bar z_y - \log \sum_{c} \exp\!\left(\bar z_c + \tfrac{1}{2}\Sigma_{cc}\right)$$

**一個關鍵提醒（AI 主動提出、我採納）**：`DiscClassification` 與 `GenClassification`
的**損失函數數值不可互相比較**（官方教學明確警告）。因此實驗中**只比較指標，不比較 loss**。

---

## 階段 3a — Colab 環境與依賴鎖定

### 3.1 要完整安裝指令，且不讓我手寫設定（要點）

> 給我完整的 `requirements.txt` + Colab 第一個 cell 的安裝指令 + CUDA 衝突說明，
> **不要叫我手動寫任何一行設定**。

**產出**：`requirements.txt`、`colab_setup.md`、`00_colab_setup.ipynb`。

**這階段的關鍵工程決策（可複用）**：Colab 上**絕不能重裝 torch** ——
這是 Colab「CUDA not available」的頭號原因。做法是安裝前把 Colab 現有的
torch/torchvision 版本寫進 **constraints 檔**，再 `pip install -c colab-constraints.txt ...`，
使任何傳遞依賴都無法把 CUDA 版 torch 換成 CPU 輪子。

**本地實測**（curl PyPI JSON）：`vbll 0.4.9` 只依賴 `torch>=1.6`、`numpy>=1.21`（極輕）；
`laplace-torch 0.2.3` 依賴鏈較重，會帶進 13 個套件，但 **`pip install --dry-run` 確認不動
torch/torchvision**。

---

## 階段 3b — Notebook 架構設計

### 3.2 只做架構設計，先不寫程式（要點）

> 已定：資料集 **MNIST(ID) + Fashion-MNIST(OOD)**；四個模型臂
> (a) MLP+softmax (b) VBLL `DiscClassification` (c) VBLL `GenClassification`
> (d) laplace-torch 最後一層 Laplace；指標 Accuracy / NLL / ECE / OOD AUROC；
> 圖 Reliability Diagram + OOD 直方圖。**這階段只做架構設計，不要寫具體程式碼。**

**產出**：`outputs/02_notebook_architecture.md`（8 節 / 28 cell 設計）。

**這階段最重要的設計決策：適配器（adapter / port）模式。**
四個臂 forward 回傳的東西完全不同（logits / VBLL output object / Laplace 預測張量），
若讓 `evaluate_model` 自己分支，就會退化成 `if arm == ...` 的階梯。做法是每臂包一個
adapter，只暴露 `predict(loader) -> Prediction`，`evaluate_model` 只認 `Prediction`。

**公平性設計（關鍵）**：四個臂的 OOD 分數定義不同，只報一列 AUROC 會誤導。
因此 `evaluate_model` 輸出**兩族**：`auroc_native`（各臂自己的分數，VBLL 的頭條）
與 `auroc_entropy`（預測熵，四臂都有，可同口徑比較）。

### 3.3 三個待拍板問題（要點）

> ① Laplace 用確定性還是採樣 `link_approx`？② ECE 的 `n_bins`？③ 本輪是否就做多種子/消融？

**我的決定**：`link_approx = "sampled"`（保留 `"deterministic"` 供除錯）、
`n_bins = 15`（等寬）、**本輪只跑單種子**（多種子/消融寫好但用 `run_robustness = False` 關掉）。

---

## 階段 3c — Notebook 完整實作

### 3.4 要求完整程式碼、不得偽程式碼（要點）

> 所有程式碼要**完整、不得偽程式碼**，嚴格按 `02_notebook_architecture.md` 的
> 28 cell / 8 節實作。

**產出**：`01_vbll_classification.ipynb`（28 cells = 7 md + 21 code）。

**這階段最有價值的部分，是 AI 從「已安裝的套件原始碼」而非文件裡讀出來的 5 個 API 事實**
（詳見 `CISC3024_AI2_AI_Workflow_Log.md` 階段 3c）。其中兩個是**方向性錯誤**，
如果沒抓到會產生「看起來很合理的錯數字」：

- `out.ood_scores` 是**最大預測機率**，越大越像**分布內** —— 與名字給人的直覺相反；
- `DiscClassification.predictive()` 內部做 **20 次 MC 取樣平均**，**不是**確定性的。

---

## 階段 3d — 熱修復（真實報錯）

### 3.5 貼真實報錯（逐字）

> 我正在運行 01_vbll_classification.ipynb，但在執行「Section 0 - environment echo and
> fail-fast guards」這個 Cell 時，出現了以下錯誤：
>
> ```
> ModuleNotFoundError: No module named 'vbll'
> ```
>
> 我確信之前已經運行過 00_colab_setup.ipynb 安裝了依賴，但 Colab 的會話可能已經重置
> 或斷開了。請幫我生成一個「熱修復（Hotfix）」程式碼區塊，我可以直接貼在
> 01_vbll_classification.ipynb 的最前面（或者作為一個單獨的 Cell 運行）。要求：
> 自動安裝缺少的 vbll 和 laplace-torch 套件。必須包含我之前 00_colab_setup.ipynb 中的
> 「約束文件（constraints file）」邏輯，確保安裝這些套件時不會意外替換掉 Colab 自帶的
> CUDA 版 PyTorch，避免 torch.cuda.is_available() 變成 False。給我完整的、可直接複製的
> 程式碼。**不要叫我手動寫任何一行代碼。**

**根因**：不是套件壞了，是**套件沒了** —— Colab 回收/斷開 session 後會丟棄所有 pip 安裝的
套件；`00_colab_setup.ipynb` 當時確實裝成功，但那個 session 已經不存在。

**修復方式（重要教訓）**：不再依賴「先跑 00 再跑 01」的兩段式流程，而是給 `01`
**在最前面加一個冪等的「依賴自修復」cell**。理由：**助教/評分者很可能只打開 `01`
直接 Run All，兩段式流程必炸。**

**產出**：`01_vbll_classification.ipynb` 從 28 cells 變成 **29 cells**；
另附 `hotfix_env_cell.py`（同一份程式碼的獨立副本，方便直接複製貼上）。

### 3.6 要求可上傳的 notebook 而不是逐格貼（逐字）

> 請給我 pynb 格式，讓我運行

**產出**：可直接上傳 Colab 的 `01_vbll_classification.ipynb`。

---

## 階段 4 — 交付前確認

### 4.1 四個待決問題（逐字回答）

| 問題 | 我的決定 |
|---|---|
| §4.3 / §4.4 數字用哪個來源？ | **等我的 Colab 跑完**（不用本機 CPU 全量結果當正式數字） |
| §6 源碼連結怎麼發佈？ | **之後再補充** |
| 封面 Submission date？ | **10 月 12 日（下週一）** |
| §4.4 消融要不要現在跑？ | **等 Colab 跑**（與主結果同一次 session 內完成） |

**這四個問題的意義**：報告裡凡是沒有真實數字的欄位，一律標 `[PENDING]` 而不是填估計值。
這是「不要做猜測」在**交付階段**的執行方式 —— 沒有數字就寫「沒有數字」。

---

## 附錄：本檔案對應的交付物

| 階段 | 交付物 |
|---|---|
| 1 | `outputs/01_algorithm_survey.md` |
| 3a | `requirements.txt`、`colab_setup.md`、`00_colab_setup.ipynb` |
| 3b | `outputs/02_notebook_architecture.md` |
| 3c | `01_vbll_classification.ipynb` |
| 3d | `01_vbll_classification.ipynb`（29 cells）、`hotfix_env_cell.py` |
| 4 | `outputs/CISC3024_AI2_Report.md` → `.docx` → `.pdf` |
