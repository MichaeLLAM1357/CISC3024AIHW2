# CISC3024 AI Assignment #2 — AI 工作流程記錄（AI Workflow Log）

**演算法：** Variational Bayesian Last Layers (VBLL) — ICLR 2024 ｜
**應用：** MNIST 分類 + 校準化不確定性 + Fashion-MNIST 分布外偵測
**最終結果：** *（待 Colab 正式跑完填入）*

---

## 這份文件是什麼

作業要求報告交代「AI 如何實作演算法」（第 3 部分）。這份檔案是那份報告的**完整後台記錄**：
每個階段 AI 產出了什麼、哪裡是錯的、怎麼發現的、怎麼修的。

它的價值不在於證明 AI 做了很多，而在於**誠實記錄 AI 在哪裡會出錯**。本專案有一類錯誤
特別值得記下來：**它們不會讓程式崩潰，只會產生看起來很合理的錯數字。**

> 報告正文 `CISC3024_AI2_Report.md` 的 §3 是本檔案的濃縮版。
> 本檔案保留的是被壓縮掉的部分：完整清單、驗證方式、以及每一次修正前後的差異。

---

## 一頁總覽

| 階段 | AI 產出 | 交付物 | 驗證方式 |
|---|---|---|---|
| 1 | 文獻檢索、篩選標準、加權評分、選型 | `outputs/01_algorithm_survey.md` | 逐條核對官方倉庫與論文（自行更正 2 處） |
| 2 | 演算法數學描述（ELBO、Jensen 下界） | 報告 §2 | 對照官方論文與課程 `4-BayesDecisionTheory.pdf` |
| 3a | Colab 環境探測 + 依賴鎖定 | `requirements.txt`、`colab_setup.md`、`00_colab_setup.ipynb` | `pip install --dry-run` 實測；抽出 cell 執行 **10/10 PASS** |
| 3b | Notebook 架構規格（28 cell / 8 節） | `outputs/02_notebook_architecture.md` | 設計審查（本階段刻意不寫程式） |
| 3c | 實驗 notebook 完整實作 | `01_vbll_classification.ipynb`（28 cells） | 端到端 **33/33 PASS** + 指標單測 **23/23 PASS** |
| 3d | 依賴自修復 cell（真實 Colab 報錯後） | `01_vbll_classification.ipynb`（29 cells）、`hotfix_env_cell.py` | 熱修復專項 **11/11 PASS**；端到端升級為 **37/37 PASS** |
| 4 | 報告撰寫 | `CISC3024_AI2_Report.md` | 待 Colab 數字填入後產出 `.docx` / `.pdf` |

---

## 階段 1 — 演算法調研與選型

**做法**：把作業的「近期 + 機率/Bayes 內核 + 電腦視覺」三個模糊條件，先轉成
**6 條可判定的篩選標準（SC1–SC6）**，再要求每個候選**全部通過**才進入評分。
然後用 7 組檢索（Q1–Q7）建立候選池 —— 前 5 組刻意分屬不同子領域，避免候選池偏向
單一研究方向；第 7 組是事後補的，用來確認沒有漏掉 2025–2026 的明顯方法。

**AI 自行更正了兩處**（這是要求「核對論文」而不是「相信摘要」的價值）：

| 更正 | 原本的說法 | 查證後 |
|---|---|---|
| 1 | Bayes by Backprop (2015) 與 MC Dropout (2016) 列為候選 | 不夠「近期」，**降級為基線**，不列入候選 |
| 2 | ELLA 被描述為 2024–25 的新方法 | 論文實為 **NeurIPS 2022**，不滿足 SC1，**剔除** |

**最終評分**：VBLL **4.93** > laplace-torch 4.68 > EDL 4.40 > GP/DKL 4.25 >
BNDL 3.90 > BNN 3.88。

**選型理由（與 HW1 一致的方法論）**：**可復現性優先於新穎性。**
BNDL 是 ICLR 2025、比 VBLL 更新，但倉庫只有 `ResNet-Sparse` / `ViT-Sparse` 兩個資料夾，
沒有 README、沒有 requirements、沒有執行說明，只有 2 次 commit。要跑起來就得從論文重寫
—— 而作業明文禁止我寫程式。**一個不能執行的新方法，分數是 0；一個能執行的舊方法，
分數是它跑出來的所有數字。**

**VBLL 勝出的三個具體理由**：
1. `pip install vbll` 即可，依賴極輕（只依賴 `torch` 與 `numpy`）；
2. 有官方文件、測試與**兩個官方 Colab 教學**（MNIST + Fashion-MNIST 的設定可直接沿用）；
3. 其數學內核（對最後一層權重的變分推論）**直接對應課程的貝葉斯決策理論**，
   報告能寫出真東西而不是複述摘要。

---

## 階段 2 — 演算法數學描述

**核心推導**：對最後一層權重做確定性變分推論，最大化 ELBO；其中對數 softmax 的期望
用 **Jensen 下界**取得閉式：

$$\mathbb{E}_{q}[\log \mathrm{softmax}(z)_y] \;\ge\; \bar z_y - \log \sum_{c} \exp\!\left(\bar z_c + \tfrac{1}{2}\Sigma_{cc}\right)$$

其中 $\bar z = M\phi(x)$、$\Sigma_{cc} = \phi(x)^{\top}\Sigma_c\,\phi(x)$。
這個下界讓訓練**不需要取樣**，成本對最後一層寬度是二次的。

**AI 主動提出、我採納的一個警告**：`DiscClassification`（判別式）與 `GenClassification`
（生成式）的**損失函數數值不可互相比較** —— 官方教學明確警告過。這直接決定了實驗設計：
**只比較指標（accuracy / NLL / ECE / AUROC），永遠不比較 loss。**

**兩種 head 的差異**（報告 §2 有完整推導）：

| | `DiscClassification` | `GenClassification` |
|---|---|---|
| 後驗物件 | 最後一層權重 | 每個類別的**輸入**條件高斯 |
| 取得機率 | 對權重取期望後做 softmax 下界 | 類別條件密度 + 貝氏反演 |
| 預測是否隨機 | **是**（內部 20 次 MC 平均，見階段 3c） | 否（確定性） |

---

## 階段 3a — Colab 環境與依賴鎖定

**目標**：讓使用者在 Colab 上一格安裝完成，且**不讓我手寫任何設定**。

### 最關鍵的工程決策

**Colab 上絕不能重裝 torch。** 這是 Colab「`torch.cuda.is_available()` 變成 False」的
頭號原因 —— 一旦 CUDA 版的 torch 被換成 CPU 輪子，整個 GPU runtime 就白開了。

**做法**：安裝前先讀出 Colab **自己的** torch/torchvision 版本，寫進 constraints 檔，
再 `pip install -c colab-constraints.txt ...`。這樣任何傳遞依賴都**無法**把 CUDA 版
torch 換掉，而且**包列表裡永遠不出現 torch 本身**。

### 實測（不是推測）

用 `curl pypi.org/pypi/<pkg>/json` 逐個核對依賴：

| 套件 | 版本 | 依賴 |
|---|---|---|
| `vbll` | 0.4.9 | 只依賴 `torch>=1.6.0`、`numpy>=1.21.0`（極輕） |
| `laplace-torch` | 0.2.3 | `asdfghjkl==0.1a4`（**pin 死在 pre-release**）、`backpack-for-pytorch`、`curvlinops-for-pytorch>=3.0.1`、`torchmetrics`、`opt_einsum` 等 |

`pip install --dry-run` 實測解析成功，會裝 **13 個套件**，且**不動 torch / torchvision**。

### 一個真實的坑（生成器層）

第一個 notebook 生成器把 cell 原始碼寫成 Python 字串字面量，裡面一個 `"\n"` 經過 JSON
轉義後，在產出的 cell 裡變成**真實換行**，導致產物出現
`SyntaxError: unterminated string literal`。

**修法（結構性，不是打補丁）**：生成器內**完全不用反斜線轉義** ——
換行用 `chr(10)`、用多個 `print()` 取代 `print("\n...")`。
到階段 3c 更進一步：**cell 原始碼改存成純文字檔**，生成器只負責「切分 + 語法校驗 + 序列化」，
**轉義層被徹底移除，這類 bug 在構造上不可能再發生**。

---

## 階段 3b — Notebook 架構設計

本階段**刻意只做設計、不寫程式**，先讓架構被審查過再投入實作。

### 最重要的設計決策：適配器（adapter / port）模式

四個臂 forward 回傳的東西**完全不同**：

| 臂 | forward 回傳 | 機率怎麼來 | OOD 分數怎麼來 |
|---|---|---|---|
| (a) MLP + softmax | 原始 logits `(N, C)` | 套 softmax | `1 − max p` |
| (b) VBLL Disc | VBLL output 物件 | 讀 `out.predictive.probs` | 讀 `out.ood_scores` |
| (c) VBLL Gen | VBLL output 物件 | 讀 `out.predictive.probs` | 讀 `out.ood_scores` |
| (d) Laplace LLA | 機率（GLM predictive） | 已正規化 | **沒有原生的** —— 必須自行推導 |

**最顯而易見的實作方式（讓評估函式自己分支）會退化成 `if arm == ...` 階梯**，
而且新增或修改一個臂就會壞掉。所以改用**適配器層**：每個臂包一個薄物件，
只暴露 `predict(loader) -> Prediction`，`evaluate_model` 永遠只看到 `Prediction`。

**`Prediction` 契約**：`probs` / `logits` / `ood_score` / `n` / 選用的不確定性欄位 /
provenance 字典。**不變式（行和為 1、值域 `[0,1]`、無 NaN、長度相符）在「產生它的
adapter 內部」斷言**，這樣壞掉的預測會被具名抓到，而不是在指標裡才爆出來。

**代價與回報**：多寫三層薄包裝；回報是**之後要加第五個臂，只需要寫一個 adapter**，
指標與繪圖程式碼一行都不用改。

### 公平性設計：兩族 OOD 分數

四個臂的 OOD 分數定義不同（`1−max p` / VBLL 的 likelihood 分數 / 推導出的熵）。
**只報一列 AUROC 會把四個不同的量放在一起比。** 因此 `evaluate_model` 輸出兩族：

- `auroc_native` —— 各臂自己的分數（VBLL 的頭條數字）；
- `auroc_entropy` —— **預測熵，四臂定義完全相同**，是唯一同口徑的比較。

### 其他設計決定

- 指標全部是**純陣列函式**（accuracy / nll / ece / auroc），可獨立單測；
- `ece` **同時回傳分箱陣列**，讓 reliability diagram 成為指標的**副產品**而不是另外寫的
  繪圖程式碼 —— 這樣圖與表**不可能不一致**；
- AUROC 用 `sklearn`，**正類 = OOD**；
- cell 順序把「契約」放在訓練之前（13–16 早於 17–18），先定義介面再實作。

---

## 階段 3c — Notebook 完整實作

**產出**：`01_vbll_classification.ipynb`（28 cells = 7 md + 21 code，nbformat 4.0 校驗通過）。

**生成方式**：cell 原始碼以**純文字**存放於 `.workbuddy-ai/scratch/cells.txt`
（用 `@@@CELL MD@@@` / `@@@CELL CODE@@@` 分隔），生成器只做「切分 + `compile()` 校驗 +
`json.dump`」，**完全沒有轉義層**；另做 round-trip 校驗（重新讀回、逐 cell 比對原始碼、再 compile）。

### 從「已安裝套件原始碼」讀出來的 6 個 API 事實

**這是整個階段最有價值的部分。** 這 6 件事**沒有一件**能從文件或名字猜對，
而且其中兩件是**方向性錯誤** —— 會產生看起來很合理的錯數字：

| # | 直覺（錯的） | 原始碼的真相 | 不修會怎樣 |
|---|---|---|---|
| 1 | `out.ood_scores` 越大越像 OOD | 它是 `max_predictive(x)`，**最大預測機率**，越大越像**分布內** | OOD AUROC **掉到 0.5 以下**，結論完全相反 |
| 2 | `out.predictive` 是可呼叫的函式 | 它**已經是 `Categorical` 物件** → 用 `out.predictive.probs` | `TypeError` |
| 3 | `train_loss_fn(y)` 要再取平均 | 它回傳**純量**（內部已 mean） | 再 `.mean()` 會把損失壓成常數，訓練悄悄失效 |
| 4 | VBLL 的預測是「sampling-free」所以確定 | `DiscClassification.predictive()` **內部做 20 次 `rsample` 的 MC 平均** | 指標不可複現；驗證集準確率抖動 → **early stopping 被 MC 噪聲觸發** |
| 5 | `link_approx` 可以填 `"sampled"` | 只接受 `{'mc','probit','bridge','bridge_norm'}` | notebook **最後一步**才 `ValueError` |
| 6 | `GenClassification` 可以用 `parameterization='dense'` | 直接 `NotImplementedError`，只能用 `diagonal` | 開跑就崩 |

**第 1 項的驗證方式**：不是「看起來對」，而是**實測符號** ——
修好之後 VBLL 臂的 native AUROC 跑出 ≈ **0.92**，遠高於 0.5，證明 `1 − ood_scores`
的方向正確。如果沒修，這個數字會是 **0.08 左右**，而且**不會有任何錯誤訊息**。

**第 4 項的修法**：一個小 context manager `deterministic_mc`，在每次 VBLL 預測呼叫期間
釘住 RNG，**呼叫後恢復呼叫者原本的 RNG state**，所以完全不影響訓練。
**驗證**：同一個 adapter 連續兩次評估，NLL 與 AUROC 的差值**恆為 0**。

### 設計文件沒預見到、實作時新增的 4 件事

| # | 新增 | 為什麼 |
|---|---|---|
| 1 | `deterministic_mc` 上下文管理器 | 上面第 4 項的直接後果 |
| 2 | `ece_equal_mass` + 結果表新增 `ece_eqmass` 欄 | 見下一節的發現 |
| 3 | 分箱佔用率打印 | 上面那項的**證據**，讓發現可被檢驗而不是只有結論 |
| 4 | 輸出佈局：圖 → `outputs/figures/`，CSV/JSON → `outputs/` | 不把 CSV 混進圖表目錄 |

### 一個「跑得動」但「是錯的」的發現

我加了分箱佔用率的 instrumentation，量到在 MNIST 上：
**15 個等寬箱有 10 個是空的，頂箱裝了 10,000 個測試樣本中的 9,330 個（93 %）。**
也就是說等寬 ECE **實際上是在量「一個箱」**。

**處理方式**：不是把這個數字藏起來，而是**加一個等質量分箱的替代欄位並列報告**。
誠實的結論是：在這個資料上兩種分箱的數值**差異不大**（VBLL-Disc 是 0.0054 vs 0.0042）
—— **結構上的集中是真的，但在這個準確率水準下並沒有嚴重扭曲數值**。
報告同時寫出兩者，因為**沒有一個是「那個」ECE**。

**這是最值得記下來的一類問題**：指標算出了數字、notebook 畫出了曲線、一切看起來完成了，
但 93 % 的樣本在同一個箱子裡。**沒有人會從輸出裡看出來。**
要發現它，必須刻意去問「**這個數字在什麼情況下會誤導人？**」。

### 其他實作要點

- **一個訓練迴圈、兩個目標**：基線用 `CrossEntropyLoss`；VBLL 臂用 ELBO，
  透過 `out.train_loss_fn(y)` 微分、`out.val_loss_fn(y)` 驗證，只依模型型別分支一次。
- **VBLL head 排除 weight decay**：它的先驗項已經在正則化它，而 ELBO 已按
  `reg_weight = 1/N` 縮放，再加 weight decay 等於**重複計算先驗**。用 parameter groups
  對 head 設 `weight_decay = 0`。
- **Laplace 是 post-hoc**：arm (d) 先用與 arm (a) 完全相同的配方訓練一個普通 MLP，
  再在其外擬合 Laplace 後驗；prior precision 用**驗證集**網格搜尋 ——
  **絕不用測試集，也絕不用 OOD 集**。
- **GLM predictive 分批 + `no_grad`**：這是 notebook 裡最重、最吃記憶體的一步，
  所以預測被包起來分批跑，而不是對整個測試集一次呼叫。
- **`grid_size` 從預設 100 降到 30**：因為 `_gridsearch` 是**逐個 prior_prec 跑一遍完整
  `val_loader`**，成本 ≈ `grid_size` × 驗證集推理。

### 驗證（跑的是**交付的 .ipynb 本身**，不是副本）

| Harness | 做什麼 | 結果 |
|---|---|---|
| `test_classification_notebook.py` | 從交付的 `.ipynb` 抽出全部 21 個 code cell，**用真實 MNIST / Fashion-MNIST 在 CPU 上端到端跑通**，再對產物斷言 33 項屬性 | **33/33 PASS** |
| `test_metrics.py` | 把 4 個指標函式拉出來對**手算答案**，涵蓋 clip、空箱、`conf == 1.0` 邊界、兩種 ECE | **23/23 PASS** |

**斷言內容包括**：`return_ood=False` 會拋**具名** `RuntimeError`、`check_contract` 會拒絕
壞掉的 `Prediction`、VBLL 機率非退化、RNG 可復現、5 張圖與 `results.csv/json` 確實落盤。

### 本機實測數值（4 epochs / CPU / `grid_size=3`）

> **這不是最終 Colab 結果**，只是用來確認形狀與符號方向的冒煙測試。

| arm | acc | nll | ece | ece_eqmass | auroc_native | auroc_entropy |
|---|---|---|---|---|---|---|
| a_mlp_softmax | 0.9747 | 0.0810 | 0.0054 | 0.0052 | 0.9100 | 0.9132 |
| b_vbll_disc | 0.9758 | 0.0779 | 0.0054 | 0.0042 | **0.9222** | 0.9270 |
| c_vbll_gen | 0.9555 | 0.1786 | 0.0297 | 0.0296 | 0.8782 | 0.8888 |
| d_laplace_lla | 0.9764 | 0.0812 | 0.0063 | 0.0068 | 0.8918 | 0.8958 |

**符號約定驗證通過**：`auroc_native > 0.5` 說明 `1 − out.ood_scores` 方向正確；
**VBLL-Disc 的 OOD AUROC 最高**，與論文主張一致。即使只有 4 epochs，結果的「形狀」已經可見。

---

## 階段 3d — 熱修復（由真實報錯驅動）

**使用者報錯**：跑 `01_vbll_classification.ipynb` 的
「Section 0 - environment echo and fail-fast guards」cell 時出現
`ModuleNotFoundError: No module named 'vbll'`。

**根因**：**不是套件壞了，是套件沒了。** Colab 回收/斷開 session 後會丟棄所有 pip
安裝的套件；`00_colab_setup.ipynb` 當時確實裝成功了，但那個 session 已經不存在。

**修復的設計判斷（重要）**：不做「請你先跑 00 再跑 01」這種補救，而是
**給 `01` 在最前面加一個冪等的「依賴自修復」cell**。理由是：
**助教/評分者很可能只打開 `01` 直接 Run All，兩段式流程必炸。**

**這個 cell 的行為**：

1. 先報告當前可匯入狀態；**什麼都不缺就直接返回**（健康 session 只花約 1 秒）；
2. **單獨守衛 `import torch`** —— 若連 torch 都沒了，說明 runtime 本身壞了，
   pip 救不了，直接提示重啟 session；
3. 把**執行期自己的** torch/torchvision 版本（去掉 `+cu124` 這類 local tag）寫進
   `colab-constraints.txt`，再 `pip install -q -c colab-constraints.txt ...`，
   **包列表裡絕不包含 torch 本身**；
4. **清 import 快取**：運行中的 kernel 會快取 `site-packages` 目錄內容，新裝的頂層套件
   看不見 → `importlib.invalidate_caches()` + `sys.path_importer_cache.clear()` +
   刪除 `sys.modules` 裡剛裝套件的相關條目；
5. 校驗所有 import，**並**確認 `torch.__version__` 未變、`torch.cuda.is_available()` 仍為真；
6. 安裝失敗時拋出**可診斷**的錯誤並印出該跑的指令，而不是丟一個裸 traceback；
7. 頂部加 `warnings.filterwarnings("ignore")` —— 因為 `import laplace` 會觸發 torch 的
   deprecation warning，輸出會很髒。

**結構變化**：28 cells → **29 cells（7 md + 22 code）**。自修復 cell 作為 **cell 00b**
插在原本的環境 echo cell 之前。

**新增專項測試**：端到端測試**夠不到**這個分支（本機 vbll/laplace 真的裝著，
cell 會走「什麼都不缺」路徑）。所以單獨寫了 `test_hotfix_install.py`：
把 `importlib.import_module` 對 vbll/laplace 打補丁讓它拋錯（**騙 cell 以為套件沒了**），
再把 `subprocess.run` 攔截成 `--dry-run`，斷言 pip 指令帶 `-c <constraints>`、
constraints 精確 pin 本機 torch/torchvision、**torch 不在包列表裡**、裝不上時會拋具名錯誤。
→ **11/11 PASS**。

**測試健壯性改進**：兩個 harness 原本用**硬編碼 cell 索引**定位 cell ——
這意味著插一個 cell 就會**靜默地測錯對象卻仍然回報成功**。
現改為**用 header 註解裡的唯一 marker 定位**（`find(marker)`，並斷言唯一匹配）。

---

## 驗證總表（最終狀態）

| 項目 | 結果 |
|---|---|
| `nbformat` 4.0 schema 校驗 | 通過 |
| 每個 code cell 在生成時 `compile()` | 全部通過 |
| 端到端（22 個 code cell，真實 MNIST / Fashion-MNIST，CPU） | **37/37 PASS** |
| 指標單元測試 | **23/23 PASS** |
| 熱修復安裝路徑專項測試 | **11/11 PASS** |
| Colab setup notebook 測試 | **10/10 PASS** |

### 這張表**沒有**覆蓋到的：第一次真實 GPU 執行

**所有 harness 都在 CPU 上跑**，這件事後來產生了後果。使用者第一次在 Colab（T4 GPU）
跑 `01_vbll_classification.ipynb` 時，在第 15 個 cell（訓練四個臂 + 擬合 Laplace）崩潰：

```
TypeError: can't convert cuda:0 device type tensor to numpy.
Use Tensor.cpu() to copy the tensor to host memory first.
```

**根因（在我們自己的程式碼裡，不在套件裡）**：
`fit_laplace()` 這一行

```python
pp = float(np.asarray(la.prior_precision).ravel()[0])
```

**laplace-torch 的 `prior_precision` setter 會把張量搬到模型所在的裝置**
（`baselaplace.py`：`prior_precision.to(device=self._device)`），
所以 GPU 上它是 `cuda:0` 張量，`np.asarray` 直接拋 `TypeError`。
**同一行在 CPU 上完全正確** —— 這就是 37/37 全過、卻在 GPU 上死掉的原因。

**定位方式**：Colab traceback 顯示 `run_pipeline` 與 `Tensor.__array__` 之間有
**2 個隱藏 frame**，正好對應 `run_pipeline → fit_laplace → np.asarray`；
再加上「最後一行輸出是 `optimising the prior precision ...`、而下一行的
`selected prior precision:` 沒有印出」—— 兩個線索交叉指向同一行。
（另外確認了 `_gridsearch` 的 `interval = torch.logspace(...)` 在 CPU 上，
`validate()` 回傳 `.item()` 純量，所以套件內部沒有問題。）

**代價**：它死在**最糟的位置** —— 四個臂全部訓練完之後、Laplace 先驗精度那一步。
前面 30 epochs × 4 臂的 GPU 時間全白費。

**修法**：
1. 該行改為 `_pp = la.prior_precision` → `if isinstance(_pp, torch.Tensor): _pp = _pp.detach().cpu()`；
2. 順手補兩處**只是碰巧正確**的轉換：資料健全性檢查的 `grid...numpy()` 加 `.cpu()`、
   `collect_targets()` 的 `np.asarray(y)` 改為先 `.detach().cpu()`。

**系統性排查**：改完後逐一檢查全部 **6 處** `.numpy()` 呼叫，確認**每一處都有 `.cpu()` 前綴**；
並確認整份 notebook 只有那一處是「未受保護的張量→numpy」。

**教訓（本專案最重要的一條）**：
**只在單一裝置上跑的測試套件，等於默默宣告其他裝置不在範圍內。**
37 項斷言全部通過，而那一行在 GPU 上必死 ——
所以**第一次真實硬體執行本身就是驗證的一部分**，而不是驗證之後的手續。


---

## 階段 4 — 消融實驗獨立化，與第二個裝置 bug

### 為什麼要獨立成一份 notebook

消融原本是主 notebook 裡「被 `CFG['run_robustness']` 閘住」的兩格。問題是：要跑它就得
把整份 notebook 的 30 epochs × 4 臂再跑一遍。所以改成獨立一份 `02_vbll_ablations.ipynb`。

**做法是「剪接」，不是重寫。** 以 `cells.txt`（主 notebook 的唯一真實來源）為底，用一份
splice plan 取出已驗證的 cell，所以「資料切分 / backbone / adapters / 訓練迴圈」與主
notebook **逐位元一致**，不可能默默漂移。排除掉 sanity 圖、30-epoch 的 Laplace 擬合、
主評估與匯出、多種子 —— 它**絕不重跑主實驗**。

**踩到的坑（靜默的那種）**：splice plan 裡兩個 extras 索引寫反了，
「Section 6 (continued)」這個 markdown 標頭跑到消融程式碼**前面**，
而且**不會報任何錯**。只有把最終 cell map 印出來才看得見。
→ 教訓：**剪接之後必須斷言或列印 cell map**，順序錯誤是靜默的。

### 第二個裝置 bug：藏在「參數化選擇」後面的上游缺陷

消融要測 `lowrank` / `dense` 兩種協方差參數化。`lowrank` 在 GPU 上直接死：

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

出在 **vbll 套件自己**的 `vbll/utils/distributions.py`：

```python
# LowRankNormal.logdet_covariance（第 327 行）
term2 = torch.linalg.det(arg1 + torch.eye(arg1.shape[-1])).log()
# cholesky_inverse（第 27 行）
torch.cholesky_solve(torch.eye(u.size(-1)).expand(u.size()), u, upper=upper)
```

`torch.eye(n)` 沒帶 `device=`，所以建立在 CPU；而 `arg1` 由 head 的 `cov_factor`／`cov_diag`
組成，`build_vbll_model` 已經把整個模型 `.to(DEVICE)`，所以它在 `cuda:0`。

**三個值得記下的點**：

1. **這是上游 bug，不是生成程式碼的 bug** —— notebook 呼叫方式完全正確。
2. **它對 `diagonal` 完全隱形**，而 `diagonal` 正是預設值：`diagonal` 對應 vbll 的
   `Normal`，其 `logdet_covariance` 只是 `2 * log(scale).sum(-1)`，根本不會碰到 `torch.eye`。
   這正是為什麼主實驗毫髮無傷、只有消融的某一臂炸掉 ——
   **缺陷是「取決於某個設定值」的**，所以再怎麼重跑預設設定都暴露不出來。
3. 直接改 `site-packages` 不是可持續的修法：Colab 每次回收 session 都會清掉已安裝套件。

**修法：device-safety shim**，在 import 時安裝一次，把這兩個函式換成
「唯一差別是 identity matrix 會繼承 `device` 與 `dtype`」的等價版本。
**已驗證與原版數值完全相同**（最大絕對誤差 `0.0`），且梯度路徑（ELBO 依賴的那條）保持完好。

### 修補方式：就地改，不重新產生

主 notebook `01` 內含**真實 Colab 執行輸出**（訓練日誌、結果表、四張圖），
重新產生會把每一格的 `outputs` 清空、毀掉報告所依據的證據。
所以改成**只替換那一格的 source、其餘逐位元保留**：

- 先備份 → 解析 JSON → 只改「Section 2 - arms (b) and (c)」那格的 `source`
  （內容取自更新後的 `cells.txt`，確保與產生器一致）；
- **斷言只有那一格變動**，並比對：`execution_count` 全部不變、
  **5 個 image 輸出全部完好**、metadata 不變。

結果：**只有 cell 11 的 source 改變**，其餘 28 格與所有輸出原封不動。
（檔案大小 546,860 → 540,243 bytes 只是 JSON 序列化格式差異 ——
Colab 存檔的縮排／跳脫方式與產生器不同，內容不變。）

### 交付：`GITOUTPUT/`

把要上 GitHub 的檔案整理成 22 個檔、1.2 MB 的目錄樹，含 README、MIT LICENSE、`.gitignore`。
報告放在 `outputs/` 而非根目錄，這樣它內文的 `figures/*.png` 相對連結剛好指向
`outputs/figures/`，**零斷鏈**（已用腳本驗證 README 與報告的每個相對連結都存在）。
三個 notebook 皆通過 `nbformat.validate`。

**用複製而非移動**（原檔留在原位，避免破壞 Colab 工作流），並向使用者說明。


---

## 目前狀態與待辦

**已完成**：演算法選型、數學描述、Colab 環境鎖定、架構設計、完整實作、三重驗證、
依賴自修復；報告 §1、§2、§3、§4.1–§4.2、§5 已完稿。

**待 Colab 正式跑完填入**：§4.3 主結果、§4.4 消融、§4.1 的真實環境版本、§3.6 的
Colab 執行列。

**待決定**：§6 的源碼託管方式（GitHub 倉庫）。

**待產出**：`CISC3024_AI2_Report.docx` / `.pdf`。

---

## 一句話總結

AI 在**架構**上表現優秀，在**介面**上不可靠。
本專案記錄到的每一個實質錯誤，都是介面錯誤 —— 一個名字、一個符號、一個回傳型別、
一個可接受的字串值。**沒有一個是靠「讀程式碼」抓到的，全部是靠「跑它並讀輸出」抓到的**，
而其中兩個（OOD 分數方向、MC 取樣）**如果不抓，產出的會是看起來完全合理的錯數字**。
