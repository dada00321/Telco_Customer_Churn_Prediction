# 準確率（使用 RandomForest 模型）
![準確率](https://github.com/user-attachments/assets/695439f4-568e-408c-b9f0-1d4d205cfefb)

# Telco Customer Churn – 可重現的訓練與推論管線（Python）

以 IBM/Kaggle 常見的 **Telco-Customer-Churn** 資料集為例，將 Kaggle 筆記本「*Telco Churn 🎯: 97% Accuracy with No Overfitting*」的重點做法，整理成三支可直接落地的腳本：

- `preprocess.py`：資料清理與特徵工程（含缺值處理、One-Hot、標準化）
- `train.py`：模型訓練、交叉驗證、門檻最佳化、產出 artifacts
- `infer.py`：載入 artifacts，對新檔預測並輸出 `prediction.csv`

> ✅ 特色：
>
> - **缺值處理**：對 `TotalCharges` 空白/非數字自動轉 NaN 並以中位數補值；類別欄以眾數補值。
> - **穩健評估**：使用 5-fold Stratified CV 的 out-of-fold 機率來選擇最佳門檻（避免在測試集上調門檻）。
> - **可攜 artifacts**：輸出 `preprocessor.joblib`、`model.joblib`、`meta.json`、`meta_preprocess.json`，方便部署。
> - **彈性模型**：支援 `xgboost` / `random_forest` / `logreg` 三種基準模型。

---

## 目錄

- [專案結構](#專案結構)
- [環境需求](#環境需求)
- [資料準備](#資料準備)
- [快速開始](#快速開始)
- [評估指標與門檻最佳化](#評估指標與門檻最佳化)
- [常見問題（FAQ）](#常見問題faq)
- [再現結果](#再現結果)
- [自訂與擴充](#自訂與擴充)
- [授權條款](#授權條款)

---

## 專案結構

```
.
├─ preprocess.py
├─ train.py
├─ infer.py
├─ data/
│   └─ Telco-Customer-Churn.csv     # 請將資料放這（或自訂路徑）
├─ artifacts/                       # 訓練後自動生成（模型與前處理）
│   ├─ preprocessor.joblib
│   ├─ model.joblib
│   ├─ meta.json
│   └─ meta_preprocess.json
└─ README.md
```

## 環境需求

- Python 3.9+（3.10/3.11 亦可）
- 主要套件：`pandas`、`numpy`、`scikit-learn`、`joblib`、（可選）`xgboost`

**安裝**

```bash
pip install -U pandas numpy scikit-learn joblib
# 可選：
pip install -U xgboost
```

> Windows 使用者如遇到安裝 `xgboost` 困難，可先改用 `--model rf`（RandomForest）。

## 資料準備

- 預設假設檔名為 `Telco-Customer-Churn.csv`；也支援有些版本會出現中文別名欄位，例如 `tenure(終身職位)`，程式會自動轉為 `tenure`。
- 常見欄位：`customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`。
- `Churn` 可為 `Yes/No` 或 `1/0`；程式會自動轉為 0/1。

## 快速開始

### 1) 訓練

```bash
python train.py --csv data/Telco-Customer-Churn.csv   --output_dir artifacts   --model xgb   --test_size 0.2   --metric f1   --random_state 42
```

> 若未安裝 `xgboost`：`--model rf`。`--metric` 可改為 `balanced_accuracy` 或 `accuracy`（見下節）。

### 2) 推論

```bash
python infer.py --csv data/Telco-Customer-Churn.csv   --artifacts_dir artifacts   --out prediction.csv
```

- 會在輸出檔新增兩欄：`churn_prob`（流失機率）與 `prediction`（0/1）。
- 若輸入檔含 `Churn` 欄位，會在終端顯示 accuracy / F1 / precision / recall / AUC。

## 評估指標與門檻最佳化

- 訓練時會用 **5-fold CV 的 out-of-fold 機率** 來尋找最佳門檻（threshold），避免在測試集上「偷看」而過度樂觀。
- 可選門檻最佳化指標：
  - `--metric f1`（預設）：偏重正負類平衡
  - `--metric balanced_accuracy`：在類別不平衡時更穩定的整體準確率
  - `--metric accuracy`：純粹追求整體正確率
  - `--metric recall` / `--metric precision`：依商業需求調整

**想衝更高準確率**：

```bash
python train.py --csv data/Telco-Customer-Churn.csv --model xgb --metric balanced_accuracy
# 或
python train.py --csv data/Telco-Customer-Churn.csv --model xgb --metric accuracy
```

## 常見問題（FAQ）

**Q1. `TotalCharges` 有空白或非數字，為什麼還能預測？**  
A1. 前處理會將空白轉為 `NaN`，並在訓練時以 **中位數** 補值；推論時使用同一組統計量補回，因此模型可正常輸入預測。

**Q2. 如果 `tenure` 是 0（新用戶），`TotalCharges` 常為空白，有更貼近商業的補法嗎？**  
A2. 可以改為：若 `tenure==0` 且 `TotalCharges` 為 NaN，則以 0 取代；否則仍用中位數。此改動可在 `preprocess.py` 中調整。

**Q3. 為什麼我的成績和別人的 Kaggle 筆記本不同？**  
A3. Kaggle Notebook 常用不同切分或（不小心）資料洩漏；本專案固定 hold-out、以 CV 選門檻，較接近真實表現。

**Q4. 我可以只用 `MonthlyCharges`、`Contract` 等少數欄位嗎？**  
A4. 可以，但指標可能下降。建議至少保留 `tenure`、`MonthlyCharges`、`TotalCharges` 與關鍵服務欄位。

## 再現結果

> 以下為範例（不同環境與隨機種子會略有差異）：

| Dataset                      | Model   | Metric for Threshold | Accuracy   | F1         | Precision  | Recall     | AUC        |
| ---------------------------- | ------- | -------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Telco-Customer-Churn (80/20) | XGBoost | F1                   | 0.81~0.84 | 0.58~0.63 | 0.63~0.68 | 0.53~0.60 | 0.84~0.87 |

### 最新實測（RandomForest）
- 評估資料：整份 `Telco-Customer-Churn.csv`（N=7043）
- 模型：RandomForest（`--model rf`）
- 準確率：**0.9253**
- 混淆矩陣：TP=1780，TN=4737，FP=437，FN=89
- 先前 hold‑out（XGBoost，F1 門檻）：Accuracy ≈ **0.8161**
- 備註：此為在**整份資料**上評估之結果，與上表的 hold‑out 測試定義不同；若需嚴謹對比請以固定切分或交叉驗證為準。

## 自訂與擴充

- **換模型**：`--model logreg`（邏輯迴歸）或 `--model rf`（隨機森林）。
- **特徵工程**：可在 `preprocess.py` 的 `fit()` 內加入新特徵（例如 `charges_per_tenure`、`addons_count`、交互作用或分箱）。
- **調參建議（XGBoost）**：`n_estimators=800~1200`、`learning_rate=0.03~0.06`、`max_depth=3~5`、`min_child_weight=3~6`、`subsample=0.7~0.9`、`colsample_bytree=0.7~0.9`。
- **再訓練**：更新資料後重跑 `train.py` 即會覆蓋 `artifacts/`。
