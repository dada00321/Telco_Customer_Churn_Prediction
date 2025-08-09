# This script searches for 'prediction.csv' in common locations,
# computes accuracy against the ground-truth 'Churn' column,
# and prints a short summary. It ignores rows without predictions.
import os
import pandas as pd
import numpy as np

def to_binary(series):
    # Robustly convert Yes/No or 1/0 (string/numeric) to 0/1 integers
    if series.dtype.kind in "biufc":
        # numeric-like
        return (series.astype(float) > 0.5).astype(int)
    # string-like
    s = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    return s.map(mapping)

path = "prediction.csv"
if path is None:
    print("❌ 找不到 prediction.csv。請把檔案放在：\n- 工作目錄同一層，或\n- data/prediction.csv，或\n- /mnt/data/prediction.csv\n然後再執行一次。")
else:
    df = pd.read_csv(path)
    # Keep rows with predictions
    if "Churn_Pred" not in df.columns:
        print(f"❌ 在 {path} 裡找不到 'Churn_Pred' 欄位。請先用 infer.py 產生 prediction.csv。")
    elif "Churn" not in df.columns:
        print(f"❌ 在 {path} 裡找不到 'Churn' 欄位。請確認 prediction.csv 是由原始資料加入預測後輸出。")
    else:
        valid = df["Churn_Pred"].notna()
        n_total = len(df)
        n_eval = int(valid.sum())
        n_dropped = n_total - n_eval

        if n_eval == 0:
            print("❌ 檔案裡沒有可用的預測（Churn_Pred 全為空）。")
        else:
            y_true = to_binary(df.loc[valid, "Churn"]).astype(int)
            y_pred = df.loc[valid, "Churn_Pred"].astype(int)

            # Remove potential NaNs after mapping
            mask_ok = y_true.notna()
            y_true = y_true[mask_ok]
            y_pred = y_pred[mask_ok]

            acc = (y_true.values == y_pred.values).mean()

            # Confusion matrix counts
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

            print(f"✅ 準確率 Accuracy: {acc:.4f}")
            print(f"- 評估筆數: {len(y_true)}（共 {n_total} 筆，忽略無法預測 {n_dropped} 筆）")
            print(f"- 混淆矩陣: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
