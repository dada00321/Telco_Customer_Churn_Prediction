# infer.py
# Load artifacts and run inference on a new CSV. If the CSV also has Churn, will print accuracy metrics.

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib

from preprocess import Preprocessor, TARGET_NAME, encode_target_column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to input CSV for inference")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts", help="Dir that contains preprocessor.joblib & model.joblib & meta.json")
    ap.add_argument("--out", type=str, default="prediction.csv")
    args = ap.parse_args()

    art_dir = Path(args.artifacts_dir)
    meta = json.loads((art_dir / "meta.json").read_text(encoding="utf-8"))
    thr = float(meta.get("threshold", 0.5))

    pp = Preprocessor.load(art_dir)
    model = joblib.load(art_dir / "model.joblib")

    df = pd.read_csv(args.csv)
    df_enc = encode_target_column(df.copy(), TARGET_NAME)  # will keep Churn if present (as 0/1)

    X = pp.transform(df_enc)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= thr).astype(int)

    out_df = df.copy()
    out_df["Churn_Prob"] = probs
    out_df["Churn_Pred"] = preds

    # If ground-truth exists, print quick metrics
    if TARGET_NAME in df_enc.columns:
        y_true = df_enc[TARGET_NAME].values.astype(int)
        print("===== Metrics on provided file =====")
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = np.nan
        print(f"accuracy: {accuracy_score(y_true, preds):.4f}")
        print(f"f1: {f1_score(y_true, preds):.4f}")
        print(f"precision: {precision_score(y_true, preds):.4f}")
        print(f"recall: {recall_score(y_true, preds):.4f}")
        print(f"auc: {auc:.4f}")

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions -> {out_path.resolve()}")


if __name__ == "__main__":
    main()