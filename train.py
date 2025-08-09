# train.py
# Train a churn model with robust preprocessing, threshold tuning, and artifact export

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import joblib

from preprocess import Preprocessor, TARGET_NAME, encode_target_column


def choose_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1") -> float:
    # Search 200 thresholds between 0.05 and 0.95 to maximize selected metric
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        if metric == "f1":
            s = f1_score(y_true, preds)
        elif metric == "recall":
            s = recall_score(y_true, preds)
        elif metric == "precision":
            s = precision_score(y_true, preds)
        elif metric == "balanced_accuracy":
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            s = ((tp / (tp + fn + 1e-9)) + (tn / (tn + fp + 1e-9))) / 2
        else:
            s = f1_score(y_true, preds)
        if s > best_score:
            best_score, best_thr = s, thr
    return float(best_thr)


def build_model(model_type: str) -> Any:
    if model_type == "xgb" and HAS_XGB:
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    else:
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )


def evaluate(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    preds = (probs >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds)),
        "recall": float(recall_score(y_true, preds)),
        "auc": float(roc_auc_score(y_true, probs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to Telco-Customer-Churn CSV")
    ap.add_argument("--output_dir", type=str, default="artifacts", help="Where to save model & preprocessors")
    ap.add_argument("--model", type=str, choices=["xgb", "rf", "logreg"], default="xgb")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--metric", type=str, default="f1", help="threshold selection metric: f1/recall/precision/balanced_accuracy")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = encode_target_column(df, TARGET_NAME)

    # Preprocess fit
    pp = Preprocessor()
    X_all, y_all = pp.fit(df, target=TARGET_NAME)

    # Hold-out split for a fair test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=args.test_size, stratify=y_all, random_state=args.random_state
    )

    model = build_model(args.model)

    # Warm CV check to reduce variance & mimic "no overfitting" claim
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_probs = np.zeros_like(y_tr, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr), start=1):
        X_tr_f, X_val_f = X_tr[tr_idx], X_tr[val_idx]
        y_tr_f, y_val_f = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        clf = build_model(args.model)
        clf.fit(X_tr_f, y_tr_f)
        cv_probs[val_idx] = clf.predict_proba(X_val_f)[:, 1]

    # Choose threshold on out-of-fold CV predictions (more honest than on test)
    best_thr = choose_threshold(y_tr.values, cv_probs, metric=args.metric)

    # Fit final model on the entire training split
    model.fit(X_tr, y_tr)
    test_probs = model.predict_proba(X_te)[:, 1]

    metrics = evaluate(y_te.values, test_probs, best_thr)
    print("===== Test Metrics (hold-out) =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Best threshold (@{args.metric}): {best_thr:.3f}")

    # Save artifacts
    pp.save(out_dir)
    joblib.dump(model, out_dir / "model.joblib")
    meta = {
        "model_type": args.model if (args.model != "xgb" or HAS_XGB) else "xgb",
        "threshold": best_thr,
        "test_metrics": metrics,
        "random_state": args.random_state,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()