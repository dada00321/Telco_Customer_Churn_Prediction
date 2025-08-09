# preprocess.py
# Robust preprocessing for IBM Telco Churn dataset
# - Cleans TotalCharges blanks -> float
# - Encodes categorical features with OneHot (handle_unknown='ignore')
# - Scales numeric features
# - Exposes a sklearn ColumnTransformer pipeline
# - Saves/loads artifacts via joblib + meta json

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Default aliases seen in various shared copies of the dataset
COLUMN_ALIASES = {
    "tenure(終身職位)": "tenure",
    "tenure(任期)": "tenure",
    "TotalCharges ": "TotalCharges",
}

NUMERIC_CANDIDATES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ID_COLUMNS = ["customerID"]
TARGET_NAME = "Churn"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace and unify duplicates
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # apply aliases
    for src, dst in COLUMN_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    return df


def _coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        # Some rows have ' ' (single space) -> treat as NaN then impute
        df["TotalCharges"] = (
            pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
        )
    return df


def _ensure_target_binary(y: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    # Accept 'Yes'/'No', 'Y'/'N', 1/0, True/False; map to {No:0, Yes:1}
    y_clean = y.copy()
    mapping = None
    if y_clean.dtype == object:
        y_norm = y_clean.astype(str).str.strip().str.lower()
        mapping = {"no": 0, "yes": 1, "0": 0, "1": 1, "false": 0, "true": 1}
        y_clean = y_norm.map(mapping)
    else:
        y_clean = y_clean.astype(float)
    y_clean = y_clean.fillna(0).astype(int)
    if mapping is None:
        mapping = {"No": 0, "Yes": 1}
    return y_clean, mapping


@dataclass
class PreprocessorArtifacts:
    numeric_features: List[str]
    categorical_features: List[str]
    feature_names_out: List[str]
    target_name: str = TARGET_NAME


class Preprocessor:
    def __init__(self):
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_columns: List[str] = []
        self.artifacts: Optional[PreprocessorArtifacts] = None
        self.label_mapping_: Dict[str, int] = {"No": 0, "Yes": 1}

    def fit(self, df_raw: pd.DataFrame, target: str = TARGET_NAME) -> Tuple[np.ndarray, pd.Series]:
        df = _normalize_columns(df_raw)
        df = _coerce_total_charges(df)

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        y_raw = df[target]
        y, mapping = _ensure_target_binary(y_raw)
        self.label_mapping_ = mapping

        X = df.drop(columns=[target])
        # Identify numeric & categorical features
        numeric_features = [c for c in NUMERIC_CANDIDATES if c in X.columns]
        categorical_features = [
            c for c in X.columns if c not in numeric_features and c not in ID_COLUMNS
        ]
        self.feature_columns = list(numeric_features + categorical_features)

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", categorical_pipe, categorical_features),
            ]
        )
        X_mat = self.pipeline.fit_transform(X)

        # Save feature names
        feature_names = []
        if numeric_features:
            feature_names.extend(numeric_features)
        if categorical_features:
            ohe = self.pipeline.named_transformers_["cat"].named_steps["ohe"]
            ohe_names = list(ohe.get_feature_names_out(categorical_features))
            feature_names.extend(ohe_names)

        self.artifacts = PreprocessorArtifacts(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            feature_names_out=feature_names,
        )
        return X_mat, y

    def transform(self, df_raw: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
        df = _normalize_columns(df_raw)
        df = _coerce_total_charges(df)
        # drop target if present
        if TARGET_NAME in df.columns:
            df = df.drop(columns=[TARGET_NAME])
        # keep only known columns (others will be ignored by ColumnTransformer)
        return self.pipeline.transform(df)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_dir / "preprocessor.joblib")
        meta = {
            "feature_columns": self.feature_columns,
            "target_name": TARGET_NAME,
            "label_mapping": self.label_mapping_,
            "artifacts": asdict(self.artifacts) if self.artifacts else None,
        }
        (out_dir / "meta_preprocess.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(out_dir: str | Path) -> "Preprocessor":
        out_dir = Path(out_dir)
        obj = Preprocessor()
        obj.pipeline = joblib.load(out_dir / "preprocessor.joblib")
        if (out_dir / "meta_preprocess.json").exists():
            meta = json.loads((out_dir / "meta_preprocess.json").read_text(encoding="utf-8"))
            obj.feature_columns = meta.get("feature_columns", [])
            obj.label_mapping_ = meta.get("label_mapping", {"No":0, "Yes":1})
            art = meta.get("artifacts")
            if art:
                obj.artifacts = PreprocessorArtifacts(**art)
        return obj


def encode_target_column(df: pd.DataFrame, target: str = TARGET_NAME) -> pd.DataFrame:
    df = _normalize_columns(df)
    if target in df.columns:
        y = df[target]
        y_enc, _ = _ensure_target_binary(y)
        df[target] = y_enc
    return df