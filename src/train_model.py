"""
train_model.py
--------------
Trains a Random Forest classifier on the cleaned city_day dataset to
predict AQI_Bucket (6-class air quality category). Saves:
  - models/aqi_model.pkl        : trained pipeline (scaler + RF)
  - models/feature_names.pkl    : ordered feature list
  - models/label_encoder.pkl    : LabelEncoder
  - models/model_metrics.json   : accuracy, per-class F1, confusion matrix

Usage:
    python src/train_model.py
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
DATA_PATH     = "data/processed_city_day.csv"
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "aqi_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
LE_PATH       = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH  = os.path.join(MODEL_DIR, "model_metrics.json")

FEATURE_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx",
                "NH3", "CO", "SO2", "O3", "Benzene", "Toluene",
                "Month", "DayOfYear"]
TARGET_COL   = "AQI_Bucket"
BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"[INFO] Loaded processed data: {df.shape[0]} rows")
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.Series, LabelEncoder]:
    le = LabelEncoder()
    le.classes_ = np.array(BUCKET_ORDER)   # enforce consistent order
    y = le.transform(df[TARGET_COL])
    print(f"[INFO] Classes: {list(le.classes_)}")
    return pd.Series(y), le


def split_data(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    rf),
    ])
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
             le: LabelEncoder) -> dict:
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average="weighted")
    f1_mac = f1_score(y_test, y_pred, average="macro")
    cm     = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    print(f"\n[RESULT] Test Accuracy : {acc:.4f}")
    print(f"[RESULT] Weighted F1   : {f1_w:.4f}")
    print(f"[RESULT] Macro F1      : {f1_mac:.4f}")
    print("\nPer-class report:")
    for cls in le.classes_:
        r = report[cls]
        print(f"  {cls:<15} precision={r['precision']:.3f}  "
              f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}  "
              f"support={int(r['support'])}")

    return {
        "accuracy":   round(acc,   4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro":   round(f1_mac,4),
        "confusion_matrix": cm,
        "class_report": {
            k: {m: round(v, 4) if isinstance(v, float) else int(v)
                for m, v in vd.items()}
            for k, vd in report.items()
            if k in le.classes_
        },
        "classes": BUCKET_ORDER,
    }


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\n[CV]  5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    return float(scores.mean())


def feature_importance(pipeline: Pipeline, feature_names: list) -> list:
    rf  = pipeline.named_steps["clf"]
    imp = rf.feature_importances_
    ranked = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
    print("\n[INFO] Top-5 feature importances:")
    for name, score in ranked[:5]:
        print(f"  {name:<12} : {score:.4f}")
    return [{"feature": n, "importance": round(float(v), 6)} for n, v in ranked]


def save_artifacts(pipeline, le, feature_names, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    with open(LE_PATH, "wb") as f:
        pickle.dump(le, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Model saved       -> {MODEL_PATH}")
    print(f"[INFO] Feature names     -> {FEATURES_PATH}")
    print(f"[INFO] Label encoder     -> {LE_PATH}")
    print(f"[INFO] Metrics           -> {METRICS_PATH}")


def main():
    df = load_data(DATA_PATH)

    # features present after cleaning
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y, le = encode_labels(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline()
    print("\n[INFO] Training Random Forest …")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete")

    metrics = evaluate(pipeline, X_test, y_test, le)
    cv_acc  = cross_validate(pipeline, X, y)
    metrics["cv_accuracy_mean"] = round(cv_acc, 4)
    metrics["feature_importances"] = feature_importance(pipeline, available)
    metrics["features_used"] = available

    save_artifacts(pipeline, le, available, metrics)
    print("\n[DONE] All artifacts saved successfully.")


if __name__ == "__main__":
    main()
