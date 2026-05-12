"""
train_models.py
---------------
Trains RandomForest, SVM, and XGBoost classifiers on genome_dataset.csv.
Compares models on accuracy, precision, recall, and F1 (macro).
Saves the best model to models/best_model.pkl and metrics to data/model_metrics.csv.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Install via: pip install xgboost", stacklevel=2)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "genome_dataset.csv"
MODELS_DIR  = BASE_DIR / "models"
METRICS_PATH = BASE_DIR / "data" / "model_metrics.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
SEPARATOR = "=" * 65

def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """
    Run stratified K-Fold cross-validation and collect mean ± std metrics.
    Falls back to leave-one-out when samples < cv to handle small datasets.
    """
    n_samples = len(y)
    effective_cv = min(cv, n_samples)          # never request more folds than samples
    if effective_cv < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}.")

    skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=42)

    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision_macro",
        "recall":    "recall_macro",
        "f1":        "f1_macro",
    }

    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    row = {"Model": name}
    for metric in ("accuracy", "precision", "recall", "f1"):
        vals = cv_results[f"test_{metric}"]
        row[f"{metric.capitalize()} (mean)"] = round(float(np.mean(vals)), 4)
        row[f"{metric.capitalize()} (std)"]  = round(float(np.std(vals)),  4)

    return row


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── 1. Load dataset ────────────────────────────────────────────────────────
    section("Loading dataset")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"  Rows × Cols : {df.shape}")
    print(f"  Label dist  :\n{df['label'].value_counts().to_string()}")

    drop_cols = [c for c in ("filename", "label") if c in df.columns]
    X = df.drop(columns=drop_cols).values.astype(float)
    y = df["label"].values.astype(int)
    feature_names = df.drop(columns=drop_cols).columns.tolist()
    print(f"  Feature cols: {len(feature_names)}")

    # ── 2. Define model zoo ────────────────────────────────────────────────────
    section("Defining models")

    n_samples = len(y)
    n_estimators = max(50, min(200, n_samples * 10))   # scale with data size

    models: dict[str, object] = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ]),
    }

    if XGBOOST_AVAILABLE:
        # Compute scale_pos_weight for binary imbalance robustness
        neg, pos = np.bincount(y)
        spw = neg / pos if pos > 0 else 1.0
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=spw,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ])
    else:
        print("  [!] XGBoost skipped (not installed).")

    for name in models:
        print(f"  [+] {name}")

    # ── 3. Cross-validated evaluation ─────────────────────────────────────────
    section("Cross-validated evaluation (StratifiedKFold)")

    CV_FOLDS = 5
    results: list[dict] = []

    for name, model in models.items():
        print(f"\n  > {name} ...", end=" ", flush=True)
        row = evaluate_model(name, model, X, y, cv=CV_FOLDS)
        results.append(row)
        print(
            f"Acc={row['Accuracy (mean)']:.4f}  "
            f"P={row['Precision (mean)']:.4f}  "
            f"R={row['Recall (mean)']:.4f}  "
            f"F1={row['F1 (mean)']:.4f}"
        )

    metrics_df = pd.DataFrame(results)

    # Pretty print full table
    section("Metrics summary")
    print(metrics_df.to_string(index=False))

    # ── 4. Save metrics CSV ────────────────────────────────────────────────────
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"\n  Metrics saved → {METRICS_PATH}")

    # ── 5. Identify best model (by mean F1) ───────────────────────────────────
    section("Best model selection")

    best_idx  = metrics_df["F1 (mean)"].idxmax()
    best_name = metrics_df.loc[best_idx, "Model"]
    best_f1   = metrics_df.loc[best_idx, "F1 (mean)"]
    print(f"  Best model : {best_name}  (F1 = {best_f1:.4f})")

    # ── 6. Refit best model on full dataset ───────────────────────────────────
    section(f"Refitting {best_name} on full dataset")

    best_pipeline = models[best_name]
    best_pipeline.fit(X, y)

    # Full-dataset metrics for the saved artifact
    y_pred = best_pipeline.predict(X)
    print(f"\n  Full-dataset performance ({best_name}):")
    print(classification_report(y, y_pred, target_names=["Non-Harmful", "Harmful"]))
    print("  Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    # ── 7. Persist best model ──────────────────────────────────────────────────
    section("Saving best model")

    model_artifact = {
        "model_name":    best_name,
        "pipeline":      best_pipeline,
        "feature_names": feature_names,
        "cv_metrics":    metrics_df.loc[best_idx].to_dict(),
        "label_map":     {0: "Non-Harmful", 1: "Harmful"},
    }

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Best model saved → {BEST_MODEL_PATH}")

    section("Done [OK]")
    print(f"  model_metrics.csv → {METRICS_PATH}")
    print(f"  best_model.pkl    → {BEST_MODEL_PATH}\n")


if __name__ == "__main__":
    main()
