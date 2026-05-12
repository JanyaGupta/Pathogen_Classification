"""
train_model.py
--------------
Loads the genome k-mer feature dataset, trains a Random Forest classifier,
evaluates its performance, and saves the trained model to disk.

Run this AFTER process_genomes.py has generated data/genome_dataset.csv.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
DATA_PATH  = BASE_DIR / "data" / "genome_dataset.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "pathogen_model.pkl"


# ---------------------------------------------------------------------------
# Step 1 — Load the dataset
# ---------------------------------------------------------------------------
def load_data(csv_path: Path):
    """Read the CSV produced by process_genomes.py."""
    print(f"[INFO] Loading dataset from: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Please run process_genomes.py first."
        )

    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset loaded — shape: {df.shape}")
    print(f"       Harmful (1): {(df['label'] == 1).sum()} samples")
    print(f"       Non-harmful (0): {(df['label'] == 0).sum()} samples")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Separate features (X) and labels (y)
# ---------------------------------------------------------------------------
def split_features_labels(df: pd.DataFrame):
    """
    Drop non-feature columns and return X (features) and y (labels).
    'filename' is metadata, not a feature; 'label' is our target.
    """
    # Drop the filename column — it's not a useful feature
    X = df.drop(columns=["filename", "label"])

    # The label column is what we want to predict
    y = df["label"]

    print(f"\n[INFO] Features : {X.shape[1]} columns (3-mer frequencies)")
    print(f"[INFO] Samples  : {X.shape[0]}")
    return X, y


# ---------------------------------------------------------------------------
# Step 3 — Train / test split
# ---------------------------------------------------------------------------
def get_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and test sets.
    - test_size=0.2  → 20 % of data used for testing
    - stratify=y     → keeps label proportions the same in both splits
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,   # stratify only when possible
    )

    print(f"\n[INFO] Train samples : {len(X_train)}")
    print(f"[INFO] Test  samples : {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 4 — Train the Random Forest model
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train a Random Forest classifier.
    - n_estimators : number of decision trees in the forest
    - random_state : seed for reproducibility
    """
    print("\n[INFO] Training Random Forest classifier ...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,      # use all available CPU cores
    )
    model.fit(X_train, y_train)

    print("[INFO] Training complete.")
    return model


# ---------------------------------------------------------------------------
# Step 5 — Evaluate the model
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Run predictions on the test set and print key metrics:
    - Accuracy           : overall correct predictions
    - Classification report : precision, recall, F1 per class
    - Confusion matrix   : TP / TN / FP / FN counts
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  Model Evaluation Results")
    print("=" * 60)

    print(f"\nAccuracy : {acc:.4f}  ({acc * 100:.2f} %)\n")

    print("Classification Report:")
    print("-" * 40)
    print(
        classification_report(
            y_test, y_pred,
            target_names=["Non-Harmful (0)", "Harmful (1)"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':20s}  Predicted 0   Predicted 1")
    print(f"  {'Actual 0':20s}  {cm[0][0]:<12d}  {cm[0][1]}")
    print(f"  {'Actual 1':20s}  {cm[1][0]:<12d}  {cm[1][1]}")
    print()


# ---------------------------------------------------------------------------
# Step 6 — Save the trained model
# ---------------------------------------------------------------------------
def save_model(model, model_path: Path):
    """Persist the trained model to disk using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Model saved to: {model_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Pathogen Classification — Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Separate features and labels
    X, y = split_features_labels(df)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    # 4. Train the model
    model = train_model(X_train, y_train)

    # 5. Evaluate
    evaluate_model(model, X_test, y_test)

    # 6. Save model
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
