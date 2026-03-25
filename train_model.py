"""
train_model.py
==============
Train a Random Forest model for landslide susceptibility.
Outputs: model.joblib, model_metrics.json, feature_importance.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 60)
    print("LANDSLIDE SUSCEPTIBILITY — MODEL TRAINING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    csv_path = os.path.join(BASE_DIR, "training_data.csv")
    meta_path = os.path.join(BASE_DIR, "feature_metadata.json")

    df = pd.read_csv(csv_path)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_names = meta["features"]
    X = df[feature_names].values
    y = df["label"].values

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

    # ------------------------------------------------------------------
    # 2. Train/test split  (stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Train Random Forest
    # ------------------------------------------------------------------
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n--- Evaluation on Test Set ---")
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Landslide','Landslide'])}")

    # ------------------------------------------------------------------
    # 5. Cross-validation
    # ------------------------------------------------------------------
    print("--- 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  AUC scores: {cv_scores}")
    print(f"  Mean AUC:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Feature importance
    # ------------------------------------------------------------------
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    print("\n--- Feature Importance ---")
    fi_data = {}
    for i in idx:
        print(f"  {feature_names[i]:20s}  {importances[i]:.4f}")
        fi_data[feature_names[i]] = float(importances[i])

    # ------------------------------------------------------------------
    # 7. Save model & metrics
    # ------------------------------------------------------------------
    model_path = os.path.join(BASE_DIR, "model.joblib")
    joblib.dump(rf, model_path)
    print(f"\nModel saved to: {model_path}")

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": {"TN": int(cm[0,0]), "FP": int(cm[0,1]),
                             "FN": int(cm[1,0]), "TP": int(cm[1,1])},
        "feature_importance": fi_data,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_estimators": 300,
    }
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
