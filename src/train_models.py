"""
train_models.py
-----------------
Trains four classifiers on the engineered credit-card-approval dataset:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. XGBoost (falls back to sklearn's GradientBoostingClassifier if the
       xgboost package is not installed, so this script always runs)

Evaluates each on a held-out test set (accuracy, precision, recall, F1,
ROC-AUC), prints a comparison table, saves a bar-chart comparison image,
and persists the best model (as a full preprocessing+model Pipeline) to
../models/best_model.pkl for use by the Flask app.

Run from inside src/:
    python train_models.py
"""

import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

from feature_engineering import build_dataset
from data_preprocessing import build_preprocessor, get_train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, use_label_encoder=False,
        )
    else:
        print("[WARN] xgboost not installed - using GradientBoostingClassifier as a stand-in.")
        models["XGBoost (fallback: GradientBoosting)"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, random_state=42
        )
    return models


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main():
    print("Loading & engineering dataset...")
    df = build_dataset(
        os.path.join(DATA_DIR, "application_record.csv"),
        os.path.join(DATA_DIR, "credit_record.csv"),
    )
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    results = {}
    fitted_pipelines = {}

    for name, model in get_models().items():
        print(f"\nTraining: {name}")
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        results[name] = metrics
        fitted_pipelines[name] = pipeline

        print(f"  {metrics}")
        print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    results_df = pd.DataFrame(results).T.sort_values("f1_score", ascending=False)
    print("\n=== Model Comparison (sorted by F1) ===")
    print(results_df)

    results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))

    # Bar chart comparison
    ax = results_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(
        kind="bar", figsize=(10, 6), rot=20
    )
    ax.set_title("Model Comparison - Credit Card Approval Prediction")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "model_comparison.png"), dpi=150)
    print(f"Saved comparison chart -> {os.path.join(MODELS_DIR, 'model_comparison.png')}")

    best_name = results_df.index[0]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, best_model_path)
    print(f"Saved best model pipeline -> {best_model_path}")

    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump({"best_model": best_name, "metrics": results[best_name]}, f, indent=2)

    # Also copy to flask_app/model for the web app to use directly
    flask_model_dir = os.path.join(os.path.dirname(__file__), "..", "flask_app", "model")
    os.makedirs(flask_model_dir, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(flask_model_dir, "best_model.pkl"))
    print(f"Copied best model -> {flask_model_dir}/best_model.pkl")


if __name__ == "__main__":
    main()
