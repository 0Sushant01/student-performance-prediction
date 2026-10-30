"""
train_model.py
==============
Model training and evaluation routine for the Teacher-Oriented Student Performance Prediction System.

Workflow:
1. Loads labeled historical student records (or generates sample synthetic data if absent).
2. Validates data integrity, derived columns, and data types.
3. Separates predictive features (study habits, IA marks, attendance, submissions) from target.
4. Performs an 80:20 train-test split (random_state=42).
5. Fits a Scikit-Learn Pipeline (ColumnTransformer with StandardScaler + OneHotEncoder + LinearRegression).
6. Computes empirical test metrics: MAE, RMSE, and R² Score.
7. Generates visual evaluation plots (Actual vs Predicted).
8. Persists trained model and metadata using Joblib.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from preprocessing import (
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    build_model_pipeline,
    generate_synthetic_student_data,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

REAL_DATA_CSV = os.path.join(DATA_DIR, "student_records.csv")
SYNTHETIC_DATA_CSV = os.path.join(DATA_DIR, "sample_synthetic_data.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "student_performance_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
PLOT_PATH = os.path.join(MODELS_DIR, "actual_vs_predicted.png")


def load_training_dataset(data_path: str = None) -> tuple[pd.DataFrame, bool]:
    """
    Loads historical dataset for model training.
    If real data exists in data/student_records.csv, it is prioritized.
    Otherwise, generates and loads data/sample_synthetic_data.csv.
    Returns: (DataFrame, is_synthetic_boolean)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        is_synthetic = "synthetic" in os.path.basename(data_path).lower()
        return df, is_synthetic

    if os.path.exists(REAL_DATA_CSV):
        print(f"[Dataset] Loading real historical records from: {REAL_DATA_CSV}")
        df = pd.read_csv(REAL_DATA_CSV)
        return df, False

    if not os.path.exists(SYNTHETIC_DATA_CSV):
        print(f"[Dataset] Generating baseline synthetic demonstration data (300 records)...")
        synth_df = generate_synthetic_student_data(num_samples=300, random_state=42)
        synth_df.to_csv(SYNTHETIC_DATA_CSV, index=False)
        print(f"[Dataset] Saved sample synthetic dataset to: {SYNTHETIC_DATA_CSV}")

    print(f"[Dataset] Loading sample synthetic demonstration data from: {SYNTHETIC_DATA_CSV}")
    print("           NOTE: Synthetic data is used for interface demonstration and pipeline validation.")
    df = pd.read_csv(SYNTHETIC_DATA_CSV)
    return df, True


def train_model(custom_data_path: str = None):
    """Executes the full training and evaluation workflow."""
    print("=" * 75)
    print(" TEACHER-ORIENTED STUDENT PERFORMANCE PREDICTION SYSTEM - TRAINING")
    print("=" * 75)

    df, is_synthetic = load_training_dataset(custom_data_path)

    # 1. Validation of required columns
    required_cols = MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Drop any records with missing target
    initial_len = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    if len(df) < 20:
        raise ValueError(
            f"Insufficient valid labeled records for training (found {len(df)}). "
            "At least 20 labeled student records are required."
        )

    # 2. Extract features and target
    X = df[MODEL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"\n[Dataset Summary]")
    print(f" Total records loaded: {len(df)}")
    print(f" Data category:        {'Synthetic Demonstration Data' if is_synthetic else 'Real Historical Records'}")
    print(f" Input Features ({len(MODEL_FEATURE_COLUMNS)}): {MODEL_FEATURE_COLUMNS}")
    print(f" Target Variable:      {TARGET_COLUMN} (0 - 100% scale)")

    # 3. Train-test split (80:20, reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[Data Partitioning]")
    print(f" Training samples: {len(X_train)} (80%)")
    print(f" Testing samples:  {len(X_test)} (20%)")

    # 4. Construct & Train Scikit-Learn Pipeline
    pipeline = build_model_pipeline()
    print("\n[Model Fitting]")
    print(" Training Scikit-Learn Pipeline (StandardScaler + OneHotEncoder + LinearRegression)...")
    pipeline.fit(X_train, y_train)
    print(" Training completed successfully.")

    # 5. Evaluate on Test Set
    y_pred = pipeline.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0.0, 100.0)

    mae = float(mean_absolute_error(y_test, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
    r2 = float(r2_score(y_test, y_pred_clipped))

    regressor = pipeline.named_steps["regressor"]
    intercept = float(regressor.intercept_)

    print("\n" + "-" * 75)
    print(" EMPIRICAL MODEL EVALUATION METRICS (Held-Out Test Set)")
    print("-" * 75)
    print(f" 1. Mean Absolute Error (MAE):     {mae:.4f} %")
    print(f"    -> Average absolute difference between predicted and actual final marks.")
    print(f" 2. Root Mean Squared Error (RMSE): {rmse:.4f} %")
    print(f"    -> Square root of mean squared error, sensitive to large individual errors.")
    print(f" 3. R-squared Score (R²):           {r2:.4f}")
    print(f"    -> {r2 * 100:.2f}% of final examination variance is explained by the model.")
    if is_synthetic:
        print("    -> PRELIMINARY TEST: Results reflect synthetic relations for system testing.")
    print("-" * 75)

    # 6. Save Model Artifact and Metadata
    os.makedirs(MODELS_DIR, exist_ok=True)

    metadata = {
        "model_name": "Teacher Academic Performance Linear Regression",
        "algorithm": "sklearn.linear_model.LinearRegression",
        "data_source": "sample_synthetic_data.csv" if is_synthetic else "student_records.csv",
        "is_synthetic": is_synthetic,
        "feature_names": MODEL_FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_name": TARGET_COLUMN,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": 42,
        "test_split_ratio": 0.20,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        },
        "intercept": round(intercept, 4),
        "target_scale": {"min": 0.0, "max": 100.0},
    }

    model_artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
    }

    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\n[Persistence] Model saved to: {MODEL_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Persistence] Metrics metadata saved to: {METRICS_PATH}")

    # 7. Generate Evaluation Plot
    generate_evaluation_plot(y_test, y_pred_clipped, is_synthetic)

    print("\n" + "=" * 75)
    print(" MODEL TRAINING COMPLETE & READY FOR STREAMLIT!")
    print("=" * 75)
    return model_artifact


def generate_evaluation_plot(y_test: pd.Series, y_pred: np.ndarray, is_synthetic: bool):
    """Creates Actual vs Predicted examination score scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.scatter(y_test, y_pred, color="#2563eb", alpha=0.75, edgecolors="k", s=50, label="Test Predictions")
    ax.plot([0, 100], [0, 100], color="#dc2626", linestyle="--", linewidth=2, label="Perfect Line (y = x)")

    subtitle = "(Synthetic Test Data)" if is_synthetic else "(Historical Records)"
    ax.set_title(f"Actual vs. Predicted Final Exam Score {subtitle}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Final Marks (%)", fontsize=11)
    ax.set_ylabel("Predicted Final Marks (%)", fontsize=11)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"[Visualization] Saved Actual vs. Predicted plot to: {PLOT_PATH}")


def predict_final_marks(pipeline, record_dict: dict) -> float:
    """Predicts final examination score from input record dictionary."""
    df_input = pd.DataFrame([record_dict])[MODEL_FEATURE_COLUMNS]
    pred = float(pipeline.predict(df_input)[0])
    return float(np.clip(round(pred, 2), 0.0, 100.0))


if __name__ == "__main__":
    custom_path = sys.argv[1] if len(sys.argv) > 1 else None
    train_model(custom_path)
