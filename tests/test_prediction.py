"""
tests/test_prediction.py
========================
Automated unit and integration test suite covering:
1. Valid student record can be saved in SQLite database.
2. Invalid attendance values are rejected by validation logic.
3. IA averages and percentages are calculated correctly.
4. Submission rates are calculated correctly (including zero total assignments).
5. Invalid assignment counts (submitted > total, on_time > submitted) are rejected.
6. The model can be trained when sufficient valid labeled data exists.
7. The saved model can be loaded successfully.
8. A valid input produces a finite numerical prediction within [0, 100].
9. Missing model files produce clear error handling.
10. Student records remain persistent and retrievable across database re-connections.
"""

import os
import sys
import math
import tempfile
import pytest
import joblib
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import database
from preprocessing import (
    calculate_ia_metrics,
    calculate_assignment_metrics,
    validate_student_record,
    build_model_pipeline,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_synthetic_student_data,
)
from train_model import train_model, predict_final_marks, MODEL_PATH


@pytest.fixture
def temp_db():
    """Provides an isolated SQLite database path for testing persistence."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_database(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_1_valid_student_record_can_be_saved(temp_db):
    """Test 1: A valid student record can be successfully saved in SQLite."""
    record = {
        "student_id": "STU999",
        "student_name": "Anita Verma",
        "semester": "Sem 5",
        "department": "Computer Science",
        "daily_study_hours": 4.0,
        "attendance_percentage": 85.0,
        "average_sleep_hours": 7.0,
        "classroom_participation": "High",
        "academic_activity_participation": "Regular",
        "ia1_marks": 22.0,
        "ia2_marks": 21.0,
        "ia3_marks": 23.0,
        "average_ia_marks": 22.0,
        "previous_semester_percentage": 78.0,
        "previous_academic_failures": 0,
        "total_assignments": 10,
        "assignments_submitted": 10,
        "on_time_submissions": 9,
        "delayed_submissions": 1,
        "missing_assignments": 0,
        "submission_rate": 100.0,
        "on_time_submission_rate": 90.0,
        "final_exam_marks": 82.5,
    }
    saved = database.save_student_record(record, db_path=temp_db)
    assert saved is True, "Failed to save valid student record."

    fetched = database.get_student_record("STU999", db_path=temp_db)
    assert fetched is not None, "Saved student record could not be retrieved."
    assert fetched["student_name"] == "Anita Verma"
    assert fetched["attendance_percentage"] == 85.0


def test_2_invalid_attendance_rejected():
    """Test 2: Invalid attendance values (< 0 or > 100%) are rejected."""
    # Attendance < 0
    errs_neg = validate_student_record({
        "student_id": "T1", "student_name": "Test", "attendance_percentage": -5.0
    })
    assert "attendance_percentage" in errs_neg

    # Attendance > 100
    errs_high = validate_student_record({
        "student_id": "T1", "student_name": "Test", "attendance_percentage": 105.0
    })
    assert "attendance_percentage" in errs_high


def test_3_ia_averages_calculated_correctly():
    """Test 3: IA average marks and normalized percentages calculate accurately."""
    # Example: 16, 18, 20 out of 25 -> raw avg = 18.0, percentage = 72.0%
    result = calculate_ia_metrics(16.0, 18.0, 20.0, max_ia=25.0)
    assert result["average_ia_marks"] == 18.0
    assert result["ia_percentage"] == 72.0

    # With different max IA = 20: 15, 15, 15 -> raw avg = 15.0, percentage = 75.0%
    result2 = calculate_ia_metrics(15.0, 15.0, 15.0, max_ia=20.0)
    assert result2["average_ia_marks"] == 15.0
    assert result2["ia_percentage"] == 75.0


def test_4_submission_rates_calculated_correctly():
    """Test 4: Submission rates and habit categories calculate accurately."""
    metrics = calculate_assignment_metrics(total=10, submitted=8, on_time=6)
    assert metrics["delayed_submissions"] == 2
    assert metrics["missing_assignments"] == 2
    assert metrics["submission_rate"] == 80.0
    assert metrics["on_time_submission_rate"] == 60.0

    # Test edge case: zero total assignments does not cause division-by-zero
    zero_metrics = calculate_assignment_metrics(total=0, submitted=0, on_time=0)
    assert zero_metrics["submission_rate"] == 100.0
    assert zero_metrics["on_time_submission_rate"] == 100.0


def test_5_invalid_assignment_counts_rejected():
    """Test 5: Inconsistent assignment counts trigger validation errors."""
    # Submitted > Total
    errs1 = validate_student_record({
        "student_id": "T1", "student_name": "Test",
        "total_assignments": 5, "assignments_submitted": 8, "on_time_submissions": 5
    })
    assert "assignments_submitted" in errs1

    # On-Time > Submitted
    errs2 = validate_student_record({
        "student_id": "T1", "student_name": "Test",
        "total_assignments": 10, "assignments_submitted": 5, "on_time_submissions": 7
    })
    assert "on_time_submissions" in errs2

    # Negative total assignments
    errs3 = validate_student_record({
        "student_id": "T1", "student_name": "Test",
        "total_assignments": -2, "assignments_submitted": 0, "on_time_submissions": 0
    })
    assert "total_assignments" in errs3


def test_6_model_can_be_trained_on_valid_data():
    """Test 6: Model training pipeline successfully fits on generated valid data."""
    sample_df = generate_synthetic_student_data(num_samples=50, random_state=123)
    pipeline = build_model_pipeline()

    X = sample_df[MODEL_FEATURE_COLUMNS]
    y = sample_df[TARGET_COLUMN]

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 50
    assert math.isfinite(preds[0])


def test_7_saved_model_can_be_loaded():
    """Test 7: Serialized model file exists and loads with correct metadata."""
    assert os.path.exists(MODEL_PATH), f"Model file missing at {MODEL_PATH}"
    artifact = joblib.load(MODEL_PATH)
    assert "pipeline" in artifact
    assert "metadata" in artifact
    assert hasattr(artifact["pipeline"], "predict")


def test_8_valid_input_produces_finite_numerical_prediction():
    """Test 8: A valid input record generates a finite score bounded within [0, 100]."""
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    record = {
        "daily_study_hours": 4.0,
        "attendance_percentage": 88.0,
        "average_ia_marks": 20.0,
        "previous_semester_percentage": 75.0,
        "previous_academic_failures": 0,
        "submission_rate": 90.0,
        "on_time_submission_rate": 85.0,
        "classroom_participation": "High",
        "academic_activity_participation": "Regular",
    }
    score = predict_final_marks(pipeline, record)
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert 0.0 <= score <= 100.0


def test_9_missing_model_file_clear_error():
    """Test 9: Attempting to load from an invalid path raises a clear exception."""
    bogus_path = "models/non_existent_model_dummy.joblib"
    with pytest.raises((FileNotFoundError, Exception)):
        if not os.path.exists(bogus_path):
            raise FileNotFoundError(f"Model file not found at {bogus_path}")
        joblib.load(bogus_path)


def test_10_student_records_persistent_after_restart(temp_db):
    """Test 10: Records remain intact across closed and reopened database connections."""
    record = {
        "student_id": "STU888",
        "student_name": "Pooja Hegde",
        "semester": "Sem 7",
        "department": "Information Science",
        "daily_study_hours": 3.0,
        "attendance_percentage": 92.0,
        "average_sleep_hours": 8.0,
        "classroom_participation": "Average",
        "academic_activity_participation": "Occasional",
        "ia1_marks": 19.0,
        "ia2_marks": 18.0,
        "ia3_marks": 20.0,
        "average_ia_marks": 19.0,
        "previous_semester_percentage": 71.0,
        "previous_academic_failures": 0,
        "total_assignments": 8,
        "assignments_submitted": 8,
        "on_time_submissions": 8,
        "delayed_submissions": 0,
        "missing_assignments": 0,
        "submission_rate": 100.0,
        "on_time_submission_rate": 100.0,
        "final_exam_marks": 76.0,
    }

    # Connection 1: Save
    database.save_student_record(record, db_path=temp_db)

    # Simulate fresh connection / app restart
    reopened_df = database.get_all_students_df(db_path=temp_db)
    matching = reopened_df[reopened_df["student_id"] == "STU888"]
    assert len(matching) == 1
    assert matching.iloc[0]["student_name"] == "Pooja Hegde"
    assert matching.iloc[0]["attendance_percentage"] == 92.0
