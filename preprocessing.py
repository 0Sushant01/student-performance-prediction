"""
preprocessing.py
================
Feature definitions, validation routines, calculations, and Scikit-Learn
pipeline construction for the Teacher-Oriented Student Performance Prediction System.
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Predictive features
NUMERICAL_FEATURES = [
    "daily_study_hours",
    "attendance_percentage",
    "average_ia_marks",
    "previous_semester_percentage",
    "previous_academic_failures",
    "submission_rate",
    "on_time_submission_rate",
]

CATEGORICAL_FEATURES = [
    "classroom_participation",
    "academic_activity_participation",
]

MODEL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "final_exam_marks"

ALL_DATASET_COLUMNS = [
    "student_id",
    "student_name",
    "semester",
    "daily_study_hours",
    "attendance_percentage",
    "average_sleep_hours",
    "classroom_participation",
    "academic_activity_participation",
    "ia1_marks",
    "ia2_marks",
    "ia3_marks",
    "average_ia_marks",
    "previous_semester_percentage",
    "previous_academic_failures",
    "total_assignments",
    "assignments_submitted",
    "on_time_submissions",
    "delayed_submissions",
    "missing_assignments",
    "submission_rate",
    "on_time_submission_rate",
    "final_exam_marks",
]


def calculate_ia_metrics(
    ia1: float, ia2: float, ia3: float, max_ia: float = 25.0
) -> Dict[str, float]:
    """
    Calculates average raw IA marks and normalized IA percentage.
    Example: 16, 18, 20 out of 25 -> raw avg = 18.0, percentage = 72.0%
    """
    if max_ia <= 0:
        raise ValueError("Maximum IA marks must be greater than zero.")

    raw_avg = (ia1 + ia2 + ia3) / 3.0
    ia_percentage = (raw_avg / max_ia) * 100.0
    return {
        "average_ia_marks": round(raw_avg, 2),
        "ia_percentage": round(ia_percentage, 2),
    }


def calculate_assignment_metrics(
    total: int, submitted: int, on_time: int
) -> Dict[str, Any]:
    """
    Calculates delayed submissions, missing assignments, submission rate,
    and on-time submission rate from raw assignment counts.
    Handles zero total assignments safely without division-by-zero.
    """
    total = int(total)
    submitted = int(submitted)
    on_time = int(on_time)

    delayed = max(0, submitted - on_time)
    missing = max(0, total - submitted)

    if total > 0:
        submission_rate = (submitted / total) * 100.0
        on_time_rate = (on_time / total) * 100.0
    else:
        submission_rate = 100.0
        on_time_rate = 100.0

    # Derive submission habit category
    if submission_rate < 50.0 or missing > (total * 0.4):
        habit = "Frequently missing"
    elif on_time_rate < 60.0 or delayed > (total * 0.4):
        habit = "Frequently delayed"
    elif on_time_rate >= 90.0:
        habit = "Consistently on time"
    else:
        habit = "Usually on time"

    return {
        "delayed_submissions": delayed,
        "missing_assignments": missing,
        "submission_rate": round(submission_rate, 2),
        "on_time_submission_rate": round(on_time_rate, 2),
        "submission_habit": habit,
    }


def validate_student_record(data: Dict[str, Any], max_ia: float = 25.0) -> Dict[str, str]:
    """
    Validates complete teacher record input according to academic bounds.
    Returns a dictionary of field -> error message (empty if all valid).
    """
    errors: Dict[str, str] = {}

    # Section A: Student Information
    student_id = str(data.get("student_id", "")).strip()
    student_name = str(data.get("student_name", "")).strip()
    if not student_id:
        errors["student_id"] = "Student ID cannot be empty."
    if not student_name:
        errors["student_name"] = "Student Name cannot be empty."

    # Section B: Study Habits & Attendance
    try:
        study_hours = float(data.get("daily_study_hours", 0))
        if not (0.0 <= study_hours <= 24.0):
            errors["daily_study_hours"] = "Daily study hours must be between 0 and 24."
    except (ValueError, TypeError):
        errors["daily_study_hours"] = "Daily study hours must be a valid number."

    try:
        att = float(data.get("attendance_percentage", 0))
        if not (0.0 <= att <= 100.0):
            errors["attendance_percentage"] = "Attendance percentage must be between 0 and 100%."
    except (ValueError, TypeError):
        errors["attendance_percentage"] = "Attendance must be a valid percentage."

    try:
        sleep = float(data.get("average_sleep_hours", 7))
        if not (0.0 <= sleep <= 24.0):
            errors["average_sleep_hours"] = "Sleep hours must be between 0 and 24."
    except (ValueError, TypeError):
        errors["average_sleep_hours"] = "Sleep hours must be a valid number."

    # Dropdowns
    if data.get("classroom_participation") not in ["Low", "Average", "High"]:
        errors["classroom_participation"] = "Classroom participation must be Low, Average, or High."
    if data.get("academic_activity_participation") not in ["None", "Occasional", "Regular"]:
        errors["academic_activity_participation"] = "Academic activity participation must be None, Occasional, or Regular."

    # Section C: Academic Performance
    for ia_field in ["ia1_marks", "ia2_marks", "ia3_marks"]:
        try:
            val = float(data.get(ia_field, 0))
            if val < 0.0:
                errors[ia_field] = f"{ia_field.upper()} marks cannot be negative."
            elif val > max_ia:
                errors[ia_field] = f"{ia_field.upper()} marks cannot exceed maximum IA marks ({max_ia})."
        except (ValueError, TypeError):
            errors[ia_field] = f"{ia_field.upper()} must be a valid number."

    try:
        prev_pct = float(data.get("previous_semester_percentage", 0))
        if not (0.0 <= prev_pct <= 100.0):
            errors["previous_semester_percentage"] = "Previous semester percentage must be between 0 and 100%."
    except (ValueError, TypeError):
        errors["previous_semester_percentage"] = "Previous semester percentage must be a valid number."

    try:
        failures = int(data.get("previous_academic_failures", 0))
        if failures < 0:
            errors["previous_academic_failures"] = "Previous failures cannot be negative."
    except (ValueError, TypeError):
        errors["previous_academic_failures"] = "Previous failures must be a valid integer."

    # Section D: Assignment Habits
    try:
        total = int(data.get("total_assignments", 0))
        submitted = int(data.get("assignments_submitted", 0))
        on_time = int(data.get("on_time_submissions", 0))

        if total < 0:
            errors["total_assignments"] = "Total assignments cannot be negative."
        if submitted < 0:
            errors["assignments_submitted"] = "Submitted assignments cannot be negative."
        if on_time < 0:
            errors["on_time_submissions"] = "On-time submissions cannot be negative."

        if total >= 0 and submitted > total:
            errors["assignments_submitted"] = "Submitted assignments cannot exceed total assignments."
        if submitted >= 0 and on_time > submitted:
            errors["on_time_submissions"] = "On-time submissions cannot exceed submitted assignments."
    except (ValueError, TypeError):
        errors["total_assignments"] = "Assignment counts must be valid integers."

    return errors


def build_model_pipeline() -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline combining:
    1. Numerical feature standard scaling (with median imputation for robustness).
    2. Categorical feature one-hot encoding (with handle_unknown='ignore').
    3. LinearRegression estimator.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    return full_pipeline


def generate_synthetic_student_data(
    num_samples: int = 300, random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, educational synthetic dataset matching the exact schema.
    Clearly designated as synthetic demonstration data for development, pipeline testing,
    and initial classroom demonstration when real historical records are not yet compiled.
    """
    rng = np.random.RandomState(random_state)

    records = []
    departments = ["Computer Science", "Information Science", "Electronics", "Mechanical"]
    semesters = ["Sem 3", "Sem 4", "Sem 5", "Sem 6", "Sem 7"]
    participation_levels = ["Low", "Average", "High"]
    activity_levels = ["None", "Occasional", "Regular"]

    for i in range(1, num_samples + 1):
        s_id = f"STU{1000 + i}"
        s_name = f"Student {i}"
        dept = rng.choice(departments)
        sem = rng.choice(semesters)

        # Realistic academic behaviors
        daily_study = float(np.clip(rng.normal(3.5, 1.8), 0.5, 12.0))
        attendance = float(np.clip(rng.normal(78.0, 14.0), 30.0, 100.0))
        sleep_hours = float(np.clip(rng.normal(7.0, 1.2), 4.0, 10.0))

        # Categorical probabilities
        c_part = rng.choice(participation_levels, p=[0.25, 0.50, 0.25])
        a_part = rng.choice(activity_levels, p=[0.30, 0.45, 0.25])

        # Previous performance
        prev_pct = float(np.clip(rng.normal(68.0, 12.0), 35.0, 98.0))
        # Failure count usually 0, occasionally 1-3
        p_fail = rng.choice([0, 1, 2, 3], p=[0.75, 0.15, 0.07, 0.03])

        # IA marks out of 25 correlated with previous percentage and attendance
        ia_base = (prev_pct / 100.0) * 22.0 + (attendance / 100.0) * 3.0
        ia1 = float(np.clip(rng.normal(ia_base, 2.5), 0.0, 25.0))
        ia2 = float(np.clip(rng.normal(ia_base + 0.5, 2.2), 0.0, 25.0))
        ia3 = float(np.clip(rng.normal(ia_base + 1.0, 2.0), 0.0, 25.0))
        avg_ia = round((ia1 + ia2 + ia3) / 3.0, 2)

        # Assignments
        total_assign = rng.randint(6, 12)
        submit_ratio = np.clip((attendance / 100.0) * 0.6 + (prev_pct / 100.0) * 0.4 + rng.normal(0, 0.08), 0.3, 1.0)
        submitted = int(np.clip(round(total_assign * submit_ratio), 1, total_assign))
        on_time_ratio = np.clip(rng.uniform(0.6, 1.0) if submitted > 0 else 0.0, 0.2, 1.0)
        on_time = int(np.clip(round(submitted * on_time_ratio), 0, submitted))

        delayed = submitted - on_time
        missing = total_assign - submitted
        sub_rate = round((submitted / total_assign) * 100.0, 2)
        on_time_rate = round((on_time / total_assign) * 100.0, 2)

        # Target final exam marks (0-100) with realistic weights + residual noise
        part_bonus = 3.0 if c_part == "High" else (1.5 if c_part == "Average" else 0.0)
        act_bonus = 2.0 if a_part == "Regular" else (1.0 if a_part == "Occasional" else 0.0)

        exam_score = (
            0.35 * prev_pct
            + 0.30 * (avg_ia / 25.0 * 100.0)
            + 0.15 * attendance
            + 0.08 * sub_rate
            + 0.05 * on_time_rate
            + 0.8 * daily_study
            - 2.5 * p_fail
            + part_bonus
            + act_bonus
            + rng.normal(0, 3.5)
        )
        final_marks = float(np.clip(round(exam_score, 1), 0.0, 100.0))

        records.append({
            "student_id": s_id,
            "student_name": s_name,
            "semester": sem,
            "department": dept,
            "daily_study_hours": round(daily_study, 1),
            "attendance_percentage": round(attendance, 1),
            "average_sleep_hours": round(sleep_hours, 1),
            "classroom_participation": c_part,
            "academic_activity_participation": a_part,
            "ia1_marks": round(ia1, 1),
            "ia2_marks": round(ia2, 1),
            "ia3_marks": round(ia3, 1),
            "average_ia_marks": avg_ia,
            "previous_semester_percentage": round(prev_pct, 1),
            "previous_academic_failures": p_fail,
            "total_assignments": total_assign,
            "assignments_submitted": submitted,
            "on_time_submissions": on_time,
            "delayed_submissions": delayed,
            "missing_assignments": missing,
            "submission_rate": sub_rate,
            "on_time_submission_rate": on_time_rate,
            "final_exam_marks": final_marks,
        })

    return pd.DataFrame(records)
