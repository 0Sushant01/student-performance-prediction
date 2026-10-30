"""
database.py
===========
SQLite persistence layer for Teacher-Oriented Student Performance Prediction System.
Handles schema creation, parameterized CRUD operations, search, and dashboard aggregation.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "students.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a connection to the SQLite database with row_factory configured."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes tables using clean, normalized schemas with foreign key relationships."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Table 1: Student Information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                student_name TEXT NOT NULL,
                semester TEXT,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table 2: Academic & Behavioral Records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS academic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                daily_study_hours REAL,
                attendance_percentage REAL,
                average_sleep_hours REAL,
                classroom_participation TEXT,
                academic_activity_participation TEXT,
                ia1_marks REAL,
                ia2_marks REAL,
                ia3_marks REAL,
                average_ia_marks REAL,
                previous_semester_percentage REAL,
                previous_academic_failures INTEGER,
                total_assignments INTEGER,
                assignments_submitted INTEGER,
                on_time_submissions INTEGER,
                delayed_submissions INTEGER,
                missing_assignments INTEGER,
                submission_rate REAL,
                on_time_submission_rate REAL,
                final_exam_marks REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            )
        """)

        # Table 3: Prediction History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            )
        """)

        conn.commit()


def save_student_record(data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Inserts or updates a complete student record using parameterized queries.
    Prevents SQL injection and maintains referential integrity.
    """
    init_database(db_path)
    student_id = str(data["student_id"]).strip()
    student_name = str(data["student_name"]).strip()
    semester = str(data.get("semester", "")).strip()
    department = str(data.get("department", "")).strip()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Upsert student basic info
        cursor.execute("""
            INSERT INTO students (student_id, student_name, semester, department, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(student_id) DO UPDATE SET
                student_name = excluded.student_name,
                semester = excluded.semester,
                department = excluded.department,
                updated_at = CURRENT_TIMESTAMP
        """, (student_id, student_name, semester, department))

        # 2. Upsert academic record
        cursor.execute("""
            INSERT INTO academic_records (
                student_id, daily_study_hours, attendance_percentage, average_sleep_hours,
                classroom_participation, academic_activity_participation,
                ia1_marks, ia2_marks, ia3_marks, average_ia_marks,
                previous_semester_percentage, previous_academic_failures,
                total_assignments, assignments_submitted, on_time_submissions,
                delayed_submissions, missing_assignments, submission_rate, on_time_submission_rate,
                final_exam_marks
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                daily_study_hours = excluded.daily_study_hours,
                attendance_percentage = excluded.attendance_percentage,
                average_sleep_hours = excluded.average_sleep_hours,
                classroom_participation = excluded.classroom_participation,
                academic_activity_participation = excluded.academic_activity_participation,
                ia1_marks = excluded.ia1_marks,
                ia2_marks = excluded.ia2_marks,
                ia3_marks = excluded.ia3_marks,
                average_ia_marks = excluded.average_ia_marks,
                previous_semester_percentage = excluded.previous_semester_percentage,
                previous_academic_failures = excluded.previous_academic_failures,
                total_assignments = excluded.total_assignments,
                assignments_submitted = excluded.assignments_submitted,
                on_time_submissions = excluded.on_time_submissions,
                delayed_submissions = excluded.delayed_submissions,
                missing_assignments = excluded.missing_assignments,
                submission_rate = excluded.submission_rate,
                on_time_submission_rate = excluded.on_time_submission_rate,
                final_exam_marks = excluded.final_exam_marks,
                recorded_at = CURRENT_TIMESTAMP
        """, (
            student_id,
            float(data.get("daily_study_hours", 0)),
            float(data.get("attendance_percentage", 0)),
            float(data.get("average_sleep_hours", 7)),
            str(data.get("classroom_participation", "Average")),
            str(data.get("academic_activity_participation", "Occasional")),
            float(data.get("ia1_marks", 0)),
            float(data.get("ia2_marks", 0)),
            float(data.get("ia3_marks", 0)),
            float(data.get("average_ia_marks", 0)),
            float(data.get("previous_semester_percentage", 0)),
            int(data.get("previous_academic_failures", 0)),
            int(data.get("total_assignments", 0)),
            int(data.get("assignments_submitted", 0)),
            int(data.get("on_time_submissions", 0)),
            int(data.get("delayed_submissions", 0)),
            int(data.get("missing_assignments", 0)),
            float(data.get("submission_rate", 0)),
            float(data.get("on_time_submission_rate", 0)),
            float(data["final_exam_marks"]) if data.get("final_exam_marks") is not None else None,
        ))

        conn.commit()
    return True


def record_prediction(student_id: str, score: float, db_path: str = DEFAULT_DB_PATH) -> None:
    """Logs a generated prediction into prediction_history."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (student_id, predicted_score)
            VALUES (?, ?)
        """, (student_id, float(score)))
        conn.commit()


def get_all_students_df(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Returns a unified pandas DataFrame joining student details and academic records."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        query = """
            SELECT 
                s.student_id,
                s.student_name,
                s.semester,
                s.department,
                a.daily_study_hours,
                a.attendance_percentage,
                a.average_sleep_hours,
                a.classroom_participation,
                a.academic_activity_participation,
                a.ia1_marks,
                a.ia2_marks,
                a.ia3_marks,
                a.average_ia_marks,
                a.previous_semester_percentage,
                a.previous_academic_failures,
                a.total_assignments,
                a.assignments_submitted,
                a.on_time_submissions,
                a.delayed_submissions,
                a.missing_assignments,
                a.submission_rate,
                a.on_time_submission_rate,
                a.final_exam_marks,
                s.updated_at
            FROM students s
            LEFT JOIN academic_records a ON s.student_id = a.student_id
            ORDER BY s.updated_at DESC
        """
        df = pd.read_sql_query(query, conn)
    return df


def get_student_record(student_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetches a single student's record by student_id."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                s.student_id, s.student_name, s.semester, s.department,
                a.daily_study_hours, a.attendance_percentage, a.average_sleep_hours,
                a.classroom_participation, a.academic_activity_participation,
                a.ia1_marks, a.ia2_marks, a.ia3_marks, a.average_ia_marks,
                a.previous_semester_percentage, a.previous_academic_failures,
                a.total_assignments, a.assignments_submitted, a.on_time_submissions,
                a.delayed_submissions, a.missing_assignments, a.submission_rate,
                a.on_time_submission_rate, a.final_exam_marks
            FROM students s
            LEFT JOIN academic_records a ON s.student_id = a.student_id
            WHERE s.student_id = ?
        """, (student_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def search_students(query_str: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Searches students by student ID or Name."""
    init_database(db_path)
    pattern = f"%{query_str.strip()}%"
    with get_db_connection(db_path) as conn:
        query = """
            SELECT 
                s.student_id, s.student_name, s.semester, s.department,
                a.attendance_percentage, a.average_ia_marks, a.submission_rate,
                a.previous_semester_percentage, a.final_exam_marks
            FROM students s
            LEFT JOIN academic_records a ON s.student_id = a.student_id
            WHERE s.student_id LIKE ? OR s.student_name LIKE ?
            ORDER BY s.student_id ASC
        """
        df = pd.read_sql_query(query, conn, params=(pattern, pattern))
    return df


def delete_student_record(student_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes a student record and associated academic records."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM academic_records WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM prediction_history WHERE student_id = ?", (student_id,))
        conn.commit()
    return True


def get_dashboard_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Calculates overall metrics from saved records for the dashboard."""
    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]

        cursor.execute("""
            SELECT 
                COUNT(*),
                AVG(attendance_percentage),
                AVG(average_ia_marks),
                AVG(submission_rate)
            FROM academic_records
            WHERE attendance_percentage IS NOT NULL
        """)
        row = cursor.fetchone()
        records_with_data = row[0] if row else 0
        avg_att = row[1] if row and row[1] is not None else 0.0
        avg_ia = row[2] if row and row[2] is not None else 0.0
        avg_sub = row[3] if row and row[3] is not None else 0.0

    return {
        "total_students": total_students,
        "available_records": records_with_data,
        "average_attendance": round(avg_att, 2),
        "average_ia_marks": round(avg_ia, 2),
        "average_submission_rate": round(avg_sub, 2),
    }


def populate_sample_data(num_samples: int = 50, db_path: str = DEFAULT_DB_PATH) -> int:
    """Pre-populates the database with synthetic demonstration records if empty."""
    from preprocessing import generate_synthetic_student_data

    init_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        if cursor.fetchone()[0] > 0:
            return 0  # Already has records

    sample_df = generate_synthetic_student_data(num_samples=num_samples, random_state=42)
    count = 0
    for _, row in sample_df.iterrows():
        save_student_record(row.to_dict(), db_path=db_path)
        count += 1
    return count
