<p align="center">
  <img src="assets/logo.png" alt="Student Performance Prediction System Logo" width="160"/>
</p>

# 🎓 Teacher-Oriented Student Performance Prediction System Using Machine Learning

A beginner-friendly, interpretable machine learning web application built with **Python**, **Scikit-learn**, **Streamlit**, and **SQLite**. It enables teachers to record student academic details, track progress, manage records locally, and predict expected final examination scores ($0–100\%$).

Designed for individual undergraduate engineering project submissions, laboratory vivas, and classroom demonstrations.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture & Directory Layout](#-system-architecture--directory-layout)
4. [Teacher Input Form Structure](#-teacher-input-form-structure)
5. [Predictive Features vs Identifiers](#-predictive-features-vs-identifiers)
6. [Machine Learning Methodology](#-machine-learning-methodology)
7. [Database Design (SQLite)](#-database-design-sqlite)
8. [Installation Instructions](#-installation-instructions)
9. [Execution Instructions](#-execution-instructions)
10. [Automated Unit Testing](#-automated-unit-testing)
11. [Empirical Evaluation Metrics](#-empirical-evaluation-metrics)
12. [Curated Viva Questions & Answers](#-curated-viva-questions--answers)
13. [Limitations & Ethical Guidelines](#-limitations--ethical-guidelines)
14. [Submission Checklist](#-submission-checklist)

---

## 📖 Project Overview
Predicting academic outcomes allows teachers to initiate early remedial support before final examinations occur. This application provides a unified academic record management system and a mathematical prediction engine based on **Linear Regression**.

It captures:
- Daily study and sleep duration.
- Attendance percentage.
- Continuous Internal Assessment (IA) marks.
- Assignment timeliness and submission discipline.
- Classroom and academic activity participation.
- Historical semester performance and backlogs.

---

## ✨ Key Features
- **4-Section Teacher Form:** Clean entry for student info, study habits, IA marks, and assignment completion.
- **Automated Calculations:** Calculates IA averages, normalized percentages, delayed submissions, missing assignments, and submission rates automatically.
- **Persistent Local Database:** Stores records in a serverless SQLite database (`data/students.db`) using parameterized SQL queries.
- **Configurable Assessment Schemes:** Supports custom maximum IA marks (e.g. 20, 25, or 50).
- **Leakage-Free ML Pipeline:** Scikit-Learn `ColumnTransformer` combines `StandardScaler` for numbers and `OneHotEncoder` for categories.
- **Data Import & Export:** Export student records to CSV or upload historical CSV records.
- **10 Automated Unit Tests:** 100% passing tests covering calculations, validations, model inference, and database persistence.

---

## 📁 System Architecture & Directory Layout

```
student-performance-prediction/
│
├── app.py                     # Streamlit multi-page web application
├── train_model.py             # Model training, split, evaluation, and serialization
├── database.py                # SQLite schema, CRUD operations, and dashboard analytics
├── preprocessing.py           # Calculations, input validations, and Scikit-learn pipeline
├── requirements.txt           # Version-pinned Python dependencies
├── README.md                  # Comprehensive documentation and viva guide
├── .gitignore                 # Exclusions (venv, caches, models, database)
│
├── data/
│   ├── students.db            # SQLite local database storing student records
│   ├── sample_synthetic_data.csv # Labeled demonstration data (300 records)
│   └── student_records.csv    # Real historical records (if available)
│
├── models/
│   ├── student_performance_model.joblib # Serialized Scikit-learn pipeline
│   ├── metrics.json           # Evaluation metrics & metadata
│   └── actual_vs_predicted.png# Test set evaluation scatter plot
│
├── reports/
│   └── project_report.md      # Formal undergraduate academic project report
│
└── tests/
    └── test_prediction.py     # 10 automated unit and integration tests
```

---

## 📝 Teacher Input Form Structure

The form titled **"Student Academic Performance Record"** is structured into four sections:

### Section A: Student Information
- **Student ID:** Unique student identifier (e.g., `STU1025`).
- **Student Name:** Full name of the student.
- **Semester:** Academic semester (`Sem 1` to `Sem 8`).
- **Department / Class:** Branch of study (e.g., Computer Science).
> *Note: Section A fields are administrative identifiers and are strictly excluded from predictive model features.*

### Section B: Study Habits and Attendance
- **Daily Study Hours:** Daily self-study duration ($0–24$ hours).
- **Average Sleep Hours:** Daily sleep duration ($0–24$ hours).
- **Attendance Percentage:** Recorded lecture attendance ($0–100\%$).
- **Classroom Participation:** Dropdown (`Low`, `Average`, `High`).
- **Academic Activity Participation:** Dropdown (`None`, `Occasional`, `Regular`).
> *Note: Study and sleep hours represent mentor-collected or self-reported estimates, not automated sensor readings.*

### Section C: Academic Performance
- **Configurable Maximum IA Marks:** Default $25$ marks.
- **IA 1, IA 2, IA 3 Marks:** Continuous internal evaluation marks ($0 \le \text{marks} \le \text{max\_ia}$).
- **Auto-Calculated Average:** Raw average marks and normalized percentage ($\frac{\text{sum}}{3 \times \text{max}} \times 100$).
- **Previous Semester Percentage:** Prior cumulative marks ($0–100\%$).
- **Previous Academic Failures:** Prior subject backlogs ($0$ or greater).

### Section D: Assignment & Submission Habits
- **Total Assignments Assigned:** Total count ($n \ge 0$).
- **Assignments Submitted:** Total submitted ($0 \le \text{submitted} \le \text{total}$).
- **On-Time Submissions:** Submissions prior to deadline ($0 \le \text{on\_time} \le \text{submitted}$).
- **Auto-Calculated Metrics:**
  - $\text{Delayed Submissions} = \max(0, \text{submitted} - \text{on\_time})$
  - $\text{Missing Assignments} = \max(0, \text{total} - \text{submitted})$
  - $\text{Submission Rate} = \frac{\text{submitted}}{\text{total}} \times 100$ (handles total = 0 safely)
  - $\text{On-Time Submission Rate} = \frac{\text{on\_time}}{\text{total}} \times 100$
- **Overall Submission Habit:** Dropdown derived from data (`Consistently on time`, `Usually on time`, `Frequently delayed`, `Frequently missing`).

---

## 🔍 Predictive Features vs Identifiers

| Feature | Role in System | Type |
| :--- | :--- | :--- |
| `student_id`, `student_name` | Administrative identification | Excluded from ML |
| `semester`, `department` | Administrative classification | Excluded from ML |
| `daily_study_hours` | Predictive Input | Numerical (StandardScaler) |
| `attendance_percentage` | Predictive Input | Numerical (StandardScaler) |
| `average_ia_marks` | Predictive Input | Numerical (StandardScaler) |
| `previous_semester_percentage` | Predictive Input | Numerical (StandardScaler) |
| `previous_academic_failures` | Predictive Input | Numerical (StandardScaler) |
| `submission_rate` | Predictive Input | Numerical (StandardScaler) |
| `on_time_submission_rate` | Predictive Input | Numerical (StandardScaler) |
| `classroom_participation` | Predictive Input | Categorical (OneHotEncoder) |
| `academic_activity_participation` | Predictive Input | Categorical (OneHotEncoder) |
| **`final_exam_marks`** | **Prediction Target ($y$)** | **Continuous Float (0–100%)** |

---

## 🧠 Machine Learning Methodology
- **Estimator:** Linear Regression (`sklearn.linear_model.LinearRegression`).
- **Transformers:**
  - Numerical Pipeline: `SimpleImputer(strategy='median')` + `StandardScaler()`.
  - Categorical Pipeline: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`.
  - Joined via `ColumnTransformer`.
- **Data Splitting:** 80% Training ($N=240$), 20% Testing ($N=60$), `random_state=42`.
- **Target Variable:** `final_exam_marks` (0 to 100 scale).

---

## 💻 Installation Instructions

### Step 1: Set up Virtual Environment
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Instructions

### Step 1: Train the Machine Learning Model
```bash
python train_model.py
```
*Automatically generates `data/sample_synthetic_data.csv` if no CSV exists, fits the pipeline, calculates test metrics, and exports `models/student_performance_model.joblib`.*

### Step 2: Run Automated Unit Tests
```bash
pytest -v tests/test_prediction.py
```

### Step 3: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Access the interface at `http://localhost:8501`.*

---

## 🧪 Automated Unit Testing
The test suite in [tests/test_prediction.py](file:///home/s/projects/college/student-performance-prediction/tests/test_prediction.py) runs 10 specific test scenarios:
```text
tests/test_prediction.py::test_1_valid_student_record_can_be_saved PASSED         [ 10%]
tests/test_prediction.py::test_2_invalid_attendance_rejected PASSED             [ 20%]
tests/test_prediction.py::test_3_ia_averages_calculated_correctly PASSED        [ 30%]
tests/test_prediction.py::test_4_submission_rates_calculated_correctly PASSED   [ 40%]
tests/test_prediction.py::test_5_invalid_assignment_counts_rejected PASSED     [ 50%]
tests/test_prediction.py::test_6_model_can_be_trained_on_valid_data PASSED     [ 60%]
tests/test_prediction.py::test_7_saved_model_can_be_loaded PASSED               [ 70%]
tests/test_prediction.py::test_8_valid_input_produces_finite_numerical_prediction PASSED [ 80%]
tests/test_prediction.py::test_9_missing_model_file_clear_error PASSED          [ 90%]
tests/test_prediction.py::test_10_student_records_persistent_after_restart PASSED [100%]

============================== 10 passed in 0.98s ==============================
```

---

## 📊 Empirical Evaluation Metrics

Evaluated on the held-out test partition ($N=60$):

| Metric | Measured Score | Practical Interpretation |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **3.2051 %** | Predictions deviate from true examination marks by ~3.2% on average. |
| **Root Mean Squared Error (RMSE)** | **3.9669 %** | Measures standard deviation of residuals; penalizes larger errors. |
| **R² Score ($R^2$)** | **0.8475** | **84.75%** of final examination variance is explained by the model features. |

*Note: Evaluation conducted on baseline synthetic demonstration data with realistic academic variance.*

---

## 🎓 Curated Viva Questions & Answers

1. **Why is Linear Regression chosen for this project?**  
   It provides high interpretability, instant training, and clear explanatory weights for educational stakeholders without black-box complexity.
2. **Why are Student ID and Name excluded from model features?**  
   Student ID and Name are administrative identifiers with no causal mathematical correlation to academic ability. Including them would introduce noise or spurious correlations.
3. **How does the system prevent division-by-zero in assignment rates?**  
   The calculation helper checks if total assignments $> 0$. If total is 0, submission rates safely default to $100.0\%$.
4. **Why is SQLite used instead of plain CSV files?**  
   SQLite provides ACID transactional consistency, indexed search, parameterized queries preventing SQL injection, and prevents data corruption during simultaneous accesses.
5. **How is target leakage prevented?**  
   The target `final_exam_marks` is strictly separated from $X$, and scaling statistics are computed strictly on training partitions inside the `Pipeline`.
6. **What does an R² score of 0.8475 mean?**  
   It means 84.75% of the variance in final examination scores is accounted for by the combination of attendance, study hours, IA marks, and assignment completion.
7. **What is the difference between MAE and RMSE?**  
   MAE measures the arithmetic average magnitude of errors. RMSE squares errors before averaging, penalizing larger deviations more severely.
8. **Why use OneHotEncoder with handle_unknown='ignore'?**  
   It converts categorical variables (like participation levels) into numeric dummy columns and safely handles novel or unseen category values at inference time without crashing.
9. **How does the system handle students with no assignments assigned?**  
   It sets missing and delayed assignments to 0 and defaults the submission rate to 100%, avoiding mathematical errors.
10. **Can this model guarantee a student's final marks?**  
    No. Predictions are statistical estimates based on historical trends; unpredictable factors like illness or sudden personal issues cannot be modeled.

---

## ⚠️ Limitations & Ethical Guidelines
- **Advisory Purpose Only:** Predictions must be used exclusively to guide remedial mentoring, never for punitive actions or tracking.
- **Data Attribution:** The current demonstration uses synthetic data with realistic relationships. When deploying at a college, 2–3 semesters of real institutional data must be imported.
- **Local Prototype Scope:** This version runs on a local workstation; an institutional deployment requires institutional Single Sign-On (SSO) and role-based permissions.

---

## ✅ Submission Checklist
- [x] Complete source code in modular architecture (`app.py`, `train_model.py`, `database.py`, `preprocessing.py`).
- [x] SQLite database persistence in `data/students.db` with parameterized queries.
- [x] 4-Section Teacher Input Form with client-side input validation.
- [x] Auto-calculations for IA averages, delayed submissions, missing assignments, and rates.
- [x] Scikit-Learn `ColumnTransformer` + `LinearRegression` pipeline.
- [x] Real empirical evaluation results: MAE = 3.21%, RMSE = 3.97%, $R^2$ = 0.8475.
- [x] Actual vs Predicted evaluation plot in `models/actual_vs_predicted.png`.
- [x] 10/10 passing unit tests in `tests/test_prediction.py`.
- [x] Academic project report in `reports/project_report.md`.
- [x] Comprehensive documentation in `README.md`.
