"""
app.py
======
Teacher-Oriented Student Performance Prediction System.
A Streamlit web application enabling educators to record academic details,
manage student records in a local SQLite database, and generate explainable
final examination predictions using a Scikit-Learn Linear Regression pipeline.

Run with:
    streamlit run app.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

import database
from preprocessing import (
    calculate_ia_metrics,
    calculate_assignment_metrics,
    validate_student_record,
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from train_model import predict_final_marks, MODEL_PATH, METRICS_PATH, PLOT_PATH

# Asset Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_ICON_PATH = os.path.join(ASSETS_DIR, "logo_icon.png")
LOGO_FULL_PATH = os.path.join(ASSETS_DIR, "logo.png")

# Streamlit Page Config
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon=LOGO_ICON_PATH if os.path.exists(LOGO_ICON_PATH) else "🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    .pred-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-val {
        font-size: 3.2rem;
        font-weight: 800;
        color: #1e40af;
    }
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Loads the Scikit-learn model artifact with Streamlit caching."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at '{MODEL_PATH}'. Please run 'python train_model.py' first."
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact, None
    except Exception as exc:
        return None, f"Error loading model artifact: {exc}"


@st.cache_data
def load_evaluation_metrics():
    """Loads empirical model evaluation metrics from metrics.json."""
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_page_header(title: str, subtitle: str):
    """Renders a consistent, professional page header with the institutional logo."""
    hcol1, hcol2 = st.columns([0.08, 0.92])
    with hcol1:
        if os.path.exists(LOGO_ICON_PATH):
            st.image(LOGO_ICON_PATH, width=64)
        else:
            st.markdown("# 🎓")
    with hcol2:
        st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)


def main():
    # Initialize SQLite database
    database.init_database()
    database.populate_sample_data(num_samples=50)

    # Sidebar Navigation Branding
    if os.path.exists(LOGO_ICON_PATH):
        st.sidebar.image(LOGO_ICON_PATH, width=100)
    elif os.path.exists(LOGO_FULL_PATH):
        st.sidebar.image(LOGO_FULL_PATH, width=100)
    else:
        st.sidebar.markdown("# 🎓")
    st.sidebar.title("Faculty Portal")
    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "📊 Dashboard",
            "📝 Add Student Record",
            "🎯 Student Performance Prediction",
            "📚 Student Records Directory",
            "📈 Model Diagnostics & Viva Info",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Teacher-Oriented Decision Support System v2.0")
    st.sidebar.caption("Backend: SQLite | ML: Scikit-learn Linear Regression")

    # Load Model
    model_artifact, load_err = load_trained_model()
    metrics_info = load_evaluation_metrics()

    # =========================================================================
    # PAGE 1: DASHBOARD
    # =========================================================================
    if page == "📊 Dashboard":
        render_page_header(
            "📊 Student Performance Overview Dashboard",
            "Real-time academic indicators calculated from saved institutional records."
        )

        summary = database.get_dashboard_summary()
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{summary['total_students']}</div>
                <div class="kpi-label">Total Students</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{summary['available_records']}</div>
                <div class="kpi-label">Records Analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{summary['average_attendance']}%</div>
                <div class="kpi-label">Avg Attendance</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{summary['average_ia_marks']}</div>
                <div class="kpi-label">Avg IA Marks</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{summary['average_submission_rate']}%</div>
                <div class="kpi-label">Avg Submission Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        c_left, c_right = st.columns([1.2, 1])

        with c_left:
            st.subheader("📋 Recent Student Records")
            recent_df = database.get_all_students_df().head(8)
            if not recent_df.empty:
                display_cols = [
                    "student_id", "student_name", "semester", "attendance_percentage",
                    "average_ia_marks", "submission_rate", "final_exam_marks"
                ]
                st.dataframe(recent_df[display_cols], width="stretch")
            else:
                st.info("No records found in database. Add students via the 'Add Student Record' tab.")

        with c_right:
            st.subheader("⚙️ Prediction Engine Status")
            if model_artifact:
                st.success("✅ Machine Learning Model Loaded & Ready")
                meta = model_artifact.get("metadata", {})
                st.markdown(f"""
                - **Algorithm:** `{meta.get('algorithm', 'LinearRegression')}`
                - **Predictive Features:** {len(meta.get('feature_names', []))} variables
                - **Training Source:** `{meta.get('data_source', 'N/A')}`
                - **Data Category:** *{'Synthetic Demonstration Data' if meta.get('is_synthetic') else 'Real Records'}*
                """)
                if metrics_info and "metrics" in metrics_info:
                    m = metrics_info["metrics"]
                    st.markdown(f"""
                    **Test Performance:**
                    - **MAE:** `{m['mae']}%` (Average error margin)
                    - **RMSE:** `{m['rmse']}%`
                    - **R² Score:** `{m['r2']}` ({m['r2'] * 100:.1f}% variance explained)
                    """)
            else:
                st.warning(f"⚠️ Model Unavailable: {load_err}")

    # =========================================================================
    # PAGE 2: ADD STUDENT RECORD
    # =========================================================================
    elif page == "📝 Add Student Record":
        render_page_header(
            "📝 Student Academic Performance Record",
            "Teacher entry form for tracking academic progress and generating predictions."
        )

        # Max IA Marks Configuration
        with st.expander("⚙️ Assessment Scheme Settings", expanded=False):
            max_ia = st.number_input("Maximum Marks for Internal Assessment (IA)", min_value=10.0, max_value=100.0, value=25.0, step=5.0)

        with st.form("student_record_form"):
            # SECTION A: Student Information
            st.markdown("### Section A: Student Information")
            st.caption("Identifiers used for administrative records. These fields are strictly excluded from predictive model features.")
            sa1, sa2, sa3, sa4 = st.columns(4)
            with sa1:
                s_id = st.text_input("Student ID *", placeholder="e.g. STU1025").strip()
            with sa2:
                s_name = st.text_input("Student Name *", placeholder="e.g. Rahul Sharma").strip()
            with sa3:
                sem = st.selectbox("Semester", ["Sem 1", "Sem 2", "Sem 3", "Sem 4", "Sem 5", "Sem 6", "Sem 7", "Sem 8"], index=4)
            with sa4:
                dept = st.selectbox("Department / Class", ["Computer Science", "Information Science", "Electronics & Comm.", "Mechanical", "Civil"], index=0)

            st.markdown("---")

            # SECTION B: Study Habits and Attendance
            st.markdown("### Section B: Study Habits and Attendance")
            st.caption("ℹ️ Study hours and sleep hours represent student-reported or mentor-collected estimates, not automated sensor measurements.")
            sb1, sb2, sb3 = st.columns(3)
            with sb1:
                study_hours = st.number_input("Daily Study Hours (0–24)", min_value=0.0, max_value=24.0, value=3.5, step=0.5)
                sleep_hours = st.number_input("Average Sleep Hours (0–24)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            with sb2:
                att_pct = st.slider("Attendance Percentage (0–100%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
            with sb3:
                class_part = st.selectbox("Classroom Participation", ["Low", "Average", "High"], index=1)
                act_part = st.selectbox("Academic Activity Participation", ["None", "Occasional", "Regular"], index=1)

            st.markdown("---")

            # SECTION C: Academic Performance
            st.markdown("### Section C: Academic Performance")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                ia1 = st.number_input(f"IA 1 Marks (0–{max_ia})", min_value=0.0, max_value=float(max_ia), value=18.0, step=0.5)
            with sc2:
                ia2 = st.number_input(f"IA 2 Marks (0–{max_ia})", min_value=0.0, max_value=float(max_ia), value=19.5, step=0.5)
            with sc3:
                ia3 = st.number_input(f"IA 3 Marks (0–{max_ia})", min_value=0.0, max_value=float(max_ia), value=20.0, step=0.5)
            with sc4:
                prev_pct = st.number_input("Previous Semester % (0–100)", min_value=0.0, max_value=100.0, value=74.0, step=0.5)
                failures = st.number_input("Previous Academic Failures", min_value=0, max_value=10, value=0, step=1)

            # Auto calculate IA summary
            ia_calc = calculate_ia_metrics(ia1, ia2, ia3, max_ia=max_ia)
            st.info(f"📊 **Calculated IA Summary:** Average Raw IA = **{ia_calc['average_ia_marks']} / {max_ia}** | Normalized IA Score = **{ia_calc['ia_percentage']}%**")

            st.markdown("---")

            # SECTION D: Assignment & Submission Habits
            st.markdown("### Section D: Assignment and Submission Habits")
            sd1, sd2, sd3, sd4 = st.columns(4)
            with sd1:
                total_assign = st.number_input("Total Assignments Assigned", min_value=0, max_value=50, value=10, step=1)
            with sd2:
                submitted_assign = st.number_input("Assignments Submitted", min_value=0, max_value=50, value=9, step=1)
            with sd3:
                on_time_assign = st.number_input("On-Time Submissions", min_value=0, max_value=50, value=8, step=1)
            with sd4:
                habit_override = st.selectbox(
                    "Overall Submission Habit Classification",
                    ["Consistently on time", "Usually on time", "Frequently delayed", "Frequently missing"],
                    index=1,
                    help="Calculated automatically from recorded data or manually classified."
                )

            # Assignment metrics calculation
            assign_calc = calculate_assignment_metrics(total_assign, submitted_assign, on_time_assign)
            st.info(
                f"📈 **Assignment Breakdown:** Delayed: **{assign_calc['delayed_submissions']}** | "
                f"Missing: **{assign_calc['missing_assignments']}** | "
                f"Submission Rate: **{assign_calc['submission_rate']}%** | "
                f"On-Time Rate: **{assign_calc['on_time_submission_rate']}%**"
            )

            st.markdown("---")
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                btn_save = st.form_submit_button("💾 Save Student Record", width="stretch")
            with b_col2:
                btn_predict = st.form_submit_button("🎯 Predict Marks", width="stretch")
            with b_col3:
                btn_both = st.form_submit_button("⚡ Save & Predict", width="stretch")

        # Form Action Handling
        if btn_save or btn_predict or btn_both:
            record_data = {
                "student_id": s_id,
                "student_name": s_name,
                "semester": sem,
                "department": dept,
                "daily_study_hours": study_hours,
                "attendance_percentage": att_pct,
                "average_sleep_hours": sleep_hours,
                "classroom_participation": class_part,
                "academic_activity_participation": act_part,
                "ia1_marks": ia1,
                "ia2_marks": ia2,
                "ia3_marks": ia3,
                "average_ia_marks": ia_calc["average_ia_marks"],
                "previous_semester_percentage": prev_pct,
                "previous_academic_failures": failures,
                "total_assignments": total_assign,
                "assignments_submitted": submitted_assign,
                "on_time_submissions": on_time_assign,
                "delayed_submissions": assign_calc["delayed_submissions"],
                "missing_assignments": assign_calc["missing_assignments"],
                "submission_rate": assign_calc["submission_rate"],
                "on_time_submission_rate": assign_calc["on_time_submission_rate"],
                "final_exam_marks": None,
            }

            validation_errors = validate_student_record(record_data, max_ia=max_ia)
            if validation_errors:
                for field, msg in validation_errors.items():
                    st.error(f"❌ {msg}")
            else:
                pred_score = None
                if btn_predict or btn_both:
                    if model_artifact:
                        pred_score = predict_final_marks(model_artifact["pipeline"], record_data)
                    else:
                        st.error("Model is not loaded. Cannot generate prediction.")

                if btn_save or btn_both:
                    database.save_student_record(record_data)
                    if pred_score is not None:
                        database.record_prediction(s_id, pred_score)
                    st.success(f"✅ Student record for **{s_name} ({s_id})** saved successfully to SQLite database.")

                if pred_score is not None:
                    st.markdown("### 🎯 Predicted Result")
                    is_passing = pred_score >= 40.0
                    st.markdown(f"""
                    <div class="pred-card">
                        <div style="font-size: 1.15rem; color: #1e3a8a; font-weight: 600;">Predicted Final Examination Score</div>
                        <div class="pred-val">{pred_score:.1f}%</div>
                        <div style="color: #475569; font-size: 1rem;">Scale: 0 – 100% (Advisory Statistical Estimate)</div>
                        <div class="status-badge" style="background-color: {'#dcfce7' if is_passing else '#fee2e2'}; color: {'#166534' if is_passing else '#991b1b'};">
                            {'Projected Status: Passing Benchmark (≥ 40%)' if is_passing else 'Projected Status: Below Passing Benchmark (< 40%)'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(
                        "📌 **Academic Advisory:** This prediction is generated using a Linear Regression model "
                        "trained on historical academic indicators. It is an estimate to assist academic mentoring "
                        "and does not represent a guaranteed examination outcome or label a student definitively."
                    )

    # =========================================================================
    # PAGE 3: STUDENT PERFORMANCE PREDICTION
    # =========================================================================
    elif page == "🎯 Student Performance Prediction":
        render_page_header(
            "🎯 Student Performance Prediction",
            "Select a saved student or fine-tune indicators to evaluate expected examination performance."
        )

        if not model_artifact:
            st.error(f"⚠️ Model unavailable: {load_err}. Please execute `python train_model.py` first.")
            st.stop()

        all_students = database.get_all_students_df()
        options = ["-- Select Existing Student --"] + [f"{r['student_id']} - {r['student_name']}" for _, r in all_students.iterrows()]

        selected_opt = st.selectbox("Choose a Saved Student to Load Details:", options=options)

        # Pre-fill values
        default_data = {
            "daily_study_hours": 3.5,
            "attendance_percentage": 78.0,
            "average_ia_marks": 18.0,
            "previous_semester_percentage": 70.0,
            "previous_academic_failures": 0,
            "submission_rate": 80.0,
            "on_time_submission_rate": 70.0,
            "classroom_participation": "Average",
            "academic_activity_participation": "Occasional",
        }

        selected_student_id = None
        if selected_opt != "-- Select Existing Student --":
            selected_student_id = selected_opt.split(" - ")[0]
            fetched = database.get_student_record(selected_student_id)
            if fetched:
                default_data.update(fetched)
                st.success(f"Loaded records for **{fetched['student_name']}** ({selected_student_id})")

        st.markdown("#### Adjust Predictive Features for Scenario Analysis")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            in_study = st.number_input("Daily Study Hours", 0.0, 24.0, float(default_data["daily_study_hours"]), 0.5)
            in_att = st.slider("Attendance Percentage", 0.0, 100.0, float(default_data["attendance_percentage"]), 1.0)
            in_prev = st.number_input("Previous Semester %", 0.0, 100.0, float(default_data["previous_semester_percentage"]), 0.5)
        with pcol2:
            in_ia = st.number_input("Average IA Marks (out of 25)", 0.0, 25.0, float(default_data["average_ia_marks"]), 0.5)
            in_fail = st.number_input("Past Failures", 0, 10, int(default_data["previous_academic_failures"]), 1)
            in_cpart = st.selectbox("Classroom Participation", ["Low", "Average", "High"], index=["Low", "Average", "High"].index(default_data["classroom_participation"]) if default_data["classroom_participation"] in ["Low", "Average", "High"] else 1)
        with pcol3:
            in_sub = st.slider("Submission Rate %", 0.0, 100.0, float(default_data["submission_rate"]), 1.0)
            in_ontime = st.slider("On-Time Submission Rate %", 0.0, 100.0, float(default_data["on_time_submission_rate"]), 1.0)
            in_apart = st.selectbox("Academic Activity Participation", ["None", "Occasional", "Regular"], index=["None", "Occasional", "Regular"].index(default_data["academic_activity_participation"]) if default_data["academic_activity_participation"] in ["None", "Occasional", "Regular"] else 1)

        if st.button("🔮 Calculate Predicted Examination Score", width="stretch"):
            feature_dict = {
                "daily_study_hours": in_study,
                "attendance_percentage": in_att,
                "average_ia_marks": in_ia,
                "previous_semester_percentage": in_prev,
                "previous_academic_failures": in_fail,
                "submission_rate": in_sub,
                "on_time_submission_rate": in_ontime,
                "classroom_participation": in_cpart,
                "academic_activity_participation": in_apart,
            }

            pred = predict_final_marks(model_artifact["pipeline"], feature_dict)
            if selected_student_id:
                database.record_prediction(selected_student_id, pred)

            st.markdown("---")
            is_pass = pred >= 40.0
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size: 1.15rem; color: #1e3a8a; font-weight: 600;">Predicted Final Examination Score</div>
                <div class="pred-val">{pred:.1f}%</div>
                <div style="font-size: 1rem; color: #475569;">Scale: 0 – 100 Marks</div>
                <div class="status-badge" style="background-color: {'#dcfce7' if is_pass else '#fee2e2'}; color: {'#166534' if is_pass else '#991b1b'};">
                    {'Advisory Status: Passing Benchmark (≥ 40%)' if is_pass else 'Advisory Status: Potential Academic Risk (< 40%)'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature breakdown card
            st.markdown("#### Key Input Metrics Summary")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Attendance", f"{in_att}%")
            k2.metric("Average IA", f"{in_ia} / 25")
            k3.metric("Submission Rate", f"{in_sub}%")
            k4.metric("Daily Study", f"{in_study} hrs")

            st.warning("""
            ⚠️ **Model Limitations & Ethics:**
            - This prediction is an estimate derived from a mathematical regression model trained on student historical/synthetic data.
            - It does not capture personal health issues, examination anxiety, or unmeasured life events.
            - Educational policy recommends using predictions solely for constructive remedial guidance, not punitive classification.
            """)

    # =========================================================================
    # PAGE 4: STUDENT RECORDS DIRECTORY
    # =========================================================================
    elif page == "📚 Student Records Directory":
        render_page_header(
            "📚 Student Records Directory",
            "Search, inspect, delete, and export records saved in the local SQLite database."
        )

        # Search bar
        search_query = st.text_input("🔍 Search by Student ID or Name:", placeholder="e.g. STU1001 or Sharma")

        if search_query:
            students_df = database.search_students(search_query)
            st.caption(f"Showing results for '{search_query}' ({len(students_df)} matching records)")
        else:
            students_df = database.get_all_students_df()

        if not students_df.empty:
            st.dataframe(students_df, width="stretch")

            # CSV Download
            csv_data = students_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Student Records to CSV",
                data=csv_data,
                file_name="student_records_export.csv",
                mime="text/csv",
            )
        else:
            st.info("No student records found matching the query.")

        st.markdown("---")
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            st.subheader("🗑️ Delete Student Record")
            del_id = st.text_input("Enter Student ID to Delete:", placeholder="e.g. STU1005")
            if st.button("Delete Record", type="secondary"):
                if del_id:
                    database.delete_student_record(del_id)
                    st.success(f"Record for {del_id} deleted successfully.")
                    st.rerun()

        with d_col2:
            st.subheader("📤 Import Records from CSV")
            uploaded_file = st.file_uploader("Upload CSV containing historical student data:", type=["csv"])
            if uploaded_file is not None:
                try:
                    import_df = pd.read_csv(uploaded_file)
                    st.write(f"Detected {len(import_df)} rows.")
                    if st.button("Import Records into Database"):
                        count = 0
                        for _, row in import_df.iterrows():
                            database.save_student_record(row.to_dict())
                            count += 1
                        st.success(f"Successfully imported {count} student records!")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Error reading uploaded CSV: {exc}")

        st.info("🔒 **Institutional Security Notice:** This prototype operates as a local workstation demonstration. A full university deployment must implement institutional Single Sign-On (SSO), Role-Based Access Control (RBAC), and FERPA/GDPR compliance.")

    # =========================================================================
    # PAGE 5: MODEL DIAGNOSTICS & VIVA INFO
    # =========================================================================
    elif page == "📈 Model Diagnostics & Viva Info":
        render_page_header(
            "📈 Model Diagnostics & Academic Viva Guide",
            "Technical evaluation metrics, plots, and standard questions for project evaluation."
        )

        if metrics_info and "metrics" in metrics_info:
            m = metrics_info["metrics"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean Absolute Error (MAE)", f"{m['mae']}%", help="Average magnitude of prediction errors.")
            m2.metric("Root Mean Squared Error (RMSE)", f"{m['rmse']}%", help="Outlier-sensitive quadratic error metric.")
            m3.metric("R² Score (Variance Explained)", f"{m['r2']}", help="Proportion of final mark variation explained.")

            st.markdown(f"""
            - **MAE = {m['mae']}%**: On average, the model's prediction deviates by ~{m['mae']}% from true marks.
            - **RMSE = {m['rmse']}%**: Confirms predictions are tightly distributed without large catastrophic outlier errors.
            - **R² Score = {m['r2']}**: **{m['r2'] * 100:.1f}%** of final exam score variance is explained by the combination of attendance, study hours, IA performance, and submission discipline.
            """)
        else:
            st.info("Metrics not found. Run `python train_model.py` to regenerate metrics.")

        if os.path.exists(PLOT_PATH):
            st.image(PLOT_PATH, caption="Held-out Test Set: Actual vs. Predicted Examination Scores", width="stretch")

        st.markdown("---")
        st.subheader("🎓 Key Viva Questions and Model Answers")
        with st.expander("1. Why is Linear Regression chosen for this project?"):
            st.write("Linear Regression provides high interpretability, instant training, and clear explanatory weights for educational stakeholders without black-box complexity.")
        with st.expander("2. Why are Student ID and Name excluded from model features?"):
            st.write("Student ID and Name are administrative identifiers with no predictive mathematical correlation to academic ability. Including them would cause noise or spurious correlation.")
        with st.expander("3. How is target leakage prevented?"):
            st.write("The target variable (final_exam_marks) is strictly excluded from input features, and preprocessing transformations (StandardScaler, OneHotEncoder) are fit strictly on training splits.")
        with st.expander("4. What role does SQLite play in this system?"):
            st.write("SQLite provides reliable, serverless, local relational data persistence for student demographics, academic evaluations, and prediction history using parameterized SQL queries.")


if __name__ == "__main__":
    main()
