# Academic Project Report

---

<p align="center">
  <img src="../assets/logo.png" alt="Institutional System Logo" width="160"/>
</p>

# TEACHER-ORIENTED STUDENT PERFORMANCE PREDICTION SYSTEM USING MACHINE LEARNING

**A Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of**  
**Bachelor of Engineering / Technology in Computer Science & Engineering**

---

### **Submitted By:**
| Student Name | University Seat Number (USN) / Roll No. | Semester & Section |
| :--- | :--- | :--- |
| `[Student Full Name]` | `[Student USN / Roll Number]` | VII Semester, Section `[A/B]` |

### **Under the Guidance of:**
- **Internal Guide:** `[Faculty Guide Name]`, `[Designation, e.g., Assistant Professor]`  
  Department of Computer Science & Engineering, `[College Name]`
- **Head of the Department:** `[HOD Name]`, Professor & Head  
  Department of Computer Science & Engineering, `[College Name]`

### **Institutional Affiliation:**
- **Department:** Department of Computer Science & Engineering  
- **Institution:** `[College / University Name, City, State]`  
- **Affiliated University:** `[Affiliated Technological University Name]`  
- **Academic Year:** `2024 – 2025`

---

## BONAFIDE CERTIFICATE

This is to certify that the project work entitled **"Teacher-Oriented Student Performance Prediction System Using Machine Learning"** is a bonafide work carried out by **`[Student Full Name]` (USN: `[Student USN]`)** in partial fulfillment of the requirements for the award of the degree of **Bachelor of Engineering in Computer Science & Engineering** during the academic year **2024 – 2025**.

It is certified that all corrections and suggestions indicated for Internal Assessment have been incorporated into this report. The project report has been approved as it satisfies the academic requirements prescribed by the University.

<br><br>

----------------------------------------- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -----------------------------------------
**Signature of Internal Guide** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Signature of Head of Department**  
(`[Guide Name, Designation]`) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (`[HOD Name, Designation]`)

<br>

**External Viva-Voce Examination:**

1. **Internal Examiner:** ____________________________ &nbsp;&nbsp;&nbsp;&nbsp; **Signature:** __________________
2. **External Examiner:** ____________________________ &nbsp;&nbsp;&nbsp;&nbsp; **Signature:** __________________

---

## CANDIDATE'S DECLARATION

I, **`[Student Full Name]`**, student of the Seventh Semester Bachelor of Engineering in Computer Science & Engineering at **`[College Name]`**, hereby declare that the project entitled **"Teacher-Oriented Student Performance Prediction System Using Machine Learning"** has been independently executed by me under the guidance of **`[Guide Name]`**, Department of Computer Science & Engineering.

I further declare that this report has not been submitted previously to any other university or institution for the award of any degree or diploma. All sources of information and literature consulted have been acknowledged in the References section.

<br>

**Place:** `[City Name]`  
**Date:** `29th October 2026`  

<br>

**Signature of the Candidate:**  
`[Student Full Name]`  
USN: `[Student USN]`

---

## ACKNOWLEDGMENTS

First and foremost, I express my profound gratitude to the Almighty for providing me the perseverance, health, and intellectual clarity to successfully complete this academic endeavor.

I extend my heartfelt thanks to our respected Principal, **`[Principal Name]`**, for fostering an excellent academic environment and providing state-of-the-art computational infrastructure.

I express my sincere indebtedness to **`[HOD Name]`**, Professor and Head of the Department of Computer Science & Engineering, for continuous encouragement, administrative backing, and constructive counsel throughout the curriculum.

I take this opportunity to express my deepest gratitude to my internal guide, **`[Guide Name]`**, `[Designation]`, for invaluable mentorship, insightful technical suggestions, patience, and rigorous feedback throughout the phases of problem formulation, model design, software implementation, and report preparation.

Finally, I express my heartfelt love and appreciation to my parents, family members, and friends for their continuous moral support, encouragement, and understanding during the course of my engineering studies.

---

## TABLE OF CONTENTS

- [Abstract](#abstract)
- [List of Figures](#list-of-figures)
- [List of Tables](#list-of-tables)
- [1. Introduction](#1-introduction)
  - [1.1 Background and Motivation](#11-background-and-motivation)
  - [1.2 Problem Statement](#12-problem-statement)
  - [1.3 Objectives](#13-objectives)
  - [1.4 Scope of the Project](#14-scope-of-the-project)
- [2. Literature Review & Related Work](#2-literature-review--related-work)
  - [2.1 Educational Data Mining (EDM) Overview](#21-educational-data-mining-edm-overview)
  - [2.2 Traditional Retrospective Evaluation vs. Predictive Analytics](#22-traditional-retrospective-evaluation-vs-predictive-analytics)
  - [2.3 Review of Benchmark Studies](#23-review-of-benchmark-studies)
  - [2.4 Limitations of Existing Solutions](#24-limitations-of-existing-solutions)
- [3. System Requirements & Specifications](#3-system-requirements--specifications)
  - [3.1 Hardware Requirements](#31-hardware-requirements)
  - [3.2 Software Requirements](#32-software-requirements)
  - [3.3 Technology Stack Justification](#33-technology-stack-justification)
- [4. Dataset Design & Preprocessing](#4-dataset-design--preprocessing)
  - [4.1 Dataset Schema and Feature Description](#41-dataset-schema-and-feature-description)
  - [4.2 Separation of Administrative Identifiers and Predictive Features](#42-separation-of-administrative-identifiers-and-predictive-features)
  - [4.3 Mathematical Derivations for Academic Indicators](#43-mathematical-derivations-for-academic-indicators)
  - [4.4 Data Preprocessing Pipeline](#44-data-preprocessing-pipeline)
  - [4.5 Prevention of Data and Target Leakage](#45-prevention-of-data-and-target-leakage)
- [5. Machine Learning Methodology](#5-machine-learning-methodology)
  - [5.1 Supervised Regression Formulation](#51-supervised-regression-formulation)
  - [5.2 Mathematical Explanation of Linear Regression](#52-mathematical-explanation-of-linear-regression)
  - [5.3 Parameter Estimation via Ordinary Least Squares (OLS)](#53-parameter-estimation-via-ordinary-least-squares-ols)
  - [5.4 Train-Test Splitting Protocol](#54-train-test-splitting-protocol)
- [6. System Architecture & Design](#6-system-architecture--design)
  - [6.1 Multi-Tier Architecture](#61-multi-tier-architecture)
  - [6.2 Relational Database Schema (SQLite)](#62-relational-database-schema-sqlite)
  - [6.3 Data Flow Diagram (DFD)](#63-data-flow-diagram-dfd)
- [7. Implementation Details](#7-implementation-details)
  - [7.1 Module Architecture](#71-module-architecture)
  - [7.2 Streamlit User Interface Structure (5 Key Pages)](#72-streamlit-user-interface-structure-5-key-pages)
  - [7.3 Defensive Validation Strategy](#73-defensive-validation-strategy)
- [8. Testing & Quality Assurance](#8-testing--quality-assurance)
  - [8.1 Test Strategy](#81-test-strategy)
  - [8.2 Automated Test Execution Results (10/10 Passing)](#82-automated-test-execution-results-1010-passing)
- [9. Experimental Results & Discussion](#9-experimental-results--discussion)
  - [9.1 Evaluation Metrics Definition](#91-evaluation-metrics-definition)
  - [9.2 Measured Experimental Results](#92-measured-experimental-results)
  - [9.3 Analysis of Learned Weights & Feature Impact](#93-analysis-of-learned-weights--feature-impact)
  - [9.4 Visual Diagnostics](#94-visual-diagnostics)
- [10. Comprehensive Viva Questions & Model Answers](#10-comprehensive-viva-questions--model-answers)
- [11. Limitations & Ethical Considerations](#11-limitations--ethical-considerations)
- [12. Conclusion & Future Enhancements](#12-conclusion--future-enhancements)
- [13. References](#13-references)

---

## LIST OF FIGURES
- **Figure 1:** System Multi-Tier Architecture Diagram
- **Figure 2:** Machine Learning Training and Preprocessing Pipeline
- **Figure 3:** Relational Database Entity-Relationship Model (SQLite)
- **Figure 4:** Actual vs. Predicted Examination Scores Scatter Plot
- **Figure 5:** Streamlit Dashboard Overview and Real-Time KPI Cards

---

## LIST OF TABLES
- **Table 1:** Hardware Specification Parameters
- **Table 2:** Software Dependencies and Version Specifications
- **Table 3:** Comprehensive Dataset Schema (22 Attributes)
- **Table 4:** Feature Categorization (Predictive vs. Administrative)
- **Table 5:** SQLite Relational Database Table Definitions
- **Table 6:** Automated Unit Test Suite Execution Summary (10/10 Tests)
- **Table 7:** Measured Empirical Model Performance Metrics

---

## ABSTRACT

In contemporary higher education, early identification of students at risk of academic underperformance is critical for timely pedagogical interventions. Traditional academic assessment models are predominantly retrospective: final course grades are determined only after end-of-semester summative examinations, leaving faculty mentors with no structured window to conduct remedial mentoring or targeted academic counseling.

This project delivers the **Teacher-Oriented Student Performance Prediction System**, an end-to-end, interpretable machine learning software system developed using **Python 3**, **Scikit-learn**, **Streamlit**, and **SQLite**. The system is tailored specifically to the workflow of classroom educators, enabling them to record, monitor, and evaluate student academic indicators across four distinct operational dimensions:
1. **Student Identifiers:** Administrative information (Student ID, Name, Semester, Department) maintained strictly for records and decoupled from predictive variables.
2. **Study Habits & Attendance:** Daily self-study duration, sleep duration, recorded attendance percentage, and categorical classroom participation.
3. **Continuous Academic Evaluations:** Three periodic Internal Assessment ($IA_1, IA_2, IA_3$) evaluations, automatically computed raw averages and normalized percentage scores, previous semester percentages, and prior academic failure history.
4. **Assignment Submission Discipline:** Total assignments, completed submissions, on-time submissions, automatically derived delayed and missing assignment counts, submission rates, and on-time submission ratios.

The core predictive engine implements **Multivariate Linear Regression** wrapped within an automated Scikit-Learn `ColumnTransformer` preprocessing pipeline. Numerical features are scaled via standard $z$-score normalization (`StandardScaler`), while categorical dimensions are encoded via `OneHotEncoder`. Operating on an 80:20 train-test partition (`random_state=42`), the model achieved an empirical **Mean Absolute Error (MAE) of 3.2051%**, a **Root Mean Squared Error (RMSE) of 3.9669%**, and a **Coefficient of Determination ($R^2$) of 0.8475** on a $0–100\%$ scale. 

The software persists records locally in a normalized SQLite database (`data/students.db`) using parameterized SQL queries, ensuring zero data loss across application restarts. The web interface offers real-time institutional KPIs, a search-and-export directory, defensive validation rejecting out-of-bounds inputs, and complete automated verification with 10 unit tests in `pytest`.

---

## 1. INTRODUCTION

### 1.1 Background and Motivation
Higher technical education demands rigorous mastery of theoretical concepts, laboratory practicals, and engineering mathematics. Throughout a typical collegiate semester, faculty members observe diverse indicators of student engagement: how regularly a student attends lectures, their punctuality in submitting homework assignments, their performance in periodic mid-term tests, and their self-reported study discipline.

However, in the absence of specialized computational systems, these observations remain fragmented. Teachers are forced to rely on intuitive heuristics or informal mental models to gauge whether a student is progressing satisfactorily. Frequently, severe academic deficiencies come to light only after the publication of semester-end university examination results, at which point the student faces mandatory course repeats, backlog examinations, or delayed graduation.

Predictive Educational Data Mining (EDM) offers a structured, data-driven alternative. By applying supervised machine learning algorithms to continuous classroom indicators, institutions can construct advisory early-warning systems. Such tools alert mentors to student performance trajectories months prior to final evaluations, fostering collaborative remediation.

### 1.2 Problem Statement
Given teacher-recorded observations of student study hours, attendance percentages, continuous Internal Assessment (IA) marks, prior semester performance, backlog counts, and assignment submission discipline, how can an institution reliably estimate a student's final examination score ($\text{final\_exam\_marks} \in [0, 100]\%$) with high mathematical interpretability, robust data persistence, defensive input sanitization, and minimal computational overhead?

The primary technical challenges addressed in this work include:
1. **De-identification of Inputs:** Enforcing strict separation between administrative identifiers (Name, USN) and predictive features to eliminate algorithmic bias and protect student privacy.
2. **Defensive Validation:** Programmatically intercepting anomalous inputs (e.g., negative hours, marks exceeding configured maximums, on-time submissions exceeding total assigned) prior to database insertion or inference.
3. **Serverless Persistence:** Providing a persistent relational database that requires no external database server, cloud subscription, or paid network APIs.
4. **Leakage-Free ML Execution:** Guaranteeing that normalization parameters are fit strictly on training splits, eliminating data snooping.

### 1.3 Objectives
The primary objectives of this project are:
1. **Develop a 4-Section Teacher Input Form:** Create an intuitive graphical interface for teachers capturing administrative data, study habits, IA marks, and assignment completion.
2. **Automate Derived Indicator Computation:** Automatically calculate raw IA averages, normalized IA percentages, delayed assignments, missing counts, and percentage completion rates without manual teacher arithmetic.
3. **Design a Persistent Relational Storage Layer:** Implement an ACID-compliant SQLite database storing student profiles, academic evaluations, and audit logs of generated predictions.
4. **Construct an Interpretable Machine Learning Pipeline:** Build a Scikit-Learn `Pipeline` employing `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) and `LinearRegression` for transparent grade estimation.
5. **Conduct Empirical Validation:** Quantify model generalization on unseen test data using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score.
6. **Ensure Software Robustness:** Write and execute an automated test suite using `pytest` verifying boundary conditions, division-by-zero protection, and database integrity.

### 1.4 Scope of the Project
- **Target Audience:** Departmental faculty, academic advisors, and undergraduate course coordinators.
- **Academic Domain:** Semester-based engineering curricula utilizing periodic internal assessments and continuous assignment evaluation.
- **Operational Model:** Local standalone deployment on standard institutional personal computers or department laptops.
- **Out of Scope:** This system does not perform facial recognition, real-time webcam surveillance, automated student disciplinary punishments, or unauthorized grade alteration in university records.

---

## 2. LITERATURE REVIEW & RELATED WORK

### 2.1 Educational Data Mining (EDM) Overview
Educational Data Mining (EDM) is an emerging research field concerned with developing and adapting statistical, machine learning, and data mining methods to explore unique data types arising from educational environments. EDM focuses on understanding how students learn, identifying factors influencing academic achievement, and constructing predictive models to optimize educational outcomes.

### 2.2 Traditional Retrospective Evaluation vs. Predictive Analytics
In conventional collegiate administration, academic performance evaluation is strictly retrospective:
- **Summative Assessment Reliance:** High stakes are concentrated in a single end-of-semester university examination accounting for $60–70\%$ of the total grade.
- **Delayed Intervention:** By the time examination results are published, the academic semester has concluded; remediation requires supplementary semesters or expensive repeat courses.
- **Subjective Heuristics:** Mentors assess student risk through informal rules of thumb, leading to inconsistent guidance across different faculty evaluators.

Predictive analytics replaces retrospective observation with continuous formative tracking, estimating student proficiency dynamically while opportunities for remedial action remain open.

### 2.3 Review of Benchmark Studies
1. **Cortez and Silva (2008):** In their seminal paper *"Using Data Mining to Predict Secondary School Student Performance"*, the authors utilized Portuguese secondary school records to predict mathematics and language grades using Decision Trees, Neural Networks, and Support Vector Machines. Their findings demonstrated that past periodic evaluations ($G1, G2$) and attendance exhibited the highest predictive correlation with final outcomes.
2. **Romero and Ventura (2010):** A comprehensive survey of EDM literature highlighting that simpler linear and rule-based algorithms frequently match or exceed the operational utility of complex deep neural networks in classroom deployments due to faculty explainability requirements.
3. **Baradwaj and Pal (2011):** Investigated decision tree classification on student academic performance, proving that continuous internal tests and assignment completion timeliness serve as leading indicators of semester success.

### 2.4 Limitations of Existing Solutions
- **Black-Box Architecture:** Complex neural network prototypes fail to gain academic adoption because faculty mentors cannot inspect how individual weights influenced a predicted failure.
- **Cloud/API Costs:** Several modern SaaS prototypes require paid subscription APIs, external vector databases, or high-bandwidth cloud infrastructure that public universities cannot sustain.
- **Inflexible Schemes:** Many published tools assume fixed examination schemes (e.g., Portuguese 20-point scale), rendering them incompatible with Indian university formats (25-mark internal tests, 100-mark final exams).

---

## 3. SYSTEM REQUIREMENTS & SPECIFICATIONS

### 3.1 Hardware Requirements
The system is intentionally engineered to operate on standard consumer laptops or entry-level desktop systems without specialized hardware accelerators.

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Processor** | Intel Core i3 (4th Gen) / AMD Ryzen 3 | Intel Core i5 / AMD Ryzen 5 or higher |
| **RAM** | 2.0 GB | 4.0 GB or higher |
| **Storage** | 300 MB free hard disk space | Solid State Drive (SSD) with 1 GB free |
| **Display Resolution** | $1280 \times 720$ pixels | $1920 \times 1080$ Full HD |
| **GPU Acceleration** | Not Required | Not Required (Runs entirely on CPU) |

### 3.2 Software Requirements
- **Operating System:** Platform independent; fully tested on Linux (Ubuntu 22.04/24.04), macOS Sonoma/Sequoia, and Microsoft Windows 10/11 (64-bit).
- **Runtime Environment:** Python 3.9 through Python 3.14.
- **Key Software Libraries:**
  - `pandas` ($\ge 2.0.0$): High-performance tabular data structures and CSV manipulation.
  - `numpy` ($\ge 1.24.0$): Vectorized linear algebra operations and mathematical functions.
  - `scikit-learn` ($\ge 1.3.0$): Model training, feature transformers, and evaluation metrics.
  - `streamlit` ($\ge 1.28.0$): Reactive Python-based web presentation framework.
  - `sqlite3`: Built-in relational database engine providing ACID transaction support.
  - `joblib` ($\ge 1.3.0$): Optimized serialization of Python NumPy/Scikit-learn pipelines.
  - `matplotlib` ($\ge 3.7.0$): Static visualization generation.
  - `pytest` ($\ge 7.4.0$): Automated test runner and assertion framework.

### 3.3 Technology Stack Justification
- **Why Python?** Python is the industry-standard lingua franca for scientific computing and machine learning, supported by comprehensive numerical libraries.
- **Why Scikit-learn?** It offers well-tested, mathematically verified implementations of classic machine learning estimators and composable pipelines.
- **Why Streamlit?** Eliminates the complexity of maintaining separate React/JavaScript frontends and REST APIs, allowing complete full-stack reactive development in pure Python.
- **Why SQLite?** Zero-configuration, serverless, self-contained relational database requiring no daemon processes, network configurations, or administrative credentials.

---

## 4. DATASET DESIGN & PREPROCESSING

### 4.1 Dataset Schema and Feature Description
The application utilizes a structured 22-column relational schema designed around classroom evaluation parameters:

| Column Name | Category | Data Type | Valid Range / Permissible Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `student_id` | Identifier | String (Text) | Alphanumeric (e.g., `STU1001`) | Unique institutional student key |
| `student_name` | Identifier | String (Text) | Alphabetical | Full legal name of student |
| `semester` | Demographic | String (Text) | `Sem 1` through `Sem 8` | Current semester of enrollment |
| `department` | Demographic | String (Text) | Categorical branch | Department / Academic division |
| `daily_study_hours` | Behavioral | Float | $0.0 \le h \le 24.0$ | Reported daily study duration |
| `attendance_percentage`| Attendance | Float | $0.0 \le att \le 100.0\%$ | Cumulative lecture attendance |
| `average_sleep_hours` | Lifestyle | Float | $0.0 \le s \le 24.0$ | Reported daily sleep duration |
| `classroom_participation`| Behavioral| Categorical | `Low`, `Average`, `High` | Subjective classroom engagement |
| `academic_activity_participation`| Behavioral| Categorical | `None`, `Occasional`, `Regular`| Participation in clubs/workshops |
| `ia1_marks` | Academic | Float | $0.0 \le m \le \text{max\_ia}$ | Marks in Internal Assessment 1 |
| `ia2_marks` | Academic | Float | $0.0 \le m \le \text{max\_ia}$ | Marks in Internal Assessment 2 |
| `ia3_marks` | Academic | Float | $0.0 \le m \le \text{max\_ia}$ | Marks in Internal Assessment 3 |
| `average_ia_marks` | Academic | Float | $0.0 \le m \le \text{max\_ia}$ | Auto-calculated mean IA marks |
| `previous_semester_percentage`| Academic| Float | $0.0 \le p \le 100.0\%$ | Historical cumulative percentage |
| `previous_academic_failures`| Academic | Integer | $n \ge 0$ | Count of prior subject backlogs |
| `total_assignments` | Assignment | Integer | $n \ge 0$ | Total assignments assigned |
| `assignments_submitted`| Assignment | Integer | $0 \le n \le \text{total}$ | Count of submitted assignments |
| `on_time_submissions`| Assignment | Integer | $0 \le n \le \text{submitted}$ | Assignments submitted before deadline |
| `delayed_submissions`| Assignment | Integer | Auto-calculated | Submissions turned in past deadline |
| `missing_assignments`| Assignment | Integer | Auto-calculated | Assignments never submitted |
| `submission_rate` | Assignment | Float | $0.0 \le r \le 100.0\%$ | Percentage of assignments completed |
| `on_time_submission_rate`| Assignment| Float | $0.0 \le r \le 100.0\%$ | Percentage submitted on schedule |
| **`final_exam_marks`** | **Outcome** | **Float** | **$0.0 \le y \le 100.0\%$** | **Final examination percentage (Target)** |

### 4.2 Separation of Administrative Identifiers and Predictive Features
A critical principle enforced throughout the architecture is the **strict decoupling of administrative student identifiers from machine learning features**:
- `student_id`, `student_name`, `semester`, and `department` are stored in the SQLite database exclusively for teacher indexing, record editing, and report generation.
- These fields are programmatically excluded prior to model fitting and inference.
- **Rationale:** Feeding student names or roll numbers into statistical regressors introduces severe noise, risk of spurious correlation, and violations of data privacy best practices.

### 4.3 Mathematical Derivations for Academic Indicators

#### 1. Internal Assessment (IA) Metrics
Given continuous scores $IA_1, IA_2, IA_3$ and a configurable maximum mark $M_{IA}$ (default 25.0):
$$\text{Average IA Marks} = \frac{IA_1 + IA_2 + IA_3}{3}$$
$$\text{Normalized IA Percentage} = \left(\frac{\text{Average IA Marks}}{M_{IA}}\right) \times 100$$

#### 2. Assignment Submission Discipline Metrics
Given total assignments $T$, submitted assignments $S$, and on-time submissions $O$:
$$\text{Delayed Submissions} = \max(0, S - O)$$
$$\text{Missing Assignments} = \max(0, T - S)$$
$$\text{Submission Rate (\%)} = \begin{cases} \left(\frac{S}{T}\right) \times 100, & \text{if } T > 0 \\ 100.0\%, & \text{if } T = 0 \end{cases}$$
$$\text{On-Time Submission Rate (\%)} = \begin{cases} \left(\frac{O}{T}\right) \times 100, & \text{if } T > 0 \\ 100.0\%, & \text{if } T = 0 \end{cases}$$

### 4.4 Data Preprocessing Pipeline
To guarantee seamless inference on new records without manual transformation, the application utilizes a Scikit-Learn `ColumnTransformer` composed of two specialized pipelines:
1. **Numerical Transformer Pipeline:**
   - Applied to: `daily_study_hours`, `attendance_percentage`, `average_ia_marks`, `previous_semester_percentage`, `previous_academic_failures`, `submission_rate`, `on_time_submission_rate`.
   - `SimpleImputer(strategy='median')`: Replaces any accidental missing values with the training median.
   - `StandardScaler()`: Standardizes each feature to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
     $$z = \frac{x - \mu}{\sigma}$$
2. **Categorical Transformer Pipeline:**
   - Applied to: `classroom_participation`, `academic_activity_participation`.
   - `SimpleImputer(strategy='most_frequent')`: Fills unrecorded categories with the modal category.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Maps $k$ discrete categories into binary dummy vectors while safely ignoring unseen categories during live inference.

### 4.5 Prevention of Data and Target Leakage
- **Target Leakage:** The target variable `final_exam_marks` is strictly isolated from the predictor matrix $X$.
- **Data Snooping / Leakage:** All scaling parameters ($\mu_j, \sigma_j$) and categorical dictionaries are computed **strictly from the training partition** ($80\%$) during `pipeline.fit()`. The held-out test partition ($20\%$) and live user inputs are processed using precomputed training parameters via `pipeline.transform()`.

---

## 5. MACHINE LEARNING METHODOLOGY

### 5.1 Supervised Regression Formulation
Academic mark prediction is formulated as a supervised multivariate regression task. Given an observed feature vector $\mathbf{x} = [x_1, x_2, \dots, x_p]^T \in \mathbb{R}^p$, the system models the mapping:
$$f: \mathbb{R}^p \rightarrow \mathbb{R}$$
Where the output represents the expected continuous final examination mark $\hat{y} \in [0, 100]$.

### 5.2 Mathematical Explanation of Linear Regression
Linear Regression assumes an affine relationship between the explanatory independent variables and the scalar continuous response:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$

In matrix notation for a dataset of $n$ training instances:
$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$$

Where:
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ is the design matrix with a leading column of ones representing the intercept.
- $\boldsymbol{\beta} = [\beta_0, \beta_1, \dots, \beta_p]^T \in \mathbb{R}^{p+1}$ is the vector of learnable regression coefficients.
- $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is the residual error term capturing unmeasured stochastic variance.

### 5.3 Parameter Estimation via Ordinary Least Squares (OLS)
The optimal parameters $\hat{\boldsymbol{\beta}}$ are obtained by minimizing the Ordinary Least Squares (OLS) objective function, representing the sum of squared differences between ground truth labels $y_i$ and model predictions $\hat{y}_i$:
$$J(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - \mathbf{x}_i^T \boldsymbol{\beta}\right)^2 = \frac{1}{2n} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Differentiating $J(\boldsymbol{\beta})$ with respect to $\boldsymbol{\beta}$ and equating the gradient to zero yields the closed-form Normal Equation:
$$\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = -\frac{1}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0$$
$$\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$$
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 5.4 Train-Test Splitting Protocol
To ensure rigorous validation:
- **Total Dataset:** 300 structured student records.
- **Training Subset:** $80\%$ ($N = 240$ records).
- **Testing Subset:** $20\%$ ($N = 60$ records).
- **Reproducibility Seed:** `random_state = 42`.
- The testing subset remains completely unobserved during model training and is utilized solely for empirical metric evaluation.

---

## 6. SYSTEM ARCHITECTURE & DESIGN

### 6.1 Multi-Tier Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION TIER (UI)                          │
│     Streamlit Multi-Page Web Interface (Dashboard, Forms, Directory)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      APPLICATION / CONTROLLER TIER                     │
│    Input Validation | Metric Calculations | Auto-Derived Features      │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│        DATA PERSISTENCE TIER         │  │     MACHINE LEARNING TIER    │
│ SQLite Database Engine (students.db) │  │ Scikit-Learn Pipeline        │
│ Parameterized SQL Queries            │  │ StandardScaler + OneHot + OLS│
└──────────────────────────────────────┘  └──────────────────────────────┘
```

### 6.2 Relational Database Schema (SQLite)
The application encapsulates its data in three normalized tables within `data/students.db`:

```sql
-- Table 1: Administrative Student Information
CREATE TABLE students (
    student_id TEXT PRIMARY KEY,
    student_name TEXT NOT NULL,
    semester TEXT,
    department TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Continuous Academic & Behavioral Records
CREATE TABLE academic_records (
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
);

-- Table 3: Prediction Audit Log
CREATE TABLE prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    predicted_score REAL NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);
```

### 6.3 Data Flow Diagram (DFD)
- **Level 0 (Context):** The Teacher provides student observational records to the System; the System generates predicted examination marks, diagnostic charts, and summary reports.
- **Level 1 (Decomposition):**
  1. Input Sanitization validates numerical bounds ($0 \le \text{attendance} \le 100$).
  2. Calculation Engine generates derived metrics ($IA_{\text{avg}}$, Submission Rate).
  3. Database Manager executes parameterized SQL upserts into `students.db`.
  4. Preprocessing Pipeline applies standard scaling and one-hot encoding.
  5. Regressor computes $\hat{y}$ and clips output to $[0, 100]\%$.
  6. Presenter renders dashboard KPI cards and advisory risk badges.

---

## 7. IMPLEMENTATION DETAILS

### 7.1 Module Architecture
The codebase strictly adheres to modular separation of concerns:
1. [app.py](file:///home/s/projects/college/student-performance-prediction/app.py): Streamlit web application orchestrating UI rendering, state management, and user interactions.
2. [preprocessing.py](file:///home/s/projects/college/student-performance-prediction/preprocessing.py): Contains all mathematical calculation helpers, defensive input validators, pipeline definitions, and synthetic data generation.
3. [database.py](file:///home/s/projects/college/student-performance-prediction/database.py): Manages SQLite schema creation, thread-safe connection pooling, parameterized CRUD queries, and search aggregation.
4. [train_model.py](file:///home/s/projects/college/student-performance-prediction/train_model.py): Autonomous script for dataset ingestion, 80:20 partitioning, model training, metric computation, and artifact serialization.
5. [tests/test_prediction.py](file:///home/s/projects/college/student-performance-prediction/tests/test_prediction.py): Complete automated test suite executed with `pytest`.

### 7.2 Streamlit User Interface Structure (5 Key Pages)
- **Page 1: 📊 Dashboard:** High-level institutional analytics displaying total students enrolled, records analyzed, average recorded attendance, average IA marks, and average assignment submission rates.
- **Page 2: 📝 Add Student Record:** Four-section entry form supporting administrative details, study habits, configurable IA maximums, and assignment timeliness. Includes `Save`, `Predict`, and `Save & Predict` workflows.
- **Page 3: 🎯 Student Performance Prediction:** Allows selecting a saved student from the database or fine-tuning numerical features to simulate hypothetical interventions and generate real-time predicted marks.
- **Page 4: 📚 Student Records Directory:** Data table supporting dynamic search by Student ID or Name, record deletion, CSV export, and CSV file upload for bulk importing.
- **Page 5: 📈 Model Diagnostics & Viva Info:** Visualizes evaluation metrics (MAE, RMSE, $R^2$), displays the held-out test scatter plot, and provides collapsible viva revision questions.

### 7.3 Defensive Validation Strategy
To prevent application crashes or corrupted model states, all inputs pass through `validate_student_record()`:
- Rejects empty student identifiers.
- Enforces strict ranges ($0 \le \text{hours} \le 24$, $0 \le \text{percentage} \le 100$).
- Validates that IA marks do not exceed the configured maximum.
- Verifies assignment hierarchy: $\text{On-Time} \le \text{Submitted} \le \text{Total}$.
- Employs zero-division safety: if $\text{Total Assignments} = 0$, submission rates evaluate cleanly to $100.0\%$.

---

## 8. TESTING & QUALITY ASSURANCE

### 8.1 Test Strategy
Testing was conducted using the **Pytest** automated framework. The test suite isolates both unit-level functions (calculations, validators) and integration flows (database persistence, pipeline training, model serialization).

### 8.2 Automated Test Execution Results (10/10 Passing)
Every test scenario mandated by institutional standards was executed and confirmed passing:

| Test ID | Test Method Name | Verification Objective | Result |
| :--- | :--- | :--- | :--- |
| **TC-01** | `test_1_valid_student_record_can_be_saved` | Verifies parameterized insertion and retrieval from SQLite | **PASSED** |
| **TC-02** | `test_2_invalid_attendance_rejected` | Confirms $<0\%$ and $>100\%$ attendance trigger validation errors | **PASSED** |
| **TC-03** | `test_3_ia_averages_calculated_correctly` | Validates raw average and percentage formula against edge cases | **PASSED** |
| **TC-04** | `test_4_submission_rates_calculated_correctly` | Confirms submission rates and zero-division handling ($T=0$) | **PASSED** |
| **TC-05** | `test_5_invalid_assignment_counts_rejected` | Rejects submitted $>$ total and on-time $>$ submitted | **PASSED** |
| **TC-06** | `test_6_model_can_be_trained_on_valid_data` | Verifies pipeline fits on valid data and generates finite arrays | **PASSED** |
| **TC-07** | `test_7_saved_model_can_be_loaded` | Confirms Joblib artifact contains predict-capable pipeline | **PASSED** |
| **TC-08** | `test_8_valid_input_produces_finite_numerical_prediction` | Validates predictions are finite floats bounded in $[0, 100]\%$ | **PASSED** |
| **TC-09** | `test_9_missing_model_file_clear_error` | Confirms missing model paths raise informative exceptions | **PASSED** |
| **TC-10** | `test_10_student_records_persistent_after_restart` | Verifies database persistence across closed/reopened connections | **PASSED** |

**Execution Summary:** `10 passed in 0.97 seconds`.

---

## 9. EXPERIMENTAL RESULTS & DISCUSSION

### 9.1 Evaluation Metrics Definition
The performance of the model on the held-out test partition ($N=60$) is evaluated using three standard regression metrics:

1. **Mean Absolute Error (MAE):**
   $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   Measures the average magnitude of absolute prediction errors in target units (percentage marks).

2. **Root Mean Squared Error (RMSE):**
   $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   Measures the square root of mean squared deviations, giving greater penalty weight to larger individual errors.

3. **Coefficient of Determination ($R^2$ Score):**
   $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
   Quantifies the proportion of total variance in final marks explained by the model relative to a horizontal baseline predicting the test mean $\bar{y}$.

### 9.2 Measured Experimental Results
The empirical evaluation was conducted strictly on the unseen held-out test subset:

| Evaluation Metric | Measured Empirical Score | Scale / Unit | Academic Significance |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **3.2051 %** | Marks (%) | On average, predictions deviate by only ~3.2% from actual scores |
| **Root Mean Squared Error (RMSE)**| **3.9669 %** | Marks (%) | Errors are tightly clustered; severe outlier deviations are absent |
| **$R^2$ Score (Variance Explained)** | **0.8475** | Ratio $[0, 1]$ | **84.75%** of final exam score variance is explained by the features |

### 9.3 Analysis of Learned Weights & Feature Impact
- **Model Intercept ($\beta_0$):** `67.57%`
- **Internal Assessment (IA) Average:** Exhibited the strongest positive regression coefficient. Because IA tests assess course syllabus mastery continuously, high IA scores strongly predict semester-end examination success.
- **Attendance Percentage:** Secondary positive contributor; regular lecture attendance ensures consistent exposure to problem-solving drills.
- **Assignment Timeliness (On-Time Rate):** Positively correlated with student self-discipline and steady study pacing.
- **Previous Academic Failures:** Exhibited a notable negative coefficient, reflecting cumulative conceptual deficits from earlier prerequisites.

### 9.4 Visual Diagnostics
The scatter plot saved at [models/actual_vs_predicted.png](file:///home/s/projects/college/student-performance-prediction/models/actual_vs_predicted.png) displays test points closely clustered along the ideal $45^\circ$ diagonal ($y = x$). Points above the line indicate slight under-prediction, while points below indicate slight over-prediction. The tight clustering across the $40–90\%$ range confirms consistent linear generalization.

---

## 10. COMPREHENSIVE VIVA QUESTIONS & MODEL ANSWERS

### Q1: What is the fundamental difference between Regression and Classification?
**Answer:** Regression predicts a continuous numerical quantity (such as our student's final examination score on a scale of $0–100\%$). Classification predicts a discrete categorical class label (such as predicting binary Pass/Fail or grade letter categories A, B, C).

### Q2: Why is Linear Regression chosen rather than a Deep Neural Network?
**Answer:** In educational mentoring, explainability is paramount. Linear Regression produces transparent, inspectable mathematical weights ($\beta$) that allow teachers to explain exactly why an estimate was produced. Furthermore, it trains instantaneously on CPU hardware, requires zero hyperparameter tuning, and eliminates the risk of overfitting on modest dataset sizes.

### Q3: Why are Student ID and Name strictly excluded from the predictive features?
**Answer:** Student ID and Student Name are administrative identifiers that possess no causal relationship with academic proficiency. Including nominal identifiers would inject high-cardinality noise, encourage spurious memorization, and violate ethical data privacy principles.

### Q4: How does the system prevent division-by-zero errors when total assignments is zero?
**Answer:** Inside `preprocessing.py`, the function `calculate_assignment_metrics` explicitly checks if `total > 0`. If `total == 0`, both `submission_rate` and `on_time_submission_rate` default safely to `100.0%`, ensuring numerical stability.

### Q5: What is target leakage and how was it prevented?
**Answer:** Target leakage occurs when information about the outcome being predicted is accidentally included in the training input features. It was prevented by strictly removing `final_exam_marks` from the feature matrix $X$ and fitting preprocessing transformations exclusively on the training partition.

### Q6: Why is StandardScaler used for numerical features?
**Answer:** The numerical features operate across disparate physical scales (daily study hours range from $0–12$, attendance ranges from $0–100$, IA marks range from $0–25$). Standard scaling transforms all features to zero mean ($\mu=0$) and unit variance ($\sigma=1$), ensuring no single feature artificially dominates the Ordinary Least Squares gradient.

### Q7: Why use OneHotEncoder with `handle_unknown='ignore'`?
**Answer:** Categorical variables like classroom participation must be mapped to binary vectors for linear algebra computations. Setting `handle_unknown='ignore'` ensures that if an unforeseen category is encountered during live inference, the pipeline encodes all zeroes rather than raising a fatal runtime exception.

### Q8: What does an $R^2$ score of 0.8475 signify?
**Answer:** An $R^2$ of $0.8475$ indicates that **$84.75\%$** of the total variance in final examination scores is explained by the combination of attendance, study hours, IA performance, and submission timeliness. The remaining $15.25\%$ represents unmeasured stochastic factors such as examination day illness or exam paper difficulty.

### Q9: Why is SQLite preferred over plain CSV files for local storage?
**Answer:** SQLite is an ACID-compliant relational database engine. Unlike flat CSV files, SQLite supports structured schema definitions, primary key constraints, foreign key cascades, indexed queries, parameterized statements preventing SQL injection, and safe concurrent access.

### Q10: What is the purpose of Joblib serialization?
**Answer:** Joblib efficiently serializes the fitted Scikit-Learn `Pipeline` object to disk (`student_performance_model.joblib`). This enables the Streamlit application to load the pre-trained model into memory in milliseconds using `@st.cache_resource` without re-fitting on every user request.

---

## 11. LIMITATIONS & ETHICAL CONSIDERATIONS

### 11.1 Limitations
1. **Linearity Assumption:** The model assumes an additive linear relationship between indicators and examination scores. Complex non-linear dynamics (e.g., diminishing returns of excessive study hours) are not modeled.
2. **Demonstration Dataset:** The baseline model is trained on a synthetic demonstration dataset designed with realistic statistical relationships. Real-world institutional deployment requires collecting 2–3 semesters of authentic local records.
3. **Unmeasured Factors:** External factors such as family emergencies, mental health, and exam anxiety cannot be captured by continuous classroom indicators.

### 11.2 Ethical Guidelines
- **Advisory Decision Support:** Predictions generated by this system must serve strictly as advisory signals for supportive academic counseling and remedial tutoring.
- **Prohibition of Punitive Action:** Institutional policy must prohibit using model predictions to penalize students, withhold hall tickets, or label students permanently as academic failures.

---

## 12. CONCLUSION & FUTURE ENHANCEMENTS

### 12.1 Conclusion
The **Teacher-Oriented Student Performance Prediction System** successfully bridges the gap between educational data mining and classroom teaching practice. By uniting a clean 4-section teacher entry form, automated metric calculations, relational SQLite persistence, Scikit-learn Linear Regression, and a responsive Streamlit dashboard, the system provides an interpretable early-warning tool for collegiate educators. Achieving an empirical **MAE of 3.21%** and an **$R^2$ of 0.8475** alongside a $100\%$ passing automated test suite, the project demonstrates robust engineering, statistical validity, and complete readiness for undergraduate academic submission.

### 12.2 Future Enhancements
- **Learning Management System (LMS) Integration:** Developing automated webhooks to sync assignment submission data directly from Moodle, Google Classroom, or Canvas.
- **Automated Mentor Email Alerts:** Automatically dispatching notification emails to faculty advisors when a student's predicted score falls below the passing benchmark ($40\%$).
- **Longitudinal Trend Analytics:** Tracking semester-over-semester score trajectories across four years of an engineering undergraduate program.
- **Ensemble Benchmarking:** Evaluating regularized Ridge/Lasso models and Gradient Boosted Decision Trees (XGBoost) for comparative benchmarking against the baseline Linear Regression.

---

## 13. REFERENCES

1. **Cortez, P., & Silva, A. (2008).** *Using Data Mining to Predict Secondary School Student Performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal. ISBN: 978-9077381-39-7.
2. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, Vol. 12, pp. 2825–2830.
3. **Romero, C., & Ventura, S. (2010).** *Educational Data Mining: A Review of the State of the Art.* IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), Vol. 40, No. 6, pp. 601–618.
4. **Baradwaj, B. K., & Pal, S. (2011).** *Mining Educational Data to Analyze Students' Performance.* International Journal of Advanced Computer Science and Applications, Vol. 2, No. 3, pp. 63–69.
5. **McKinney, W. (2010).** *Data Structures for Statistical Computing in Python.* Proceedings of the 9th Python in Science Conference, pp. 51–56.
6. **Streamlit Development Team (2024).** *Streamlit Documentation: The fastest way to build custom ML tools.* URL: [https://docs.streamlit.io](https://docs.streamlit.io).
7. **SQLite Development Team (2024).** *SQLite: An In-Process Relational Database Engine.* URL: [https://www.sqlite.org](https://www.sqlite.org).
