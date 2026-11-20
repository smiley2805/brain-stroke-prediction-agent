

import os
import sys
import sqlite3
import joblib
import shap
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DB_PATH = BASE_DIR / "stroke_records.db"

sys.path.insert(0, str(BASE_DIR / "src"))

from report_generator import generate_patient_report

# --------------------------------------------------
# 2. STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Stroke Prediction Agent",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# 3. DATABASE INITIALIZATION
# --------------------------------------------------

def initialize_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date TEXT,
            gender TEXT,
            age REAL,
            hypertension INTEGER,
            heart_disease INTEGER,
            ever_married TEXT,
            work_type TEXT,
            residence_type TEXT,
            avg_glucose_level REAL,
            bmi REAL,
            smoking_status TEXT,
            prediction INTEGER,
            probability REAL
        )
    """)

    connection.commit()
    connection.close()


def save_assessment(record):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO assessments (
            date,
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            avg_glucose_level,
            bmi,
            smoking_status,
            prediction,
            probability
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, record)

    assessment_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return assessment_id


def get_all_assessments():

    connection = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM assessments ORDER BY id DESC",
        connection
    )

    connection.close()

    return df


initialize_database()

# --------------------------------------------------
# 4. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = MODEL_DIR / "logistic_regression.pkl"

    if not model_path.exists():
        return None

    return joblib.load(model_path)


model = load_model()

# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🧠 Stroke AI Agent")

st.sidebar.caption("Brain Stroke Prediction System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Patient Assessment",
        "Model Analytics",
        "AI Explainability",
        "Patient History"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Educational Machine Learning Project"
)

# --------------------------------------------------
# 6. HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("🧠 Brain Stroke Prediction Agent")

    st.write(
        "An educational machine learning application "
        "for analyzing patient records and explaining "
        "classification results."
    )

    st.divider()

    records = get_all_assessments()

    if len(records) > 0:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Assessments",
            len(records)
        )

        col2.metric(
            "Classified Stroke-Positive",
            int((records["prediction"] == 1).sum())
        )

        col3.metric(
            "Classified Stroke-Negative",
            int((records["prediction"] == 0).sum())
        )

    else:

        st.info(
            "No assessment records yet. "
            "Submit a patient assessment to see dashboard analytics."
        )

    st.divider()

    st.header("About This Project")

    st.write(
        """
        This project uses machine learning algorithms to
        classify historical patient records according to
        their stroke label in the dataset.
        """
    )

    st.subheader("Main Features")

    st.markdown("""
    - Patient assessment and classification
    - Multiple machine learning models
    - Model performance analytics
    - SHAP-based individual explanations
    - AI-generated model explanations
    - Patient assessment history using SQLite
    - Individual PDF report generation
    - CSV history export
    """)

    st.warning(
        "Educational use only. Model outputs are not medical "
        "diagnoses or clinically validated predictions of "
        "future stroke risk."
    )

# --------------------------------------------------
# 7. PATIENT ASSESSMENT PAGE
# --------------------------------------------------

elif page == "Patient Assessment":

    st.title("🩺 Patient Assessment")

    st.write(
        "Enter patient details to perform an educational "
        "machine-learning classification."
    )

    if model is None:

        st.error(
            "Trained model not found. "
            "Check models/logistic_regression.pkl."
        )

    else:

        with st.form("patient_form"):

            st.subheader("Patient Information")

            col1, col2 = st.columns(2)

            with col1:

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other"]
                )

                age = st.number_input(
                    "Age",
                    min_value=0.0,
                    max_value=120.0,
                    value=45.0
                )

                hypertension = st.selectbox(
                    "Hypertension",
                    [0, 1],
                    format_func=lambda x:
                    "Yes" if x == 1 else "No"
                )

                heart_disease = st.selectbox(
                    "Heart Disease",
                    [0, 1],
                    format_func=lambda x:
                    "Yes" if x == 1 else "No"
                )

                ever_married = st.selectbox(
                    "Ever Married",
                    ["Yes", "No"]
                )

            with col2:

                work_type = st.selectbox(
                    "Work Type",
                    [
                        "Private",
                        "Self-employed",
                        "Govt_job",
                        "children",
                        "Never_worked"
                    ]
                )

                residence_type = st.selectbox(
                    "Residence Type",
                    ["Urban", "Rural"]
                )

                avg_glucose_level = st.number_input(
                    "Average Glucose Level",
                    min_value=0.0,
                    max_value=500.0,
                    value=100.0
                )

                bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0
                )

                smoking_status = st.selectbox(
                    "Smoking Status",
                    [
                        "formerly smoked",
                        "never smoked",
                        "smokes",
                        "Unknown"
                    ]
                )

            submitted = st.form_submit_button(
                "Analyze Patient",
                use_container_width=True
            )

        if submitted:

            patient_data = pd.DataFrame([{
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status
            }])

            try:

                prediction = int(
                    model.predict(patient_data)[0]
                )

                probability = float(
                    model.predict_proba(patient_data)[0][1]
                )

                assessment_date = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                record = (
                    assessment_date,
                    gender,
                    age,
                    hypertension,
                    heart_disease,
                    ever_married,
                    work_type,
                    residence_type,
                    avg_glucose_level,
                    bmi,
                    smoking_status,
                    prediction,
                    probability
                )

                assessment_id = save_assessment(record)

                # Save latest patient information
                st.session_state["latest_patient"] = patient_data

                st.session_state["latest_prediction"] = prediction

                st.session_state["latest_probability"] = probability

                st.session_state["latest_assessment_id"] = assessment_id

                st.session_state["latest_record"] = (
                    assessment_id,
                    assessment_date,
                    gender,
                    age,
                    hypertension,
                    heart_disease,
                    ever_married,
                    work_type,
                    residence_type,
                    avg_glucose_level,
                    bmi,
                    smoking_status,
                    prediction,
                    probability
                )

                st.success("Assessment saved successfully!")

                st.divider()

                st.subheader("🤖 Model Output")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Predicted Class",
                    prediction
                )

                col2.metric(
                    "Class 1 Probability",
                    f"{probability * 100:.2f}%"
                )

                if prediction == 1:

                    st.warning(
                        "The model classified this record "
                        "as stroke-positive."
                    )

                else:

                    st.success(
                        "The model classified this record "
                        "as stroke-negative."
                    )

                st.info(
                    "This is an educational classification of a "
                    "historical dataset label. It is not a medical "
                    "diagnosis or a validated prediction of future "
                    "stroke risk."
                )

                st.subheader("📄 Download Patient Report")

                pdf_bytes = generate_patient_report(
                    st.session_state["latest_record"]
                )

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"patient_report_{assessment_id}.pdf",
                    mime="application/pdf"
                )

            except Exception as error:

                st.error(
                    f"Error during assessment: {error}"
                )

# --------------------------------------------------
# 8. MODEL ANALYTICS PAGE
# --------------------------------------------------

elif page == "Model Analytics":

    st.title("📊 Model Analytics")

    st.write(
        "Compare the performance of all trained "
        "machine learning models."
    )

    st.divider()

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    chart_path = RESULTS_DIR / "model_comparison.png"

    if comparison_path.exists():

        results_df = pd.read_csv(comparison_path)

        st.subheader("Model Performance Comparison")

        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "ROC-AUC": "{:.4f}"
            }),
            use_container_width=True
        )

        st.divider()

        best_model = results_df.loc[
            results_df["F1 Score"].idxmax()
        ]

        st.subheader("Best Performing Model")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Model",
            best_model["Model"]
        )

        col2.metric(
            "F1 Score",
            f'{best_model["F1 Score"]:.4f}'
        )

        col3.metric(
            "ROC-AUC",
            f'{best_model["ROC-AUC"]:.4f}'
        )

        st.divider()

        st.subheader("Visual Model Comparison")

        if chart_path.exists():

            st.image(
                str(chart_path),
                use_container_width=True
            )

        else:

            st.warning(
                "Comparison chart not found. "
                "Run src/compare_models.py."
            )

    else:

        st.warning(
            "Model comparison results not found. "
            "Run python src/compare_models.py first."
        )

# --------------------------------------------------
# 9. AI EXPLAINABILITY PAGE
# --------------------------------------------------

elif page == "AI Explainability":

    st.title("🔍 AI Explainability")

    st.write(
        "Understand which input features influenced "
        "the model's classification."
    )

    latest_patient = st.session_state.get(
        "latest_patient"
    )

    if latest_patient is None:

        st.info(
            "Please complete a patient assessment first "
            "to view individual SHAP explanations."
        )

    elif model is None:

        st.error("Trained model not found.")

    else:

        try:

            # Get preprocessor and classifier
            preprocessor = model.named_steps["preprocessor"]

            classifier = model.named_steps["classifier"]

            transformed = preprocessor.transform(
                latest_patient
            )

            feature_names = preprocessor.get_feature_names_out()

            # SHAP explanation for Logistic Regression
            explainer = shap.LinearExplainer(
                classifier,
                transformed
            )

            shap_values = explainer.shap_values(
                transformed
            )

            if isinstance(shap_values, list):
                values = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": values
            })

            explanation_df["Feature"] = (
                explanation_df["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )

            explanation_df["SHAP Value"] = (
                explanation_df["SHAP Value"].astype(float)
            )

            explanation_df["Absolute Value"] = (
                explanation_df["SHAP Value"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute Value",
                ascending=False
            )

            st.subheader("Feature Contributions")

            st.dataframe(
                explanation_df[
                    ["Feature", "SHAP Value"]
                ].head(10),
                use_container_width=True
            )

            st.caption(
                "Positive SHAP values push the model output upward, "
                "while negative SHAP values push it downward. "
                "These contributions describe model behavior, "
                "not medical causes."
            )

            st.divider()

            st.subheader("AI-Generated Model Explanation")

            top_features = explanation_df.head(4)

            explanation_text = []

            for _, row in top_features.iterrows():

                direction = (
                    "positive"
                    if row["SHAP Value"] > 0
                    else "negative"
                )

                explanation_text.append(
                    f"**{row['Feature']}** contributed "
                    f"{direction}ly "
                    f"({row['SHAP Value']:.4f})"
                )

            st.markdown(
                ". ".join(explanation_text) + "."
            )

            st.info(
                "This explanation is automatically generated "
                "from SHAP values. It describes model behavior, "
                "not medical causes or future stroke risk."
            )

        except Exception as error:

            st.error(
                f"Could not generate SHAP explanation: {error}"
            )

            st.info(
                "Check that your Logistic Regression pipeline "
                "contains steps named 'preprocessor' and 'classifier'."
            )

# --------------------------------------------------
# 10. PATIENT HISTORY PAGE
# --------------------------------------------------

elif page == "Patient History":

    st.title("📋 Patient History")

    st.write(
        "View previous assessments and download reports."
    )

    history_df = get_all_assessments()

    if history_df.empty:

        st.info("No patient assessments available yet.")

    else:

        st.metric(
            "Total Assessments",
            len(history_df)
        )

        st.subheader("Assessment Records")

        display_df = history_df.copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Download Assessment History")

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="patient_history.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("Download Individual PDF Report")

        selected_id = st.selectbox(
            "Select Assessment ID",
            history_df["id"].tolist()
        )

        if st.button("Generate PDF Report"):

            selected_record = history_df[
                history_df["id"] == selected_id
            ].iloc[0]

            record = tuple(selected_record)

            pdf_bytes = generate_patient_report(
                record
            )

            st.download_button(
                label="Download Selected Patient PDF",
                data=pdf_bytes,
                file_name=f"patient_report_{selected_id}.pdf",
                mime="application/pdf"
            )

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Brain Stroke Prediction Agent | Educational Project"
)
