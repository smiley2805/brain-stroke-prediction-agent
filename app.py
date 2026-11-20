
import streamlit as st
import pandas as pd
import joblib
import os
import sys

# ==========================================
# 1. PROJECT PATHS AND IMPORTS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.insert(0, SRC_DIR)

from database import (
    initialize_database,
    save_assessment,
    get_all_assessments
)

from report_generator import (
    generate_patient_report,
    generate_ai_explanation
)

from explainability import explain_patient


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Brain Stroke Prediction Agent",
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# 3. INITIALIZE DATABASE
# ==========================================

initialize_database()


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression.pkl"
)


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🧠 Stroke AI Agent")

st.sidebar.caption(
    "Brain Stroke Prediction System"
)

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


# ==========================================
# 6. HOME PAGE
# ==========================================

if page == "Home":

    st.title("🧠 Brain Stroke Prediction Agent")

    st.write(
        "An educational machine learning application "
        "for analyzing patient records."
    )

    st.divider()

    records = get_all_assessments()

    if records:

        columns = [
            "ID", "Date", "Gender", "Age",
            "Hypertension", "Heart Disease",
            "Married", "Work Type", "Residence",
            "Glucose", "BMI", "Smoking",
            "Prediction", "Probability"
        ]

        df = pd.DataFrame(
            records,
            columns=columns
        )

        total = len(df)

        positive = int(
            (df["Prediction"] == 1).sum()
        )

        negative = int(
            (df["Prediction"] == 0).sum()
        )

        average_age = df["Age"].mean()

        st.subheader("📊 Dashboard Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Assessments", total)
        col2.metric("Class 1 Records", positive)
        col3.metric("Class 0 Records", negative)
        col4.metric("Average Age", f"{average_age:.1f}")

        st.divider()

        st.subheader("Prediction Distribution")

        chart_data = pd.DataFrame({
            "Classification": ["Class 0", "Class 1"],
            "Count": [negative, positive]
        })

        st.bar_chart(
            chart_data.set_index("Classification")
        )

        st.divider()

        st.subheader("Patient Age Analysis")

        age_data = pd.DataFrame({
            "Age": df["Age"]
        })

        age_data["Age Group"] = pd.cut(
            age_data["Age"],
            bins=[0, 18, 40, 60, 120],
            labels=["0-18", "19-40", "41-60", "61+"],
            include_lowest=True
        )

        age_counts = (
            age_data["Age Group"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(age_counts)

        st.divider()

        st.subheader("Recent Assessments")

        st.dataframe(
            df[[
                "ID",
                "Date",
                "Age",
                "Gender",
                "Prediction",
                "Probability"
            ]].head(5),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No assessment records yet. "
            "Submit a patient assessment to see dashboard analytics."
        )

    st.divider()

    st.subheader("About This Project")

    st.write("""
    This project uses machine learning algorithms to
    classify historical patient records according to
    their stroke label in the dataset.
    """)

    st.warning("""
    Educational use only. Model outputs are not medical
    diagnoses or clinically validated predictions of
    future stroke risk.
    """)


# ==========================================
# 7. PATIENT ASSESSMENT
# ==========================================

elif page == "Patient Assessment":

    st.title("👤 Patient Assessment")

    st.write("Enter patient information below.")

    if model is None:

        st.error(
            "Model file not found. Check "
            "models/logistic_regression.pkl"
        )

        st.stop()

    with st.form("patient_form"):

        st.subheader("Personal Information")

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

            smoking_status = st.selectbox(
                "Smoking Status",
                [
                    "formerly smoked",
                    "never smoked",
                    "smokes",
                    "Unknown"
                ]
            )

        st.divider()

        st.subheader("Medical Information")

        col3, col4 = st.columns(2)

        with col3:

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

        with col4:

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

        submitted = st.form_submit_button(
            "Analyze Patient",
            use_container_width=True
        )

    if submitted:

        patient_data = {
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
        }

        try:

            # ==================================
            # MODEL PREDICTION
            # ==================================

            patient_df = pd.DataFrame(
                [patient_data]
            )

            prediction = model.predict(
                patient_df
            )[0]

            probability = model.predict_proba(
                patient_df
            )[0][1]

            # ==================================
            # INDIVIDUAL SHAP EXPLANATION
            # ==================================

            result = explain_patient(
                patient_data
            )

            explanation_df = result["explanation"]

            # ==================================
            # SAVE ASSESSMENT
            # ==================================

            save_assessment(
                patient_data,
                prediction,
                probability
            )

            # ==================================
            # MODEL OUTPUT
            # ==================================

            st.divider()

            st.subheader("🤖 Model Output")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Class",
                int(prediction)
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

            st.success(
                "Assessment saved successfully!"
            )

            st.info("""
            This is an educational classification of a historical
            dataset label. It is not a medical diagnosis or a
            validated prediction of future stroke risk.
            """)

            # ==================================
            # INDIVIDUAL PATIENT EXPLANATION
            # ==================================

            st.divider()

            st.subheader(
                "🔍 Individual Patient Explanation"
            )

            st.write(
                "This chart explains which features "
                "influenced this particular model prediction."
            )

            st.image(
                result["chart_path"],
                caption="Individual SHAP Waterfall Plot",
                use_container_width=True
            )

            # ==================================
            # FEATURE CONTRIBUTIONS
            # ==================================

            st.subheader("Feature Contributions")

            st.dataframe(
                explanation_df[
                    ["Feature", "SHAP Value"]
                ].head(10),
                use_container_width=True,
                hide_index=True
            )

            st.caption("""
            Positive SHAP values push the model output upward,
            while negative SHAP values push it downward.
            These contributions describe model behavior,
            not medical causes.
            """)

            # ==================================
            # AI-GENERATED EXPLANATION
            # ==================================

            st.divider()

            st.subheader(
                "🤖 AI-Generated Model Explanation"
            )

            natural_explanation = generate_ai_explanation(
                explanation_df
            )

            st.markdown(
                natural_explanation,
                unsafe_allow_html=True
            )

            st.info("""
            This explanation is automatically generated
            from SHAP values. It describes model behavior,
            not medical causes or future stroke risk.
            """)

        except Exception as e:

            st.error(
                f"Unable to complete the assessment: {e}"
            )


# ==========================================
# 8. MODEL ANALYTICS
# ==========================================

elif page == "Model Analytics":

    st.title("📊 Model Analytics")

    st.write(
        "Compare the performance of six ML models."
    )

    comparison_path = os.path.join(
        BASE_DIR,
        "results",
        "model_comparison.png"
    )

    roc_path = os.path.join(
        BASE_DIR,
        "results",
        "roc_comparison.png"
    )

    if os.path.exists(comparison_path):

        st.subheader("Model Performance Comparison")

        st.image(
            comparison_path,
            use_container_width=True
        )

    else:

        st.info(
            "Run src/visualize_results.py to generate "
            "the model comparison chart."
        )

    st.divider()

    if os.path.exists(roc_path):

        st.subheader("ROC Curve Comparison")

        st.image(
            roc_path,
            use_container_width=True
        )

    else:

        st.info(
            "ROC comparison chart is not available yet."
        )


# ==========================================
# 9. AI EXPLAINABILITY
# ==========================================

elif page == "AI Explainability":

    st.title("🔍 AI Explainability")

    st.write("""
    Understand which features influence the
    Logistic Regression model's output.
    """)

    st.divider()

    shap_chart = os.path.join(
        BASE_DIR,
        "results",
        "shap_summary.png"
    )

    if os.path.exists(shap_chart):

        st.subheader("SHAP Summary Plot")

        st.image(
            shap_chart,
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP chart not found. Run "
            "python src/explainability.py first."
        )

    st.divider()

    st.subheader("Top Influential Features")

    features = [
        ("Hypertension", 0.144583),
        ("Average Glucose Level", 0.142011),
        ("Work Type: Self-employed", 0.126480),
        ("Residence Type: Rural", 0.117142),
        ("Ever Married: Yes", 0.113312),
        ("Work Type: Private", 0.112766),
        ("Gender: Male", 0.108625)
    ]

    for feature, importance in features:

        st.write(f"**{feature}**")

        st.progress(
            min(importance / 0.15, 1.0)
        )

        st.caption(
            f"Mean absolute SHAP value: {importance:.4f}"
        )

    st.divider()

    st.info("""
    These values represent average feature influence
    across the dataset. They do not explain an individual
    patient's prediction and do not establish medical
    causation.
    """)


# ==========================================
# 10. PATIENT HISTORY AND PDF REPORTS
# ==========================================

elif page == "Patient History":

    st.title("📄 Patient History")

    st.write(
        "View previous assessments and download reports."
    )

    records = get_all_assessments()

    if records:

        columns = [
            "ID",
            "Date",
            "Gender",
            "Age",
            "Hypertension",
            "Heart Disease",
            "Married",
            "Work Type",
            "Residence",
            "Glucose",
            "BMI",
            "Smoking",
            "Prediction",
            "Class 1 Probability"
        ]

        history_df = pd.DataFrame(
            records,
            columns=columns
        )

        st.metric(
            "Total Assessments",
            len(records)
        )

        st.subheader("Assessment Records")

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Download Individual Report")

        selected_id = st.selectbox(
            "Select Assessment ID",
            [record[0] for record in records]
        )

        selected_record = next(
            record for record in records
            if record[0] == selected_id
        )

        try:

            pdf = generate_patient_report(
                selected_record
            )

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf,
                file_name=f"patient_report_{selected_id}.pdf",
                mime="application/pdf"
            )

        except Exception as e:

            st.error(
                f"Unable to generate PDF report: {e}"
            )

        st.divider()

        st.subheader("Download Complete History")

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download History CSV",
            data=csv,
            file_name="patient_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No assessment records found. "
            "Submit a patient assessment first."
        )
