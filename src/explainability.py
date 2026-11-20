

import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "archive (2)",
    "healthcare-dataset-stroke-data.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# ==========================================
# 3. LOAD DATASET
# ==========================================

data = pd.read_csv(DATA_PATH)

data = data.drop("id", axis=1)

X = data.drop("stroke", axis=1)


# ==========================================
# 4. PREPROCESS DATA
# ==========================================

X_processed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()


# ==========================================
# 5. MAKE FEATURE NAMES READABLE
# ==========================================

def make_readable_name(feature):

    # Remove preprocessing prefixes
    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")

    # Convert one-hot encoded features
    categorical_features = [
        "gender",
        "work_type",
        "Residence_type",
        "ever_married",
        "smoking_status"
    ]

    for column in categorical_features:

        prefix = column + "_"

        if feature.startswith(prefix):

            value = feature[len(prefix):]

            column_name = column.replace(
                "_", " "
            ).title()

            return f"{column_name}: {value.replace('_', ' ').title()}"

    # Convert numerical feature names
    readable = feature.replace("_", " ").title()

    return readable


display_names = [
    make_readable_name(name)
    for name in feature_names
]


# ==========================================
# 6. CREATE SHAP EXPLAINER
# ==========================================

explainer = shap.LinearExplainer(
    classifier,
    X_processed
)


# ==========================================
# 7. EXPLAIN INDIVIDUAL PATIENT
# ==========================================

def explain_patient(patient_data):

    # Convert patient information into DataFrame
    patient_df = pd.DataFrame(
        [patient_data]
    )

    # Apply training preprocessor
    patient_processed = preprocessor.transform(
        patient_df
    )

    # Convert sparse matrix into normal array
    if hasattr(patient_processed, "toarray"):

        patient_values = patient_processed.toarray()[0]

    else:

        patient_values = patient_processed[0]

    # Calculate SHAP values
    shap_values = explainer(
        patient_processed
    )

    contributions = shap_values.values[0]

    base_value = shap_values.base_values[0]

    # Get prediction
    prediction = model.predict(
        patient_df
    )[0]

    probability = model.predict_proba(
        patient_df
    )[0][1]

    # ======================================
    # 8. CREATE EXPLANATION TABLE
    # ======================================

    explanation = pd.DataFrame({

        "Feature": display_names,

        "Patient Value": patient_values,

        "SHAP Value": contributions

    })

    explanation["Absolute Importance"] = (
        explanation["SHAP Value"].abs()
    )

    explanation = explanation.sort_values(
        by="Absolute Importance",
        ascending=False
    ).reset_index(drop=True)

    # ======================================
    # 9. CREATE NAMED SHAP EXPLANATION
    # ======================================

    named_explanation = shap.Explanation(

        values=contributions,

        base_values=base_value,

        data=patient_values,

        feature_names=display_names

    )

    # ======================================
    # 10. GENERATE WATERFALL CHART
    # ======================================

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        named_explanation,
        max_display=12,
        show=False
    )

    plt.title(
        "Individual Patient SHAP Explanation"
    )

    chart_path = os.path.join(
        RESULTS_DIR,
        "individual_shap.png"
    )

    plt.savefig(
        chart_path,
        bbox_inches="tight",
        dpi=150
    )

    plt.close()

    # ======================================
    # 11. RETURN RESULTS
    # ======================================

    return {

        "prediction": int(prediction),

        "probability": float(probability),

        "explanation": explanation,

        "chart_path": chart_path

    }


# ==========================================
# 12. TEST WITH SAMPLE PATIENT
# ==========================================

if __name__ == "__main__":

    sample_patient = {

        "gender": "Male",

        "age": 45,

        "hypertension": 0,

        "heart_disease": 0,

        "ever_married": "Yes",

        "work_type": "Private",

        "Residence_type": "Urban",

        "avg_glucose_level": 100,

        "bmi": 25,

        "smoking_status": "formerly smoked"

    }

    result = explain_patient(
        sample_patient
    )

    print("\nPatient Prediction:")

    print(result["prediction"])

    print("\nClass 1 Probability:")

    print(
        f'{result["probability"] * 100:.2f}%'
    )

    print("\nTop Influential Features:")

    print(
        result["explanation"][
            ["Feature", "SHAP Value"]
        ].head(10)
    )

    print("\nSHAP analysis completed successfully!")

    print(
        "Chart saved at:",
        result["chart_path"]
    )
