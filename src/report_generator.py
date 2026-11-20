
from io import BytesIO
import os
import html

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from explainability import explain_patient


# ==========================================
# 1. GENERATE NATURAL LANGUAGE EXPLANATION
# ==========================================

def generate_ai_explanation(explanation_df):

    positive_features = explanation_df[
        explanation_df["SHAP Value"] > 0
    ].sort_values(
        by="SHAP Value",
        ascending=False
    )

    negative_features = explanation_df[
        explanation_df["SHAP Value"] < 0
    ].sort_values(
        by="SHAP Value",
        ascending=True
    )

    sentences = []

    # Largest positive contribution
    if not positive_features.empty:

        feature = html.escape(
            str(positive_features.iloc[0]["Feature"])
        )

        value = positive_features.iloc[0]["SHAP Value"]

        sentences.append(
            f"<b>{feature}</b> had the largest positive "
            f"contribution to the model output ({value:+.4f})."
        )

    # Largest negative contribution
    if not negative_features.empty:

        feature = html.escape(
            str(negative_features.iloc[0]["Feature"])
        )

        value = negative_features.iloc[0]["SHAP Value"]

        sentences.append(
            f"<b>{feature}</b> had the largest negative "
            f"contribution to the model output ({value:+.4f})."
        )

    # Second positive contribution
    if len(positive_features) > 1:

        feature = html.escape(
            str(positive_features.iloc[1]["Feature"])
        )

        value = positive_features.iloc[1]["SHAP Value"]

        sentences.append(
            f"<b>{feature}</b> also contributed positively "
            f"({value:+.4f})."
        )

    # Second negative contribution
    if len(negative_features) > 1:

        feature = html.escape(
            str(negative_features.iloc[1]["Feature"])
        )

        value = negative_features.iloc[1]["SHAP Value"]

        sentences.append(
            f"<b>{feature}</b> also contributed negatively "
            f"({value:+.4f})."
        )

    if not sentences:

        return (
            "No substantial positive or negative feature "
            "contributions were identified."
        )

    return " ".join(sentences)


# ==========================================
# 2. GENERATE PATIENT PDF REPORT
# ==========================================

def generate_patient_report(record):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    # ======================================
    # TITLE
    # ======================================

    elements.append(
        Paragraph(
            "Brain Stroke Classification Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            "Educational Machine Learning Assessment",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 25))

    # ======================================
    # PATIENT INFORMATION
    # ======================================

    elements.append(
        Paragraph(
            "1. Patient Information",
            styles["Heading2"]
        )
    )

    patient_data = [
        ["Field", "Value"],
        ["Assessment ID", str(record[0])],
        ["Date", str(record[1])],
        ["Gender", str(record[2])],
        ["Age", str(record[3])],
        ["Hypertension", "Yes" if record[4] else "No"],
        ["Heart Disease", "Yes" if record[5] else "No"],
        ["Ever Married", str(record[6])],
        ["Work Type", str(record[7])],
        ["Residence", str(record[8])],
        ["Average Glucose Level", str(record[9])],
        ["BMI", str(record[10])],
        ["Smoking Status", str(record[11])]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[2.5 * inch, 3.5 * inch]
    )

    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#1F4E79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ])
    )

    elements.append(patient_table)
    elements.append(Spacer(1, 25))

    # ======================================
    # MODEL OUTPUT
    # ======================================

    elements.append(
        Paragraph(
            "2. Model Output",
            styles["Heading2"]
        )
    )

    prediction = int(record[12])
    probability = float(record[13]) * 100

    classification = (
        "Stroke-positive"
        if prediction == 1
        else "Stroke-negative"
    )

    output_data = [
        ["Field", "Value"],
        ["Predicted Class", str(prediction)],
        ["Classification", classification],
        ["Class 1 Probability", f"{probability:.2f}%"]
    ]

    output_table = Table(
        output_data,
        colWidths=[2.5 * inch, 3.5 * inch]
    )

    output_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#1F4E79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    elements.append(output_table)
    elements.append(Spacer(1, 25))

    # ======================================
    # INDIVIDUAL SHAP EXPLANATION
    # ======================================

    elements.append(
        Paragraph(
            "3. Individual Patient Explanation",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            "SHAP explains how individual features "
            "contribute to the machine learning model output.",
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 15))

    patient_data = {
        "gender": record[2],
        "age": record[3],
        "hypertension": record[4],
        "heart_disease": record[5],
        "ever_married": record[6],
        "work_type": record[7],
        "Residence_type": record[8],
        "avg_glucose_level": record[9],
        "bmi": record[10],
        "smoking_status": record[11]
    }

    try:

        shap_result = explain_patient(patient_data)

        chart_path = shap_result["chart_path"]
        explanation_df = shap_result["explanation"]

        # SHAP chart
        if os.path.exists(chart_path):

            shap_image = Image(
                chart_path,
                width=6.5 * inch,
                height=4.3 * inch,
                kind="proportional"
            )

            elements.append(shap_image)
            elements.append(Spacer(1, 15))

        # ==================================
        # TOP FIVE FEATURES
        # ==================================

        elements.append(
            Paragraph(
                "Top Five Influential Features",
                styles["Heading3"]
            )
        )

        top_features = explanation_df.head(5)

        feature_data = [
            ["Feature", "SHAP Value"]
        ]

        for _, row in top_features.iterrows():

            feature_data.append([
                html.escape(str(row["Feature"])),
                f'{row["SHAP Value"]:.4f}'
            ])

        feature_table = Table(
            feature_data,
            colWidths=[3.8 * inch, 2.2 * inch]
        )

        feature_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
            ])
        )

        elements.append(feature_table)
        elements.append(Spacer(1, 20))

        # ==================================
        # AI-GENERATED EXPLANATION
        # ==================================

        elements.append(
            Paragraph(
                "4. AI-Generated Model Explanation",
                styles["Heading2"]
            )
        )

        natural_explanation = generate_ai_explanation(
            explanation_df
        )

        elements.append(
            Paragraph(
                natural_explanation,
                styles["BodyText"]
            )
        )

        elements.append(Spacer(1, 15))

        elements.append(
            Paragraph(
                "Positive SHAP values push the model output "
                "upward, while negative SHAP values push it "
                "downward. These contributions describe model "
                "behavior, not medical causes.",
                styles["BodyText"]
            )
        )

    except Exception as error:

        elements.append(
            Paragraph(
                "The individual explanation could not be "
                "generated for this report.",
                styles["BodyText"]
            )
        )

        print("SHAP report error:", error)

    elements.append(Spacer(1, 25))

    # ======================================
    # DISCLAIMER
    # ======================================

    elements.append(
        Paragraph(
            "5. Important Disclaimer",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            "This report contains an educational machine-learning "
            "classification of a historical dataset label. It is "
            "not a medical diagnosis or a clinically validated "
            "prediction of future stroke risk. SHAP values explain "
            "model behavior and do not establish medical causation. "
            "Consult a qualified healthcare professional for "
            "medical advice.",
            styles["BodyText"]
        )
    )

    # ======================================
    # BUILD PDF
    # ======================================

    document.build(elements)

    buffer.seek(0)

    return buffer.getvalue()
