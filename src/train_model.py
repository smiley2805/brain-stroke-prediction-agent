
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv(
    "data/archive (2)/healthcare-dataset-stroke-data.csv"
)

data = data.drop("id", axis=1)

X = data.drop("stroke", axis=1)
y = data["stroke"]


# ==========================================
# 2. IDENTIFY COLUMN TYPES
# ==========================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. NUMERICAL PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==========================================
# 5. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 6. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# ==========================================
# 7. CREATE LOGISTIC REGRESSION MODEL
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("Training Logistic Regression Model...")

model.fit(X_train, y_train)

print("Training Completed!")


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred, zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred, zero_division=0))

print("F1 Score:",
      f1_score(y_test, y_pred, zero_division=0))

print("ROC-AUC Score:",
      roc_auc_score(y_test, y_probability))


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# ==========================================
# 13. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_regression.pkl")

print("\nModel saved successfully!")
