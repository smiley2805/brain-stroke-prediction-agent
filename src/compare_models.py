
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 1. Load dataset
data = pd.read_csv(
    "data/archive (2)/healthcare-dataset-stroke-data.csv"
)

data = data.drop("id", axis=1)

# 2. Separate features and target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# 3. Use the same train-test split as before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Load trained models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "MLP": joblib.load("models/mlp.pkl")
}

# 5. Store evaluation results
results = []

for name, model in models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

# 6. Create comparison table
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# 7. Find the model with the highest F1 score
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBest model based on F1 Score:")
print(best_model)
