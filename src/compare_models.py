
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv(
    "data/archive (2)/healthcare-dataset-stroke-data.csv"
)

data = data.drop("id", axis=1)

X = data.drop("stroke", axis=1)
y = data["stroke"]

# --------------------------------------------------
# 2. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Load trained models
# --------------------------------------------------

models = {
    "Logistic Regression": joblib.load(
        "models/logistic_regression.pkl"
    ),
    "Random Forest": joblib.load(
        "models/random_forest.pkl"
    ),
    "XGBoost": joblib.load(
        "models/xgboost.pkl"
    ),
    "SVM": joblib.load(
        "models/svm.pkl"
    ),
    "KNN": joblib.load(
        "models/knn.pkl"
    ),
    "MLP": joblib.load(
        "models/mlp.pkl"
    )
}

# --------------------------------------------------
# 4. Evaluate models
# --------------------------------------------------

results = []

for name, model in models.items():

    print(f"Evaluating {name}...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

# --------------------------------------------------
# 5. Create comparison DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# --------------------------------------------------
# 6. Find best model
# --------------------------------------------------

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n========== BEST MODEL ==========")
print(best_model.to_string())

# --------------------------------------------------
# 7. Create results folder
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# 8. Save comparison as CSV
# --------------------------------------------------

csv_path = "results/model_comparison.csv"

results_df.to_csv(csv_path, index=False)

print(f"\nComparison CSV saved at: {csv_path}")

# --------------------------------------------------
# 9. Generate comparison chart
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

plot_data = results_df.set_index("Model")[metrics]

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.8
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.legend(title="Evaluation Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

chart_path = "results/model_comparison.png"

plt.savefig(chart_path, dpi=300)
plt.close()

print(f"Comparison chart saved at: {chart_path}")

print("\nModel comparison completed successfully!")
