
import pandas as pd

# Import Scikit-learn tools
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv(
    "data/archive (2)/healthcare-dataset-stroke-data.csv"
)

print("Dataset Loaded Successfully!")
print("Original Dataset Shape:", data.shape)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================

data = data.drop("id", axis=1)


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop("stroke", axis=1)

y = data["stroke"]

print("\nInput Features Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ==========================================
# 5. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 6. NUMERICAL PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==========================================
# 7. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 8. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# ==========================================
# 9. FIT AND TRANSFORM TRAINING DATA
# ==========================================

X_train_processed = preprocessor.fit_transform(X_train)


# ==========================================
# 10. TRANSFORM TESTING DATA
# ==========================================

X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n========== PREPROCESSING RESULTS ==========")

print("Original Training Shape:", X_train.shape)

print("Processed Training Shape:", X_train_processed.shape)

print("Original Testing Shape:", X_test.shape)

print("Processed Testing Shape:", X_test_processed.shape)


print("\nMissing Values After Processing:")

print("Training:", pd.isna(X_train_processed).sum())

print("Testing:", pd.isna(X_test_processed).sum())


print("\nStroke Distribution in Training Data:")
print(y_train.value_counts())

print("\nStroke Distribution in Testing Data:")
print(y_test.value_counts())


print("\nPreprocessing Completed Successfully!")

