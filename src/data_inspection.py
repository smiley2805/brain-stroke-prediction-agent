
import pandas as pd

# Load the dataset
data = pd.read_csv("data/archive (2)/healthcare-dataset-stroke-data.csv")

# Display first 5 rows
print("First 5 patient records:")
print(data.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nColumn Names:")
print(data.columns)

# Display information about the dataset
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check stroke distribution
print("\nStroke Distribution:")
print(data["stroke"].value_counts())

# Calculate stroke distribution
stroke_counts = data["stroke"].value_counts()

print("\nStroke Counts:")
print(stroke_counts)

# Calculate percentages
print("\nStroke Percentages:")
print(data["stroke"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Stroke distribution

plt.figure(figsize=(6, 4))

sns.countplot(data=data, x="stroke")

plt.title("Stroke Distribution")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.show()


# Graph 2: Age distribution

plt.figure(figsize=(8, 4))

sns.histplot(data=data, x="age", bins=30, kde=True)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.show()

# Age comparison between stroke classes

plt.figure(figsize=(8, 5))

sns.boxplot(data=data, x="stroke", y="age")

plt.title("Age vs Stroke")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.show()

# Glucose level comparison

plt.figure(figsize=(8, 5))

sns.boxplot(data=data, x="stroke", y="avg_glucose_level")

plt.title("Average Glucose Level vs Stroke")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Average Glucose Level")

plt.show()

# Hypertension and stroke relationship

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="hypertension", hue="stroke")

plt.title("Hypertension vs Stroke")
plt.xlabel("Hypertension (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")

plt.show()
