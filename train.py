# ==========================================
# Cardiovascular Risk Assessment
# Random Forest Training
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/synthetic_heart_disease_dataset.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.head())
print("\nShape:", df.shape)

# ==========================================
# Handle Missing Values
# ==========================================

for col in df.columns:

    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# Encode ALL Categorical Columns
# ==========================================

encoders = {}

for col in df.columns:

    # Encode every non-numeric column
    if not pd.api.types.is_numeric_dtype(df[col]):

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

# ==========================================
# Target and Features
# ==========================================

TARGET = "Heart_Disease"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==========================================
# Check Data Types
# ==========================================

print("\nData Types After Encoding:")
print(X.dtypes)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ==========================================
# Train Random Forest
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

print("\n✅ Random Forest Trained Successfully!")

# ==========================================
# Prediction
# ==========================================

y_pred = rf.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(10, 8))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# ==========================================
# Save Model
# ==========================================

joblib.dump(rf, "heart_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\n✅ Files Saved Successfully!")
print("heart_model.pkl")
print("encoders.pkl")
print("feature_names.pkl")

print("\nDataset Columns:")
print(df.columns)