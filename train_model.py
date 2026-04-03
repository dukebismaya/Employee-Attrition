import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

print("===== STARTING PROJECT =====")

# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print(" Dataset Loaded Successfully")
except Exception as e:
    print(" Error loading dataset:", e)
    exit()

# =========================
# DATA CLEANING
# =========================
df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Features & target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# =========================
# COLUMN TYPES
# =========================
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

print(f"Categorical Columns: {len(cat_cols)}")
print(f"Numerical Columns: {len(num_cols)}")

# =========================
# PREPROCESSOR
# =========================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(" Data Split Completed")

# =========================
# PIPELINE
# =========================
model = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=5,
        use_label_encoder=False
    ))
])

# =========================
# TRAIN MODEL
# =========================
print(" Training Model...")
model.fit(X_train, y_train)
print(" Training Completed")

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_pred))
}

print("\n===== MODEL PERFORMANCE =====")
for k, v in metrics.items():
    print(f"{k.upper()}: {v:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

# Create folder if not exists
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/confusion.png")
plt.close()

print(" Confusion matrix saved in 'static/confusion.png'")

# =========================
# SAVE MODEL & METRICS
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(metrics, open("metrics.pkl", "wb"))

print(" Model saved as 'model.pkl'")
print(" Metrics saved as 'metrics.pkl'")

print("===== PROJECT COMPLETED SUCCESSFULLY =====")