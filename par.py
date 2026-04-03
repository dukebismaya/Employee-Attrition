# ============================================
# EMPLOYEE ATTRITION PREDICTION 
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# 1. LOAD DATA
# ============================================
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
# ============================================
# 2. CLEAN DATA
# ============================================
df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# ============================================
# 3. ENCODING
# ============================================
categorical_cols = df.select_dtypes(include=['object']).columns

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ============================================
# 4. SPLIT
# ============================================
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 5. SCALING
# ============================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================
# 6. MODEL (IMPROVED)
# ============================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# 7. EVALUATION
# ============================================
y_pred = model.predict(X_test)

print("\nMODEL PERFORMANCE")
print("="*40)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ============================================
# 8. SIMPLIFIED INPUT SYSTEM
# ============================================
def get_user_input():
    print("\nEnter Employee Details:\n")

    data = {}

    try:
        data['Age'] = int(input("Age: "))
        data['DailyRate'] = int(input("Daily Rate: "))
        data['DistanceFromHome'] = int(input("Distance From Home: "))
        data['MonthlyIncome'] = int(input("Monthly Income: "))
        data['TotalWorkingYears'] = int(input("Total Working Years: "))
        data['YearsAtCompany'] = int(input("Years At Company: "))
        data['JobLevel'] = int(input("Job Level (1-5): "))

        data['BusinessTravel'] = input("Business Travel (Travel_Rarely/Travel_Frequently/Non-Travel): ")
        data['Gender'] = input("Gender (Male/Female): ")
        data['OverTime'] = input("OverTime (Yes/No): ")
        data['MaritalStatus'] = input("Marital Status (Single/Married/Divorced): ")

    except:
        print("Invalid input! Try again.")
        return None

    return data

# ============================================
# 9. PREPROCESS USER INPUT
# ============================================
def preprocess_input(data):
    user_df = pd.DataFrame([data])

    # Fill missing columns with median/default
    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = df[col].median()

    # Ensure correct order
    user_df = user_df[X.columns]

    # Encode categorical
    for col in categorical_cols:
        if col in user_df.columns:
            try:
                user_df[col] = le_dict[col].transform(user_df[col])
            except:
                user_df[col] = 0  # fallback

    # Scale
    user_scaled = scaler.transform(user_df)

    return user_scaled

# ============================================
# 10. PREDICTION
# ============================================
def predict():
    data = get_user_input()
    if data is None:
        return

    processed = preprocess_input(data)

    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]

    print("\nPrediction Result:")
    if pred == 1:
        print(f"Employee will LEAVE (Probability: {prob:.2f})")
    else:
        print(f"Employee will STAY (Probability: {1-prob:.2f})")

# ============================================
# 11. RUN LOOP
# ============================================
while True:
    ch = input("\nDo prediction? (yes/no): ").lower()
    if ch == 'yes':
        predict()
    else:
        print("Exit")
        break