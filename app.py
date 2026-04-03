from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model (replace with your actual model file)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(request.form.get(f)) for f in ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'JobLevel']]
    # Add categorical features as needed
    # Example: BusinessTravel, Gender, OverTime, MaritalStatus (encode as in your model)
    # For demo, use dummy values
    features += [0, 1, 0, 2]  # Replace with actual encoding logic

    # Scale features
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    result = f"Employee will {'LEAVE' if prediction == 1 else 'STAY'} (Probability: {prob:.2f})"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)