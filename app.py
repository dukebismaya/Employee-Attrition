from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load trained model and metrics
model = pickle.load(open('model.pkl', 'rb'))
metrics = pickle.load(open('metrics.pkl', 'rb')) if os.path.exists('metrics.pkl') else {}

# Feature options for dropdowns
FEATURE_OPTIONS = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
    'Gender': ['Male', 'Female'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 
                'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['No', 'Yes']
}

# All feature columns in order (must match training)
FEATURE_COLUMNS = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]


@app.route('/')
def home():
    return render_template('index.html', options=FEATURE_OPTIONS, metrics=metrics)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Build feature dictionary from form
        data = {}
        for col in FEATURE_COLUMNS:
            value = request.form.get(col)
            if col in FEATURE_OPTIONS:
                data[col] = value
            else:
                data[col] = int(value) if value else 0
        
        # Create DataFrame matching training format
        df = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        result = {
            'prediction': 'LEAVE' if prediction == 1 else 'STAY',
            'confidence': round(max(probability) * 100, 1),
            'stay_prob': round(probability[0] * 100, 1),
            'leave_prob': round(probability[1] * 100, 1)
        }
        
        return render_template('result.html', result=result, data=data)
    
    except Exception as e:
        return render_template('result.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions"""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        prediction = int(model.predict(df)[0])
        probability = model.predict_proba(df)[0]
        
        return jsonify({
            'success': True,
            'prediction': 'LEAVE' if prediction == 1 else 'STAY',
            'stay_probability': round(float(probability[0]) * 100, 1),
            'leave_probability': round(float(probability[1]) * 100, 1)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/metrics')
def show_metrics():
    """Display model performance metrics"""
    return render_template('metrics.html', metrics=metrics)


if __name__ == '__main__':
    app.run(debug=True)