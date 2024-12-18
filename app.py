from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    age = float(request.form.get('age', 0))
    sex = float(request.form.get('sex', 0))
    cp = float(request.form.get('cp', 0))
    trestbps = float(request.form.get('trestbps', 0))
    chol = float(request.form.get('chol', 0))
    fbs = float(request.form.get('fbs', 0))
    restecg = float(request.form.get('restecg', 0))
    thalach = float(request.form.get('thalach', 0))
    exang = float(request.form.get('exang', 0))
    oldpeak = float(request.form.get('oldpeak', 0))
    slope = float(request.form.get('slope', 0))
    ca = float(request.form.get('ca', 0))
    thal = float(request.form.get('thal', 0))

    # Prepare input for prediction
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)
    
    # Generate prediction result
    if prediction[0] == 1:
        prediction_text = "Heart Disease Detected!"
    else:
        prediction_text = "No Heart Disease Detected!"

    return render_template('index.html',
                           age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, 
                           restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak, 
                           slope=slope, ca=ca, thal=thal, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
