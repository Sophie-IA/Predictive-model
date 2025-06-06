from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('churn_model.pkl')

#@app.route('/features')
feature_info = [
    {"name": "Gender", "description": "Gender (0=Female, 1=Male)"},
    {"name": "SeniorCitizen", "description": "Senior Citizen (0=No, 1=Yes)"},
    {"name": "Partner", "description": "Partner Status (0=No, 1=Yes)"},
    {"name": "Dependents", "description": "Dependents Status (0=No, 1=Yes)"},
    {"name": "Tenure", "description": "Tenure (in months 0-72)"},
    {"name": "PhoneService", "description": "Phone Service (0=No, 1=Yes)"},
    {"name": "MultipleLines", "description": "Multiple Lines (0=No, 1=Yes, 2=No Phone Service)"},
    {"name": "InternetService", "description": "Internet Service Type (0=DSL, 1=Fiber Optic, 2=None)"},
    {"name": "OnlineSecurity", "description": "Online Security (0=No, 1=Yes, 2=No Internet Service)"},
    {"name": "OnlineBackup", "description": "Online Backup (0=No, 1=Yes, 2=No Internet Service)"},
    {"name": "DeviceProtection", "description": "Device Protection (0=No, 1=Yes, 2=No Internet Service)"},
    {"name": "TechSupport", "description": "Tech Support (0=No, 1=Yes, 2=No Internet Service)"},
    {"name": "StreamingTV", "description": "Streaming TV (0=No, 1=Yes, 2=No Internet Service)"},
    {"name": "StreamingMovies", "description": "Streaming Movies (0=No, 1=Yes, 2=No Internet Service)"},
    {"name": "Contract", "description": "Contract Type (0=Month-to-Month, 1=One Year, 2=Two Year)"},
    {"name": "PaperlessBilling", "description": "Paperless Billing (0=No, 1=Yes)"},
    {"name": "PaymentMethod", "description": "Payment Method (0=Electronic Check, 1=Mailed Check, 2=Bank Transfer, 3=Credit Card)"},
    {"name": "MonthlyCharges", "description": "Monthly Charges (18.25 - 118.75)"},
    {"name": "TotalCharges", "description": "Total Charges (18.8 - 8684.8)"},
    
    # ... add all feature names in correct order
]

@app.route('/')
def home():
    return render_template('form.html', feature_info=feature_info)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])[0]
    try:
        probability = model.predict_proba([data])[0][1]
    except AttributeError:
        probability = None
    return jsonify({'prediction': int(prediction), 'probability': float(probability) if probability is not None else None})


if __name__ == '__main__':
    app.run(debug=True)


