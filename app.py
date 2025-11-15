from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from heart_disease_model import HeartDiseasePredictor

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Initialize predictor
predictor = HeartDiseasePredictor()

# Ensure model is available at startup. If model files are missing, train and save a new model.
if not predictor.load_model():
    try:
        print("No pre-trained model found. Training a new model now...")
        predictor.train_model()
        predictor.save_model()
    except Exception as e:
        # If training fails at startup, print the error but allow the app to run
        print(f"Warning: failed to train model at startup: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            chest_pain_type = int(request.form['chest_pain_type'])
            resting_bp = int(request.form['resting_bp'])
            cholesterol = int(request.form['cholesterol'])
            fasting_bs = int(request.form['fasting_bs'])
            resting_ecg = int(request.form['resting_ecg'])
            max_hr = int(request.form['max_hr'])
            exercise_angina = int(request.form['exercise_angina'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            
            # Prepare input data
            input_data = [
                age, sex, chest_pain_type, resting_bp, cholesterol,
                fasting_bs, resting_ecg, max_hr, exercise_angina,
                oldpeak, slope
            ]
            
            # Make prediction
            prediction, probability = predictor.predict(input_data)
            
            # Prepare result
            result = {
                'prediction': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
                'probability': round(probability * 100, 2),
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
                'input_data': {
                    'Age': age,
                    'Sex': 'Male' if sex == 1 else 'Female',
                    'Chest Pain Type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][chest_pain_type],
                    'Resting BP': f"{resting_bp} mmHg",
                    'Cholesterol': f"{cholesterol} mg/dl",
                    'Fasting Blood Sugar': '> 120 mg/dl' if fasting_bs == 1 else '<= 120 mg/dl',
                    'Resting ECG': ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'][resting_ecg],
                    'Max Heart Rate': max_hr,
                    'Exercise Angina': 'Yes' if exercise_angina == 1 else 'No',
                    'ST Depression': oldpeak,
                    'Slope': ['Upsloping', 'Flat', 'Downsloping'][slope]
                }
            }
            
            # include model accuracy (if available)
            model_accuracy = None
            if getattr(predictor, 'last_accuracy', None) is not None:
                model_accuracy = round(predictor.last_accuracy * 100, 2)

            return render_template('predict.html', result=result, show_result=True, model_accuracy=model_accuracy)
            
        except Exception as e:
            error = f"Error processing your request: {str(e)}"
            model_accuracy = None
            if getattr(predictor, 'last_accuracy', None) is not None:
                model_accuracy = round(predictor.last_accuracy * 100, 2)
            return render_template('predict.html', error=error, show_result=False, model_accuracy=model_accuracy)
    
    model_accuracy = None
    if getattr(predictor, 'last_accuracy', None) is not None:
        model_accuracy = round(predictor.last_accuracy * 100, 2)
    return render_template('predict.html', show_result=False, model_accuracy=model_accuracy)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        input_data = [
            data['age'], data['sex'], data['chest_pain_type'],
            data['resting_bp'], data['cholesterol'], data['fasting_bs'],
            data['resting_ecg'], data['max_hr'], data['exercise_angina'],
            data['oldpeak'], data['slope']
        ]
        
        prediction, probability = predictor.predict(input_data)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability * 100, 2),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use environment variables to control production vs development behavior.
    # - Set FLASK_DEBUG=1 to enable the Flask debugger (only for development).
    # - Set USE_WAITRESS=1 to run with the Waitress WSGI server (recommended on Windows).
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    use_waitress = os.environ.get('USE_WAITRESS', '0') == '1'
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5000'))

    if use_waitress:
        # Waitress is a production-ready WSGI server that works well on Windows.
        try:
            from waitress import serve
        except Exception:
            raise RuntimeError('Waitress is not installed. Please install it with `pip install waitress`')

        print('Starting with Waitress on %s:%s' % (host, port))
        serve(app, host=host, port=port)
    else:
        # Default: run Flask development server. Make sure debug is False in production.
        app.run(debug=debug, host=host, port=port)