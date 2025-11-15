import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.last_accuracy = None
        self.feature_names = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 
            'cholesterol', 'fasting_bs', 'resting_ecg', 
            'max_hr', 'exercise_angina', 'oldpeak', 'slope'
        ]
    
    def create_sample_data(self):
        """Create sample heart disease dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(29, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'chest_pain_type': np.random.randint(0, 4, n_samples),
            'resting_bp': np.random.randint(90, 200, n_samples),
            'cholesterol': np.random.randint(100, 400, n_samples),
            'fasting_bs': np.random.randint(0, 2, n_samples),
            'resting_ecg': np.random.randint(0, 3, n_samples),
            'max_hr': np.random.randint(60, 210, n_samples),
            'exercise_angina': np.random.randint(0, 2, n_samples),
            'oldpeak': np.round(np.random.uniform(0, 6, n_samples), 1),
            'slope': np.random.randint(0, 3, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on realistic patterns
        heart_disease_prob = (
            df['age'] * 0.05 +
            df['chest_pain_type'] * 0.3 +
            (df['resting_bp'] - 120) * 0.02 +
            (df['cholesterol'] - 200) * 0.01 +
            df['exercise_angina'] * 0.4 +
            df['oldpeak'] * 0.3 +
            np.random.normal(0, 0.5, n_samples)
        )
        
        df['target'] = (heart_disease_prob > heart_disease_prob.mean()).astype(int)
        
        return df
    
    def train_model(self):
        """Train the logistic regression model"""
        print("Creating and training heart disease prediction model...")
        
        # Create sample dataset
        df = self.create_sample_data()
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # store last training accuracy
        self.last_accuracy = float(accuracy)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def save_model(self):
        """Save the trained model and scaler"""
        if not os.path.exists('model'):
            os.makedirs('model')
            
        with open('model/heart_disease_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # save metrics (accuracy) for display
        try:
            metrics = {'accuracy': None if self.last_accuracy is None else float(self.last_accuracy)}
            with open('model/metrics.json', 'w') as mf:
                import json
                json.dump(metrics, mf)
        except Exception:
            pass
        
        print("Model and scaler saved successfully!")
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            with open('model/heart_disease_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('model/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            # try to load metrics
            try:
                import json
                with open('model/metrics.json', 'r') as mf:
                    metrics = json.load(mf)
                    if 'accuracy' in metrics and metrics['accuracy'] is not None:
                        self.last_accuracy = float(metrics['accuracy'])
            except Exception:
                # ignore missing/invalid metrics
                pass
            
            print("Model and scaler loaded successfully!")
            return True
        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
            return False
    
    def predict(self, input_data):
        """Make prediction on new data"""
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Model is not loaded. Train or provide model files before prediction.")
        
        # Convert input to numpy array and scale
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0][1]
        
        return prediction, probability

# Create and train the model if running directly
if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    predictor.train_model()
    predictor.save_model()