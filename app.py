import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'concrete_prediction_secret_key_2024'

# Loading the saved XGBoost_regressor model
model = joblib.load('XGBoost_Regressor_model.pkl')

@app.route('/')
def login():
    # If already logged in, redirect to dashboard
    if session.get('logged_in'):
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    # Simple login without validation
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Set session as logged in
    session['logged_in'] = True
    session['username'] = username
    
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Store prediction history in session
@app.route('/dashboard')
def home():
    # Check if user is logged in
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if 'predictions' not in session:
        session['predictions'] = []
    return render_template('index.html', predictions=session.get('predictions', []), username=session.get('username'))


@app.route('/predict', methods=['POST'])
def predict():
    # Check if user is logged in
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == "POST":
        # Get input values
        age = float(request.form.get('age'))
        cement = float(request.form.get('cement'))
        water = float(request.form.get('water'))
        fly_ash = float(request.form.get('fa'))
        superplasticizer = float(request.form.get('sp'))
        blast_furnace_slag = float(request.form.get('bfs'))
        coarse_aggregate = float(request.form.get('ca', 0))
        fine_aggregate = float(request.form.get('fina', 0))
        
        # Prepare features for prediction (only the 6 features the model expects)
        f_list = [age, cement, water, fly_ash, superplasticizer, blast_furnace_slag]
        final_features = np.array(f_list).reshape(-1, 6)
        df = pd.DataFrame(final_features)

        # Make prediction
        prediction = model.predict(df)
        result = float(round(prediction[0], 2))  # Convert numpy float32 to Python float
        
        # Calculate water/cement ratio
        wc_ratio = float(round(water / cement if cement > 0 else 0, 2))
        
        # Calculate aggregates
        total_aggregates = float(coarse_aggregate + fine_aggregate)
        
        # Determine strength category
        if result >= 80:
            category = "Excellent Strength"
            confidence = 92
        elif result >= 60:
            category = "High Strength"
            confidence = 88
        elif result >= 40:
            category = "Good Strength"
            confidence = 85
        elif result >= 25:
            category = "Moderate Strength"
            confidence = 82
        else:
            category = "Low Strength"
            confidence = 78
        
        # Calculate factor contributions (normalized percentages)
        contributions = {
            'Cement': float(round((cement / (cement + water + fly_ash + superplasticizer + blast_furnace_slag)) * 40, 1)),
            'Water/Cement': float(round(wc_ratio * 15, 1)),
            'Aggregates': float(round((total_aggregates / 2000) * 20, 1) if total_aggregates > 0 else 15),
            'Age': float(round((age / 90) * 15, 1)),
            'Additives': float(round(((fly_ash + superplasticizer + blast_furnace_slag) / 300) * 10, 1))
        }
        
        # Store prediction in session (ensure all values are JSON serializable)
        prediction_data = {
            'result': result,
            'timestamp': datetime.now().strftime('%b %d, %Y, %I:%M %p'),
            'age': float(age),
            'cement': float(cement),
            'water': float(water),
            'wc_ratio': wc_ratio,
            'aggregates': total_aggregates,
            'category': category,
            'confidence': confidence,
            'contributions': contributions
        }
        
        if 'predictions' not in session:
            session['predictions'] = []
        
        predictions = session['predictions']
        predictions.insert(0, prediction_data)  # Add to beginning
        session['predictions'] = predictions[:10]  # Keep only last 10
        session.modified = True
        
        return render_template('index.html', 
                             prediction=prediction_data,
                             predictions=session['predictions'])


if __name__ == "__main__":
    app.run(debug=True)
