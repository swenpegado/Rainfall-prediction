from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = 'xgboost_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html', prediction="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract essential data from the form
        temp = float(request.form['temp'])
        humidity = 55
        pressure = 1000
        wind_speed = float(request.form['wind_speed'])
        previous_rainfall =5
        humidity_index = 1
        temperature_variation = float(request.form['temperature_variation'])
        cloud_cover = float(request.form['cloud_cover'])

        # Validate input ranges
        if not (0 <= humidity <= 100):
            return render_template('index.html', prediction="Humidity must be between 0 and 100.")
        if not (-10 <= temp <= 50):
            return render_template('index.html', prediction="Temperature must be between -10 and 50°C.")
        if not (950 <= pressure <= 1050):
            return render_template('index.html', prediction="Pressure must be between 950 and 1050 hPa.")
        if wind_speed < 0 or previous_rainfall < 0 or humidity_index < 0 or temperature_variation < 0:
            return render_template('index.html', prediction="Invalid input. No negative values allowed.")
        if not (0 <= cloud_cover <= 100):
            return render_template('index.html', prediction="Cloud cover must be between 0 and 100.")

        # Prepare the input data (fill other 5 missing features with default values)
        input_data = np.array([[temp, humidity, default_pressure, default_wind_speed, 
                                default_previous_rainfall, default_humidity_index, 
                                default_temperature_variation, cloud_cover]])

        # Make prediction (returns probability of rainfall)
        prediction_prob = model.predict_proba(input_data)[0][1]  # Get probability for 'rain' class (1)

        # Convert the result to a percentage
        prediction_percentage = prediction_prob * 100

        # Return result to the user
        result = f"There is a {prediction_percentage:.2f}% chance of rain."
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
