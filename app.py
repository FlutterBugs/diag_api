from flask import Flask, jsonify, request
import pickle
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "<center><h1> This is the Home Route </h1></center>"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    lr = joblib.load("diagnosis_model.pkl")
    if lr:
        try:
            radius_mean = request.form['radius_mean']
            texture_mean = request.form['texture_mean']
            perimeter_mean = request.form['perimeter_mean']
            area_mean = request.form['area_mean']
            smoothness_mean = request.form['smoothness_mean']
            compactness_mean = request.form['compactness_mean']
            concavity_mean = request.form['concavity_mean']
            concave_points_mean = request.form['concave points_mean']
            symmetry_mean = request.form['symmetry_mean']
            fractal_dimension_mean = request.form['fractal_dimension_mean']
            radius_se = request.form['radius_se']
            texture_se = request.form['texture_se']
            perimeter_se = request.form['perimeter_se']
            area_se = request.form['area_se']
            smoothness_se = request.form['smoothness_se']
            compactness_se = request.form['compactness_se']
            concavity_se = request.form['concavity_se']
            concave_points_se = request.form['concave points_se']
            symmetry_se = request.form['symmetry_se']
            fractal_dimension_se = request.form['fractal_dimension_se']
            radius_worst = request.form['radius_worst']
            texture_worst = request.form['texture_worst']
            perimeter_worst = request.form['perimeter_worst']
            area_worst = request.form['area_worst']
            smoothness_worst = request.form['smoothness_worst']
            compactness_worst = request.form['compactness_worst']
            concavity_worst = request.form['concavity_worst']
            concave_points_worst = request.form['concave points_worst']
            symmetry_worst = request.form['symmetry_worst']
            fractal_dimension_worst = request.form['fractal_dimension_worst']
            
            data = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]]
            prediction_data = pd.DataFrame(data, columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'])
            prediction = lr.predict(prediction_data)
            
            return jsonify(prediction[0])
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')
    

if __name__ == '__main__':
    app.run(debug=True)
