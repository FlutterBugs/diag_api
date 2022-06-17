from flask import Flask, jsonify, request
import pickle
import joblib
import numpy as np
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
            json = request.get_json()  
            model_columns = joblib.load("diagnosis_cols.pkl")
            temp=list(json[0].values())
            vals=np.array(temp)
            prediction = lr.predict(temp)
            print("here:",prediction)        
            return jsonify({'prediction': str(prediction[0])})
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')

if __name__ == '__main__':
    app.run(debug=True)