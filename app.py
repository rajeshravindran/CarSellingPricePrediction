from flask import Flask, render_template, request
from datetime import datetime

import jsonify
import requests
import pickle
import numpy as np
import sklearn
import os

from sklearn.preprocessing import StandardScaler

Current_Year = datetime.now().year

app = Flask(__name__)

model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

standard_to = StandardScaler()
@app.route('/predict', methods=['POST'])
def predict():
    Fuel_Type_Diesel = 0
    Fuel_Type_Petrol = 0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        Fuel_Type = request.form['Fuel_Type']
        if Fuel_Type.lower() == 'petrol':
            Fuel_Type_Diesel = 0
            Fuel_Type_Petrol = 1
        elif Fuel_Type.lower() == 'diesel':
            Fuel_Type_Diesel = 1
            Fuel_Type_Petrol = 0
        else:
            Fuel_Type_Diesel = 0
            Fuel_Type_Petrol = 0

        Seller_Type = request.form['Seller_Type']
        if Seller_Type.lower()=='individual':
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0

        Transmission = request.form['Transmission']
        if Transmission.lower() == 'manual':
            Transmission_Manual = 1
        else:
            Transmission_Manual = 0

        prediction = model.predict([[Present_Price, Kms_Driven, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                     Seller_Type_Individual, Transmission_Manual]])
        output = round(prediction[0], 2)
        if output < 0:
            return render_template('index.html', prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html', prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

