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

app = Flask(__name__, template_folder='templates')

model_RF = pickle.load(open('RandomForestRegression.pkl', 'rb'))
model_LIN = pickle.load(open('LinearRegression.pkl', 'rb'))
model_RDG = pickle.load(open('RidgeRegression.pkl', 'rb'))
model_LAS = pickle.load(open('LassoRegression.pkl', 'rb'))

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
        age_of_car = Current_Year - Year
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

        prediction_RF = model_RF.predict([[Present_Price, Kms_Driven, Owner, age_of_car, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                     Seller_Type_Individual, Transmission_Manual]])

        prediction_LIN = model_LIN.predict([[Present_Price, Kms_Driven, Owner, age_of_car, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                           Seller_Type_Individual, Transmission_Manual]])

        prediction_RDG = model_RDG.predict([[Present_Price, Kms_Driven, Owner, age_of_car, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                             Seller_Type_Individual, Transmission_Manual]])

        prediction_LAS = model_LAS.predict([[Present_Price, Kms_Driven, Owner, age_of_car, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                             Seller_Type_Individual, Transmission_Manual]])

        output_RF = round(prediction_RF[0], 2)
        output_LIN = round(prediction_LIN[0], 2)
        output_RDG = round(prediction_RDG[0], 2)
        output_LAS = round(prediction_LAS[0], 2)

        predictions = 'PREDICTIONS RandomForest: ' + str(output_RF) + ' Linear Regression: ' + str(output_LIN) + \
                      ' Ridge Regression: ' + str(output_RDG) + ' Lasso Regression: ' + str(output_LAS)

        return render_template('index.html', prediction_text=predictions)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

