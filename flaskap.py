# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:24:29 2020

@author: srikanth
"""

from flask import Flask,request
import pickle
import pandas as pd
import numpy as np
import flasgger
from flasgger import Swagger


app=Flask(__name__)
Swagger(app)

pickle_in=open("rf.pkl",'rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return"welcome all"
    
@app.route('/predict')
def predict_diab():
    
    """Diabetis Prediction Engine 
    This is using certain health related inputs.
    ---
    parameters:  
      - name: Pregnancies
        in: query
        type: number
        required: true
      - name: Glucose
        in: query
        type: number
        required: true
      - name: BloodPressure
        in: query
        type: number
        required: true
      - name: SkinThickness
        in: query
        type: number
        required: true
      - name: Insulin
        in: query
        type: number
        required: true
      - name: BMI
        in: query
        type: number
        required: true
      - name: DiabetesPedigreeFunction
        in: query
        type: number
        required: true
      - name: Age
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """    
    
    
    Pregnancies=int(request.args.get("Pregnancies"))
    Glucose=int(request.args.get("Glucose"))
    BloodPressure=int(request.args.get("BloodPressure"))
    SkinThickness=int(request.args.get("SkinThickness"))
    Insulin=int(request.args.get("Insulin"))
    BMI=float(request.args.get("BMI"))
    DiabetesPedigreeFunction=float(request.args.get("DiabetesPedigreeFunction"))
    Age=int(request.args.get("Age"))
    prediction=classifier.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI,DiabetesPedigreeFunction, Age]])
    return "the outcome of diabetes test is :"+str(prediction)


@app.route('/predictfile',methods=["POST"])
def predict_diabfile():
    """Diabetis Prediction Engine 
    This is using certain health related inputs.
    ---
    parameters:  
      - name: Pregnancies
        in: file
        type: file
        required: true
     
    responses:
        200:
            description: The output values
        
    """   
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "the outcome of diabetes file test are :"+str(list(prediction))
            

if __name__=='__main__':
    app.run()

