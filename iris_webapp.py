
from flask import Flask, request
from flask import render_template
import pickle
import pandas as pd 
import numpy as np
from sklearn.svm import SVC as svc
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR

Algorithm = "SVM" or "Decision Tree" or "Logistic Regression"

app = Flask(__name__, template_folder='template')
svc = pickle.load(open('svc_model.pkl','rb'))
DTC = pickle.load(open('LR_model.pkl','rb'))
LR = pickle.load(open('DTC_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    sepallength = request.form.get('Sepal length')
    sepalwidth = request.form.get('Sepal Width')
    petallength = request.form.get('Petal Length')
    petalwidth = request.form.get('Petal Width')
    arr = np.array([[sepallength, sepalwidth, petallength,
       petalwidth ]]) 
    if Algorithm == "SVM":
       pred = svc.predict(arr)
       return render_template('index.html', prediction_text =' The Predicted class of the flower is {}'.format(pred))
    elif Algorithm == "Logistic Regression":
        pred = LR.predict(arr)
        return render_template('index.html', prediction_text =' The Predicted class of the flower is {}'.format(pred))
    else:
        pred = DTC.predict(arr)
        return render_template('index.html', prediction_text =' The Predicted class of the flower is {}'.format(pred))
    
    
if __name__=='__main__':
    app.run(debug= True)