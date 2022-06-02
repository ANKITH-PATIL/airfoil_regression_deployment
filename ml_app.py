import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open("C:/Users/Ankith/airfoil_model.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict1():
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]

    print(data)

    new_data=[np.array(data)]
    output=model.predict(new_data)
    print (output)
    return render_template('home.html',prediction_text="Airfoil Pressure is {}".format(output))


if __name__=='__main__':
    app.run(debug=True)
