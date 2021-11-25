import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    feature_value = [np.array(input_feature)]

    feature_name = ['AGE', 'SEXE', 'Doul_Thorax', 'P_Arterielle', 'CHOLESTEROL','Glycemie_AJ', 'ECG', 'Freq_CardMax', 'ANGINE', 'Depression', 'PENTE']

    df = pd.DataFrame(feature_value, colunms=feature_name)
    output = model.predict(df)

    if output == 1:
        res_val = "sain"
    else :
        res_val = "malade"

    return render_template('index.html', prediction_text="Le patient est {} du coeur".format(res_val))

if __name__== "__main__":
    app.run()