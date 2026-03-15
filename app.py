from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "model.sav")
regressor = joblib.load(model_path)

# -----------------------------
# Load Datasets
# -----------------------------
dataset = pd.read_csv('Final_Dataset.csv')
dataset2 = pd.read_csv('Trainset.csv')

dataset2.drop('ElectricalConductivity(ds/m)', axis=1, inplace=True)

X = dataset.loc[:, dataset.columns != 'Production']
X = X.drop('Unnamed: 0', axis=1)

y = dataset['Production']

columns = list(X.columns)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(season, crop, area, rainfall, temperature, pH, nitrogen):

    # fresh feature vector each time
    features = [0] * len(columns)

    features[columns.index('pH')] = pH
    features[columns.index('Nitrogen(kg/ha)')] = nitrogen
    features[columns.index('Area')] = area
    features[columns.index('Rainfall')] = rainfall
    features[columns.index('Temperature')] = temperature

    # one-hot encoding
    if season in columns:
        features[columns.index(season)] = 1

    if crop in columns:
        features[columns.index(crop)] = 1

    features = [features]

    prediction = regressor.predict(features)

    return int(prediction[0])

# -----------------------------
# Graph Generation
# -----------------------------
def generateGraph(season, crop, area, rainfall, temperature, pH, nitrogen):

    plt.close('all')

    df = dataset2.loc[dataset2['Season'] == season]
    crops = df['Crop'].unique()

    crop_names = []
    predictions = []

    for c in crops:

        if (c != 'Sugarcane' and c != crop and c != 'Potato') or season == 'Whole Year':

            crop_names.append(c)

            predictions.append(
                predict(season, c, area, rainfall, temperature, pH, nitrogen)
            )

    return crop_names, predictions

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def result():

    try:

        season = request.form['season']
        crop = request.form['crop']

        area_input = float(request.form['area'])

        area = area_input / dataset2['Area'].max()
        rainfall = float(request.form['rainfall']) / dataset2['Rainfall'].max()
        temperature = float(request.form['temperature']) / dataset2['Temperature'].max()
        pH = float(request.form['pH']) / dataset2['pH'].max()
        nitrogen = float(request.form['nitrogen']) / dataset2['Nitrogen(kg/ha)'].max()

        prediction = predict(season, crop, area, rainfall, temperature, pH, nitrogen)

        if area_input == 0:
            area_input = 1

        prediction = prediction / area_input

        crop_list, pred_list = generateGraph(
            season,
            crop,
            area,
            rainfall,
            temperature,
            pH,
            nitrogen
        )

        top_pred, top_crop = (list(t) for t in zip(*sorted(zip(pred_list, crop_list))))

        if len(top_pred) >= 3:
            top_pred = top_pred[-3:]
            top_crop = top_crop[-3:]

        return render_template(
            'result.html',
            prediction=str(prediction),
            crop=crop_list,
            pred=pred_list,
            m1=top_pred,
            m2=top_crop,
            c=crop
        )

    except Exception as e:
        return str(e)


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
