from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)

# Loading saved model
regressor = joblib.load('model.sav')

# Importing the dataset
dataset = pd.read_csv('Final_Dataset.csv')
dataset2 = pd.read_csv('Trainset.csv')
dataset2.drop('ElectricalConductivity(ds/m)', axis=1, inplace=True)

X = dataset.loc[:, dataset.columns != 'Production']
X = X.drop('Unnamed: 0', axis=1).copy()
y = dataset['Production']
l = list(X.columns)

first = np.array(X.loc[0])
first = [list(first)]

for sk in range(len(l)):
    first[0][sk] = 0

def predict(season, crop, area, rainfall, temperature, pH, nitrogen):
    for sk in range(len(l)):
        first[0][sk] = 0

    first[0][l.index('pH')] = pH
    first[0][l.index('Nitrogen(kg/ha)')] = nitrogen
    first[0][l.index('Area')] = area
    first[0][l.index('Rainfall')] = rainfall
    first[0][l.index('Temperature')] = temperature
    first[0][l.index(season)] = 1
    first[0][l.index(crop)] = 1

    gt = regressor.predict(first)

    z_pred = int(gt)

    return z_pred

def generateGraph(season, crop, area, rainfall, temperature, pH, nitrogen):
    plt.close('all')
    df = dataset2.loc[dataset2['Season'] == season]
    crops = df['Crop'].unique()

    O = []
    P = []
    for c in crops:
        if (c != 'Sugarcane' and c != crop and c != 'Potato') or season == 'Whole Year':
            O.append(c)
            P.append(predict(season, c, area, rainfall, temperature, pH, nitrogen))
    return O, P

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        # Input
        season = request.form['season']
        crop = request.form['crop']
        area = float(request.form['area']) / dataset2['Area'].max()
        rainfall = float(request.form['rainfall']) / dataset2['Rainfall'].max()
        temperature = float(request.form['temperature']) / dataset2['Temperature'].max()
        pH = float(request.form['pH']) / dataset2['pH'].max()
        nitrogen = float(request.form['nitrogen']) / dataset2['Nitrogen(kg/ha)'].max()

        z_pred = predict(season, crop, area, rainfall, temperature, pH, nitrogen)
        z_pred = z_pred / float(request.form['area'])

        O, P = generateGraph(season, crop, area, rainfall, temperature, pH, nitrogen)

        m1, m2 = (list(t) for t in zip(*sorted(zip(P, O))))
        if len(m1) >= 3:
            m1 = m1[-3:]
            m2 = m2[-3:]

        print(m1, m2)

    return render_template('result.html', prediction=str(z_pred), crop=O, pred=P, m1=m1, m2=m2, c=crop)

if __name__ == '__main__':
    app.run(debug=True)
