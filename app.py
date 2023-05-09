from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle

from sklearn import metrics
from sklearn.model_selection import train_test_split



# Load the data
data = pd.read_csv("Crop_recommendation.csv")

# Create a random forest classifier
model = RandomForestClassifier()

# Train the model on the data
X = data[['N','P','K','ph','temperature', 'humidity']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
warnings.filterwarnings("ignore")

# Create a Flask app
app = Flask(__name__)

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')



# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Accept input from the user
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorous = float(request.form['Phosphorous'])
    Potassium = float(request.form['Potassium'])
    ph = float(request.form['ph'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    
    # rainfall = float(request.form['rainfall'])

    # Make a prediction based on the user input
    prediction = model.predict([[Nitrogen,Phosphorous,Potassium,ph,temperature, humidity]])

    # Render the result template with the prediction
    return render_template('index.html', prediction_text = '{} '.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
