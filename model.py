import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("Crop_recommendation.csv")

# Create a random forest classifier
model = RandomForestClassifier()

# Train the model on the data
X = data[['N','P','K','ph','temperature', 'humidity']]
y = data['label']
model.fit(X, y)

# Accept input from the user
Nitrogen = float(input("Enter Nitrogen: "))
Phosphorous = float(input("Enter Phosphorous: "))
Potassium = float(input("Enter Potassium: "))
ph = float(input("Enter pH: "))
temperature = float(input("Enter temperature: "))
humidity = float(input("Enter humidity: "))

# rainfall = float(input("Enter rainfall: "))

# Make a prediction based on the user input

prediction = model.predict([[Nitrogen,Phosphorous,Potassium,ph,temperature, humidity]])

# Print the prediction
print("Predicted crop: ", prediction[0])

# save model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Loading model to compare the results
loaded_model = pickle.load(open(filename, 'rb'))