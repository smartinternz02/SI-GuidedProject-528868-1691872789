from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load(open(r"C:\Users\abish\OneDrive\Documents\Project\random_forest_model.joblib", 'rb'))
scaler = StandardScaler()

# Load your original dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv(r"C:\Users\abish\OneDrive\Documents\Project\natural_gas_price.csv")

# Parse 'Date' column into 'day', 'month', and 'year' columns
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

# Extract the training features from your dataset
x_train = data[['year', 'month', 'day']].values

# Fit the scaler with the training data
scaler.fit(x_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])

    # Use the fitted scaler to transform input features
    x_test = scaler.transform(np.array([[year, month, day]]))

    prediction = model.predict(x_test)
    pred = prediction[0]
    return render_template('index.html', prediction_text='Gas Price is {} Dollars'.format(pred))

if __name__ == '__main__':
    app.run(debug=True)
