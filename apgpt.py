from flask import Flask, render_template, request
import numpy as np
import joblib
import json
from sklearn.preprocessing import OneHotEncoder

# Load your trained model here
model = joblib.load("modelrandom.pkl")

feat_cols = ['age','workclass','education','marital_status','occupation',
             'relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country']

context_dict = {
    'feats': feat_cols,
    'zip': zip,
    'range': range,
    'len': len,
    'list': list
}

# Initiate the application
app = Flask(__name__)

# Helper function to one-hot encode categorical variables
def encode_input(input_data):
    enc = OneHotEncoder(handle_unknown='ignore')
    encoded = enc.fit_transform(input_data)
    return encoded.toarray()

@app.route('/')
def index():
    return render_template('salpred.html')

@app.route('/predict', methods=['POST'])
def predict_salary():
    if request.method == 'POST':
        age = int(request.form['age'])
        workclass = request.form['workclass']
        education = request.form['education']
        marital_status = request.form['marital_status']
        occupation = request.form['occupation']
        relationship = request.form['relationship']
        race = request.form['race']
        sex = request.form['sex']
        capital_gain = float(request.form['capital_gain'])
        capital_loss = float(request.form['capital_loss'])
        hours_per_week = int(request.form['hours_per_week'])
        native_country = request.form['native_country']

        # Convert categorical variables to numerical using one-hot encoding
        cat_vars = np.array([[workclass, education, marital_status, occupation, 
                              relationship, race, sex, native_country]])
        encoded_vars = encode_input(cat_vars)

        # Combine all features into a single array for prediction
        features = np.concatenate((np.array([[age, capital_gain, capital_loss, hours_per_week]]), encoded_vars), axis=1)

        # Make prediction
        pred = model.predict(features)
        print('Prediction:', pred)

        # Return prediction as JSON response
        return json.dumps({'prediction': pred.tolist()})
    else:
        # Render the template with input form
        return render_template('salpred.html', **context_dict)

if __name__ == "__main__":
    app.run(debug=True)
