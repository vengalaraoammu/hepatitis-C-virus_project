from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load the trained model
model = joblib.load('model1.pkl')
pca = joblib.load('pca.pkl')
minScaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file)
    array = data.values
    features = array[:, :-1]
    actual = array[:,-1]

    # Perform any necessary preprocessing on the data
    
    
    
    #applying MinMaxScaler
    scaler_array = minScaler.transform(features)

    # Now you can use the fitted PCA for transformations
    transformed_data = pca.transform(scaler_array)


    # Make predictions using the loaded model
    predictions = model.predict(transformed_data)
    predicted_value = predictions  # Assuming prediction is a single value

    if predicted_value == 0:
        result = "Stage 1: Mild to moderate"
    elif predicted_value == 1:
        result = "Stage 2: Advanced cirrhosis"
    else:
        result = "Unknown stage"

    # Return the predictions as a response
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
