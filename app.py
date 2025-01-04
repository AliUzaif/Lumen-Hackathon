from flask import Flask, request, jsonify
import pickle
import pandas as pd
import threading

# Load the trained model
print("Loading the trained model...")
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)
print("Model loaded successfully.")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request for prediction...")
    data = request.get_json(force=True)
    input_df = pd.DataFrame(data)
    predictions = model.predict(input_df)
    rounded_predictions = [round(pred) for pred in predictions]  # Round off predictions
    print(f"Original predictions: {predictions}")
    print(f"Rounded predictions: {rounded_predictions}")
    print("Prediction made successfully.")
    return jsonify(rounded_predictions)

# To run the flask app within Jupyter Notebook
def run_app():
    print("Starting the Flask server...")
    app.run(debug=True, use_reloader=False)

# Run the Flask app in a separate thread
threading.Thread(target=run_app).start()
print("Flask server running.")
