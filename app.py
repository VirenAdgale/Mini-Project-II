from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes (you can also restrict it to certain routes)
CORS(app)

# Load the model
with open('fraud_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the data from the request
        print("Received data:", data)  # Print the data to the console (for debugging)

        # Ensure all necessary keys are in the received data
        if 'step' not in data or 'amount' not in data or 'oldbalanceOrg' not in data or \
           'newbalanceOrig' not in data or 'oldbalanceDest' not in data or 'newbalanceDest' not in data or \
           'type' not in data:
            return jsonify({"error": "Missing required fields"}), 400

        # Prepare the features for prediction (make sure to match the number of features used in training)
        features = [
            data['step'], 
            data['amount'], 
            data['oldbalanceOrg'], 
            data['newbalanceOrig'], 
            data['oldbalanceDest'], 
            data['newbalanceDest'],
            data['type']  # this is an encoded feature, e.g., 1 for CASH_OUT, 0 for CASH_IN
        ]
        
        # Make the prediction
        prediction = model.predict([features])

        # Send back the result
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # If error occurs, send the error message

if __name__ == '__main__':
    app.run(debug=True)
