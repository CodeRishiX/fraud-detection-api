import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 Welcome to the Fraud Detection API! Use /predict to detect fraud."

# Load model, encoder, scaler, and classes
try:
    model = pickle.load(open("fraud_detection_model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoder_classes = np.load("encoder_classes.npy", allow_pickle=True)
    print(f"✅ Model, Encoder, and Scaler loaded successfully!")
    print(f"Encoder classes: {encoder_classes}")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    raise

def map_transaction_data(transaction):
    print("Starting map_transaction_data")
    required_fields = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "transaction_type"]
    for field in required_fields:
        if field not in transaction:
            raise ValueError(f"⚠️ Missing required field: {field}")

    # Validate transaction type against encoder classes
    transaction_type = transaction["transaction_type"]
    print(f"Received transaction type: {transaction_type}")
    if transaction_type not in encoder_classes:
        raise ValueError(f"Invalid transaction type: {transaction_type}. Expected one of {list(encoder_classes)}")

    # Encode transaction type
    try:
        print(f"Encoding transaction type: {transaction_type}")
        transaction_type_encoded = encoder.transform([[transaction_type]])
        transaction_type_encoded = transaction_type_encoded.toarray() if hasattr(transaction_type_encoded, "toarray") else transaction_type_encoded
        transaction_type_encoded = transaction_type_encoded[0]  # Extract first array
        print(f"Encoded transaction type: {transaction_type_encoded}")
    except ValueError as ve:
        print(f"Error encoding transaction type: {ve}")
        raise

    # Create DataFrame
    print("Creating DataFrame")
    df = pd.DataFrame([{
        "step": 1,
        "type": transaction_type_encoded[0],
        "amount": transaction["amount"],
        "oldbalanceOrg": transaction["oldbalanceOrg"],
        "newbalanceOrig": transaction["newbalanceOrig"],
        "oldbalanceDest": transaction["oldbalanceDest"],
        "newbalanceDest": transaction["newbalanceDest"]
    }])

    # Define the correct order of features
    correct_order = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df = df[correct_order].copy()
    print(f"DataFrame created: {df.to_dict(orient='records')}")

    # Scale numerical features
    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    print("Scaling numerical features")
    try:
        df[num_cols] = scaler.transform(df[num_cols])
        print("Scaling completed")
    except Exception as e:
        print(f"Error during scaling: {e}")
        raise

    return df

@app.route("/predict", methods=["POST"])
def predict_fraud():
    print("Received request to /predict")
    print(f"Headers: {request.headers}")
    try:
        transaction = request.get_json()
        print(f"Input JSON: {transaction}")
        mapped_transaction = map_transaction_data(transaction)
        print("Mapped transaction successfully")
        print("Making prediction")
        fraud_probability = model.predict_proba(mapped_transaction)[0][1]
        prediction = 1 if fraud_probability > 0.05 else 0
        print(f"Prediction: {prediction}, Probability: {fraud_probability}")
        return jsonify({"is_fraud": bool(prediction), "fraud_probability": round(float(fraud_probability), 4)})
    except ValueError as ve:
        print(f"ValueError in /predict: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"❌ Internal Server Error in /predict: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
