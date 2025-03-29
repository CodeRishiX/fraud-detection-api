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
    print("✅ Model, Encoder, and Scaler loaded successfully!")
    print(f"Encoder classes: {encoder_classes}")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    raise

def map_transaction_data(transaction):
    print("Starting map_transaction_data")
    required_fields = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "transaction_type"
    ]
    for field in required_fields:
        if field not in transaction:
            raise ValueError(f"⚠️ Missing required field: {field}")

    # Normalize transaction type (change to uppercase if your encoder expects uppercase)
    transaction_type_input = transaction["transaction_type"].strip().upper()
    print(f"Received transaction type (normalized): {transaction_type_input}")

    # Normalize encoder_classes for comparison – adjust if needed
    expected_types = [str(x).strip().upper() for x in encoder_classes]
    if transaction_type_input not in expected_types:
        raise ValueError(f"Invalid transaction type: {transaction_type_input}. Expected one of {expected_types}")

    # Encode transaction type
    try:
        print(f"Encoding transaction type: {transaction_type_input}")
        # Use a one-element list to support both LabelEncoder and OneHotEncoder
        encoded = encoder.transform([transaction_type_input])
        try:
            # If the output is sparse, convert to dense array
            encoded = encoded.toarray()
        except AttributeError:
            # Otherwise, assume it's already a NumPy array
            pass
        # If the encoder returns a 2D array (e.g., one-hot), flatten it.
        if encoded.ndim == 2:
            transaction_type_encoded = encoded[0]
        else:
            transaction_type_encoded = encoded
        print(f"Encoded transaction type: {transaction_type_encoded}")
    except Exception as e:
        print(f"Error encoding transaction type: {e}")
        raise

    # Calculate balance change features
    balance_change_orig = (transaction["oldbalanceOrg"] - transaction["newbalanceOrig"]) / (transaction["oldbalanceOrg"] + 1)
    balance_change_dest = (transaction["newbalanceDest"] - transaction["oldbalanceDest"]) / (transaction["oldbalanceDest"] + 1)
    print(f"Calculated balance_change_orig: {balance_change_orig}")
    print(f"Calculated balance_change_dest: {balance_change_dest}")

    # Create DataFrame with all features, including balance_change_orig and balance_change_dest
    print("Creating DataFrame")
    df = pd.DataFrame([{
        "step": 1,
        "type": transaction_type_encoded[0] if isinstance(transaction_type_encoded, np.ndarray) and transaction_type_encoded.ndim > 0 else transaction_type_encoded,
        "amount": transaction["amount"],
        "oldbalanceOrg": transaction["oldbalanceOrg"],
        "newbalanceOrig": transaction["newbalanceOrig"],
        "oldbalanceDest": transaction["oldbalanceDest"],
        "newbalanceDest": transaction["newbalanceDest"],
        "balance_change_orig": balance_change_orig,
        "balance_change_dest": balance_change_dest
    }])

    # Define the correct order of features, including the balance change features
    correct_order = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balance_change_orig",
        "balance_change_dest"
    ]
    df = df[correct_order].copy()
    print(f"DataFrame created: {df.to_dict(orient='records')}")

    # Scale numerical features
    num_cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balance_change_orig",
        "balance_change_dest"
    ]
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