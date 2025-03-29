import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Home Route
@app.route("/")
def home():
    return "🚀 Welcome to the Fraud Detection API! Use /predict to detect fraud."

# ✅ Load Model, Encoder & Scaler
try:
    model = pickle.load(open("fraud_detection_model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoder_classes = np.load("encoder_classes.npy", allow_pickle=True)
    print("✅ Model, Encoder, and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")

# ✅ Function to Map Transaction Data for Prediction
def map_transaction_data(transaction):
    """
    Converts transaction data from API request into the model's expected input format.
    """
    required_fields = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "transaction_type"]

    # ✅ Check if required fields exist
    for field in required_fields:
        if field not in transaction:
            raise ValueError(f"⚠️ Missing required field: {field}")

    # ✅ Encode transaction type (Handle unseen values)
    try:
        transaction_type_encoded = encoder.transform([transaction["transaction_type"]])[0]
    except ValueError:
        transaction_type_encoded = -1  # Assign unknown transaction type

    # ✅ Create DataFrame in the correct feature order
    df = pd.DataFrame([{
        "step": 1,  # Placeholder
        "type": transaction_type_encoded,
        "amount": transaction["amount"],
        "oldbalanceOrg": transaction["oldbalanceOrg"],
        "newbalanceOrig": transaction["newbalanceOrig"],
        "oldbalanceDest": transaction["oldbalanceDest"],
        "newbalanceDest": transaction["newbalanceDest"],
        # ✅ Added missing balance change features
        "balance_change_orig": (transaction["oldbalanceOrg"] - transaction["newbalanceOrig"]) / (transaction["oldbalanceOrg"] + 1),
        "balance_change_dest": (transaction["newbalanceDest"] - transaction["oldbalanceDest"]) / (transaction["oldbalanceDest"] + 1)
    }])

    # ✅ Ensure column order matches model training
    correct_order = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balance_change_orig", "balance_change_dest"]
    df = df[correct_order]

    # ✅ Scale numerical features (except `type`)
    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df

# ✅ API Route for Fraud Prediction
@app.route("/predict", methods=["POST"])
def predict_fraud():
    try:
        # ✅ Get JSON request data
        transaction = request.get_json()

        # ✅ Validate & process transaction
        mapped_transaction = map_transaction_data(transaction)

        # ✅ Make prediction
        fraud_probability = model.predict_proba(mapped_transaction)[0][1]
        prediction = 1 if fraud_probability > 0.05 else 0  # 🔽 Lowered threshold from 0.1 to 0.05

        # ✅ Return JSON response
        return jsonify({
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(fraud_probability), 4)
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400  # Return specific error

    except Exception as e:
        print(f"❌ Internal Server Error: {e}")  # Debugging log
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500  # Return generic error

# ✅ Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

