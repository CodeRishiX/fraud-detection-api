import pickle
import numpy as np
import pandas as pd
import os
from flask import Flask, request, jsonify

# ✅ Initialize Flask app
app = Flask(__name__)


# ======================
# 🚀 HEALTH CHECK ENDPOINT (CRITICAL FOR RENDER)
# ======================
@app.route('/health')
def health_check():
    """Endpoint for Render health checks and keepalive"""
    return jsonify({"status": "ok", "service": "fraud-detection-api"}), 200


# ======================
# 🏠 HOME ROUTE
# ======================
@app.route("/")
def home():
    return "🚀 Welcome to the Fraud Detection API! Endpoints: /predict (POST), /health (GET)"


# ======================
# 🔧 MODEL LOADING
# ======================
try:
    model = pickle.load(open("fraud_detection_model.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoder_classes = np.load("encoder_classes.npy", allow_pickle=True)
    print("✅ Model, Encoder, and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    # Critical failure - exit if models don't load
    if os.environ.get("ENV") == "production":
        raise


# ======================
# 🛠️ TRANSACTION PROCESSING
# ======================
def map_transaction_data(transaction):
    """Prepares transaction data for model prediction"""
    required_fields = ["amount", "oldbalanceOrg", "newbalanceOrig",
                       "oldbalanceDest", "newbalanceDest", "transaction_type"]

    # Validation
    for field in required_fields:
        if field not in transaction:
            raise ValueError(f"Missing required field: {field}")

    # Transaction type encoding
    try:
        transaction_type_encoded = encoder.transform([transaction["transaction_type"]])[0]
    except ValueError:
        transaction_type_encoded = -1  # Unknown type handling

    # Feature engineering
    df = pd.DataFrame([{
        "step": 1,  # Placeholder
        "type": transaction_type_encoded,
        "amount": transaction["amount"],
        "oldbalanceOrg": transaction["oldbalanceOrg"],
        "newbalanceOrig": transaction["newbalanceOrig"],
        "oldbalanceDest": transaction["oldbalanceDest"],
        "newbalanceDest": transaction["newbalanceDest"],
        "balance_change_orig": (transaction["oldbalanceOrg"] - transaction["newbalanceOrig"]) / (
                    transaction["oldbalanceOrg"] + 1),
        "balance_change_dest": (transaction["newbalanceDest"] - transaction["oldbalanceDest"]) / (
                    transaction["oldbalanceDest"] + 1)
    }])

    # Ensure correct feature order
    features_order = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                      "oldbalanceDest", "newbalanceDest", "balance_change_orig", "balance_change_dest"]
    df = df[features_order]

    # Scale numerical features
    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df


# ======================
# 🔮 PREDICTION ENDPOINT
# ======================
@app.route("/predict", methods=["POST"])
def predict_fraud():
    try:
        transaction = request.get_json()
        mapped_data = map_transaction_data(transaction)

        # Prediction
        fraud_prob = model.predict_proba(mapped_data)[0][1]
        prediction = 1 if fraud_prob > 0.05 else 0

        return jsonify({
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(fraud_prob), 4),
            "model_version": "1.0"
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# ======================
# 🚦 RUN APPLICATION
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT env var
    app.run(host="0.0.0.0", port=port, debug=(os.environ.get("ENV") != "production"))