from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle, numpy as np
import pandas as pd

app = Flask(__name__, static_folder="../static")
CORS(app)

# Load model once at startup
with open("diabetes_model.pkl", "rb") as f:
    payload = pickle.load(f)
model    = payload["model"]
scaler   = payload["scaler"]
features = payload["features"]

@app.route("/")
def index():
    return send_from_directory("../static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        arr    = pd.DataFrame([data["features"]], columns=features)
        arr_sc = scaler.transform(arr)
        pred   = model.predict(arr_sc)[0]
        prob   = float(model.predict_proba(arr_sc)[0][1])
        return jsonify({
            "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
            "probability_diabetic": round(prob, 4)
        })
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features": features})

if __name__ == "__main__":
    app.run(debug=True)