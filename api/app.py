# /api/index.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
import traceback
import os
import requests
from datetime import datetime
import sys
from dotenv import load_dotenv

# --- Load environment variables ---
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
print(f"Attempting to load .env file from: {dotenv_path}")
if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path); print(".env file loaded.")
else: print(".env file not found.")

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# --- Load API Key ---
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY')
if not ALPHA_VANTAGE_KEY: print("FATAL ERROR: Alpha Vantage API key not set.") ; ALPHA_VANTAGE_KEY = None
else: print("Alpha Vantage API Key loaded.")

# --- Load Model and Scaler ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "itc_lstm_model.keras")
scaler_path = os.path.join(base_dir, "scaler.pkl")
print(f"Attempting to load model from: {model_path}")
print(f"Attempting to load scaler from: {scaler_path}")
model = None; scaler = None
try:
    if os.path.exists(model_path): model = tf.keras.models.load_model(model_path); print("Model loaded.")
    else: print(f"Model file not found at {model_path}")
    if os.path.exists(scaler_path): scaler = joblib.load(scaler_path); print("Scaler loaded.")
    else: print(f"Scaler file not found at {scaler_path}")
except Exception as e: print(f"FATAL ERROR loading model/scaler: {e}"); traceback.print_exc()

# --- Routes ---
@app.route("/")
def home():
    if model is None or scaler is None: return "Error: Model/Scaler failed to load.", 500
    print("Serving index.html")
    return render_template("index.html")

@app.route("/stockdata/<symbol>")
def get_stock_data(symbol):
    if not ALPHA_VANTAGE_KEY:
        return jsonify({"error": "API key not configured."}), 500

    print(f"Fetching AV data for {symbol}")

    try:
        # Use query parameters instead of manually building the URL.
        # This avoids symbol/key encoding problems and works reliably with
        # symbols such as ITC.BSE.
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": "compact",
            "apikey": ALPHA_VANTAGE_KEY,
        }
        headers = {
            "User-Agent": "ITC-Trend-Analysis/1.0",
            "Accept": "application/json",
        }

        response = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            print("AV returned a non-JSON response:", response.text[:500])
            return jsonify({"error": "Invalid response from Alpha Vantage."}), 502

        error_message = data.get("Error Message", "")
        information = data.get("Information", "")
        note = data.get("Note", "")

        if error_message or information or note or "Time Series (Daily)" not in data:
            combined_error = f"{error_message} {information} {note}".lower()

            if "invalid api key" in combined_error or "apikey" in combined_error and "invalid" in combined_error:
                return jsonify({"error": "Invalid Alpha Vantage API Key configured."}), 500

            if "invalid api call" in combined_error or "symbol" in combined_error and "invalid" in combined_error:
                return jsonify({"error": f"Invalid or unsupported stock symbol: {symbol}"}), 400

            api_error = error_message or information or note or "Unknown Alpha Vantage API issue"
            print("AV API Error/Limit:", api_error)
            return jsonify({"error": api_error}), 503

        ts = data["Time Series (Daily)"]
        times = sorted(ts.keys())[-70:]  # Keep the existing dashboard history size.

        if len(times) < 61:
            return jsonify({"error": f"Not enough data ({len(times)}) needed ~61"}), 400

        processed_data = []
        for t in times:
            day_data = ts[t]
            try:
                processed_data.append({
                    "t": t,
                    "o": float(day_data["1. open"]),
                    "h": float(day_data["2. high"]),
                    "l": float(day_data["3. low"]),
                    "c": float(day_data["4. close"]),
                    "v": float(day_data["5. volume"]),
                })
            except (KeyError, TypeError, ValueError):
                processed_data.append({
                    "t": t,
                    "o": None,
                    "h": None,
                    "l": None,
                    "c": None,
                    "v": None,
                })

        print(f"AV Fetch OK: {len(processed_data)} points.")
        return jsonify(processed_data)

    except requests.exceptions.Timeout:
        print("AV Timeout")
        return jsonify({"error": "Stock API timed out"}), 504

    except requests.exceptions.ConnectionError as e:
        print(f"AV Connection error: {e}")
        return jsonify({"error": "Unable to connect to Alpha Vantage."}), 504

    except requests.exceptions.RequestException as e:
        print(f"AV Network error: {e}")
        return jsonify({"error": "Network error fetching stock data"}), 504

    except Exception as e:
        print(f"Stock data processing error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Server error processing stock data"}), 500

def preprocess_prices(prices, timesteps=60): # Keep timesteps=30 here for compatibility
    prices_numeric = [p for p in prices if p is not None and not np.isnan(p)]
    if not prices_numeric: return np.zeros((1, timesteps, 1))
    if len(prices_numeric) < timesteps:
        print(f"Preprocessing warning: Not enough valid data ({len(prices_numeric)}) for {timesteps} timesteps. Padding.")
        pad_width = timesteps - len(prices_numeric)
        prices_to_process = np.pad(np.array(prices_numeric), (pad_width, 0), 'constant', constant_values=0)
    else:
        prices_to_process = np.array(prices_numeric[-timesteps:])
    X = prices_to_process.reshape(-1, 1)
    if X.shape[0] != timesteps: print(f"Shape error after pad/slice: {X.shape}"); return np.zeros((1, timesteps, 1))
    if scaler is None: raise ValueError("Scaler not loaded")
    try: X_scaled = scaler.transform(X)
    except Exception as e: print(f"Scaler error: {e}"); return np.zeros((1, timesteps, 1))
    if X_scaled.shape != (timesteps, 1): print(f"Shape error before final reshape: {X_scaled.shape}"); return np.zeros((1, timesteps, 1))
    return X_scaled.reshape(1, timesteps, 1)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None: return jsonify({"error": "Model/Scaler not loaded"}), 500
    try:
        data = request.get_json();
        prices = data.get("prices", []) # Expecting last 30 prices from JS
        if not prices or len(prices) < 30: # Check based on preprocess_prices default
             return jsonify({"error": f"Not enough data points ({len(prices)}) need 30"}), 400

        prices_clean = [p for p in prices if p is not None and not np.isnan(p)]
        if not prices_clean or len(prices_clean) < 2: # Need at least 2 for basic preprocessing checks
            return jsonify({"error": "Not enough valid data points after cleaning"}), 400

        print(f"Predict: using {len(prices_clean)} clean prices for preprocessing.")
        X = preprocess_prices(prices_clean) # Pass cleaned prices, function expects 30
        if np.all(X == 0) and len(prices_clean) > 0:
             print("Preprocessing failed, returning error.")
             return jsonify({"error": "Data preprocessing failed for prediction"}), 500

        # Make prediction
        prob = float(model.predict(X, verbose=0)[0][0]);
        label = "Bullish" if prob >= 0.5 else "Bearish"
        confidence = round(prob*100 if label=="Bullish" else (1-prob)*100, 1)


        accuracy = 75.0 

        response = { "label": label, "confidence": confidence, "accuracy": accuracy };
        print("Prediction response:", response)
        return jsonify(response)
    except Exception as e: print(f"Critical prediction error: {e}"); traceback.print_exc(); return jsonify({"error": f"Prediction error: {e}"}), 500

@app.route("/status")
def get_status():
    try: status_data = { "model_version": "V1.0", "next_retraining": "2026-01-01" }; return jsonify(status_data)
    except Exception as e: print("Status error:", e); traceback.print_exc(); return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

