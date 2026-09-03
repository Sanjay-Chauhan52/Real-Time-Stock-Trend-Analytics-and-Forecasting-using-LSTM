# /api/index.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf
import traceback
import os
import requests
import json
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
cache_path = os.path.join(base_dir, "cached_stock_data.json")
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

# --- Cached sample dataset fallback ---
def load_cached_stock_data(symbol):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached dataset not found at {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    if cached.get("symbol", "").upper() != symbol.upper():
        raise ValueError(f"Cached dataset is for {cached.get('symbol')}, not {symbol}")

    ts = cached.get("time_series", {})
    times = sorted(ts.keys())[-70:]
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
            processed_data.append({"t": t, "o": None, "h": None, "l": None, "c": None, "v": None})

    if len(processed_data) < 61:
        raise ValueError(f"Cached dataset has only {len(processed_data)} usable points; ~61 are required.")

    return processed_data, cached.get("last_refreshed", "N/A")


# --- Routes ---
@app.route("/")
def home():
    if model is None or scaler is None: return "Error: Model/Scaler failed to load.", 500
    print("Serving index.html")
    return render_template("index.html")

@app.route("/stockdata/<symbol>")
def get_stock_data(symbol):
    """
    Try Alpha Vantage first.
    If the API returns usable daily data, return live data.
    If the API is rate-limited, unavailable, times out, or otherwise fails,
    return the bundled cached sample data.

    The JSON payload remains the same plain array expected by index.html.
    Response headers identify whether the returned data is live or cached.
    """

    def cached_fallback(reason, rate_limited=False):
        try:
            cached_data, cached_date = load_cached_stock_data(symbol)

            api_status = (
                "Request limit reached - using sample (cached)."
                if rate_limited
                else f"{reason} - using sample (cached)."
            )

            print(
                f"Using cached sample data for {symbol}: "
                f"{len(cached_data)} points, cached date {cached_date}. "
                f"Reason: {reason}"
            )

            response = jsonify(cached_data)
            response.headers["X-Data-Source"] = "cached"
            response.headers["X-API-Status"] = api_status
            response.headers["X-Cache-Date"] = str(cached_date)
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response

        except Exception as cache_error:
            print(f"Cached fallback failed: {cache_error}")
            traceback.print_exc()
            return jsonify({
                "error": f"{reason}. Cached sample data could not be loaded."
            }), 503

    # No key: do not break the dashboard; use the bundled cache.
    if not ALPHA_VANTAGE_KEY:
        return cached_fallback("Alpha Vantage API key not configured")

    print(f"Fetching live Alpha Vantage data for {symbol}")

    try:
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

        # Bound both connection and response-read time so the dashboard does not
        # sit on "Fetching..." indefinitely when Alpha Vantage is unreachable.
        response = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            headers=headers,
            timeout=(5, 10),
        )

        print(f"Alpha Vantage HTTP status: {response.status_code}")
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            print("Alpha Vantage returned a non-JSON response:", response.text[:500])
            return cached_fallback("Invalid response from Alpha Vantage")

        error_message = str(data.get("Error Message", "") or "")
        information = str(data.get("Information", "") or "")
        note = str(data.get("Note", "") or "")
        combined_error = f"{error_message} {information} {note}".strip().lower()
        ts = data.get("Time Series (Daily)")

        # Only a real daily series with at least 61 records is considered live.
        if not isinstance(ts, dict) or len(ts) < 61:
            rate_limit_terms = (
                "rate limit",
                "requests per day",
                "standard api rate limit",
                "premium plans",
                "daily rate limit",
                "call frequency",
                "25 requests",
                "too many requests",
            )

            if any(term in combined_error for term in rate_limit_terms):
                return cached_fallback(
                    error_message or information or note or "Alpha Vantage request limit reached",
                    rate_limited=True,
                )

            if (
                "invalid api key" in combined_error
                or ("apikey" in combined_error and "invalid" in combined_error)
            ):
                return cached_fallback("Invalid Alpha Vantage API Key")

            if (
                "invalid api call" in combined_error
                or ("symbol" in combined_error and "invalid" in combined_error)
            ):
                return cached_fallback(f"Invalid or unsupported stock symbol: {symbol}")

            reason = error_message or information or note or "No usable Time Series (Daily) returned"
            return cached_fallback(reason)

        times = sorted(ts.keys())[-70:]
        if len(times) < 61:
            return cached_fallback(f"Live API returned only {len(times)} usable points")

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

        print(f"Alpha Vantage live fetch OK: {len(processed_data)} points.")

        response = jsonify(processed_data)
        response.headers["X-Data-Source"] = "live"
        response.headers["X-API-Status"] = "Live Alpha Vantage data"
        response.headers["X-Cache-Date"] = ""
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    except requests.exceptions.Timeout:
        print("Alpha Vantage request timed out")
        return cached_fallback("Alpha Vantage request timed out")

    except requests.exceptions.ConnectionError as e:
        print(f"Alpha Vantage connection error: {e}")
        return cached_fallback("Unable to connect to Alpha Vantage")

    except requests.exceptions.RequestException as e:
        print(f"Alpha Vantage network/request error: {e}")
        return cached_fallback("Network error fetching stock data")

    except Exception as e:
        print(f"Stock data processing error: {e}")
        traceback.print_exc()
        return cached_fallback("Server error processing stock data")


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

        # --- FIX: Send accuracy as a number ---
        accuracy = 75.0 # Use float, not string "75 %"
        # --- END FIX ---

        response = { "label": label, "confidence": confidence, "accuracy": accuracy };
        print("Prediction response:", response)
        return jsonify(response)
    except Exception as e: print(f"Critical prediction error: {e}"); traceback.print_exc(); return jsonify({"error": f"Prediction error: {e}"}), 500

@app.route("/status")
def get_status():
    try: status_data = { "model_version": "V1.5", "next_retraining": "2027-01-01" }; return jsonify(status_data)
    except Exception as e: print("Status error:", e); traceback.print_exc(); return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

