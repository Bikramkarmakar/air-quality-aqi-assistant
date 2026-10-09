"""
predictor.py
------------
Loads saved model artifacts and exposes a single predict() function
used by both the Streamlit app and any backend service.

Also handles live AQI API cross-check via WAQI (World Air Quality Index)
or OpenWeatherMap Air Pollution API using a user-supplied API key loaded
securely from .env / Streamlit secrets.
"""

from __future__ import annotations

import os
import pickle
import json
import requests
from typing import Optional

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "aqi_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
LE_PATH       = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_PATH  = os.path.join(MODEL_DIR, "model_metrics.json")

BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# ── module-level singletons (lazy-loaded) ─────────────────────────────────────
_pipeline      = None
_feature_names = None
_le            = None
_metrics       = None


def _load_artifacts():
    global _pipeline, _feature_names, _le, _metrics
    if _pipeline is not None:
        return
    with open(MODEL_PATH, "rb") as f:
        _pipeline = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        _feature_names = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        _le = pickle.load(f)
    with open(METRICS_PATH) as f:
        _metrics = json.load(f)


def get_feature_names() -> list[str]:
    _load_artifacts()
    return list(_feature_names)


def get_metrics() -> dict:
    _load_artifacts()
    return _metrics


def predict(input_values: dict[str, float]) -> dict:
    """
    Predict AQI bucket from a dict of pollutant/feature values.

    Parameters
    ----------
    input_values : dict  e.g. {"PM2.5": 45.0, "PM10": 80.0, ...}
                   Missing keys are filled with 0.0 (safe default).

    Returns
    -------
    dict:
        predicted_bucket  : str
        predicted_index   : int
        probabilities     : dict {bucket_name: probability}
        top_features      : list of (feature, importance) sorted desc
    """
    _load_artifacts()
    row = [input_values.get(f, 0.0) for f in _feature_names]
    X   = pd.DataFrame([row], columns=_feature_names)
    pred_idx  = int(_pipeline.predict(X)[0])
    pred_prob = _pipeline.predict_proba(X)[0]

    bucket = _le.classes_[pred_idx]

    probs = {
        str(_le.classes_[i]): round(float(p), 4)
        for i, p in enumerate(pred_prob)
    }

    # feature importances from the stored metrics
    fi = {
        item["feature"]: item["importance"]
        for item in _metrics.get("feature_importances", [])
    }
    top_features = sorted(
        [(f, fi.get(f, 0.0)) for f in _feature_names],
        key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "predicted_bucket": bucket,
        "predicted_index":  pred_idx,
        "probabilities":    probs,
        "top_features":     top_features,
    }


# ── Live API cross-check ──────────────────────────────────────────────────────

def _get_api_key() -> Optional[str]:
    """
    Retrieve API key from:
    1. Streamlit secrets (st.secrets) if running in Streamlit
    2. Environment variable AQI_API_KEY (loaded from .env via python-dotenv)
    """
    # Try Streamlit secrets first
    try:
        import streamlit as st
        key = st.secrets.get("AQI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    # Fall back to environment / .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("AQI_API_KEY", None)


def fetch_live_aqi(city_name: str) -> dict:
    """
    Query the WAQI API (https://waqi.info) for real-time AQI of a city.

    Returns
    -------
    dict with keys:
        success      : bool
        live_aqi     : int or None
        live_bucket  : str or None
        station      : str
        message      : str   (shown to user)
    """
    api_key = _get_api_key()

    if not api_key:
        return {
            "success":     False,
            "live_aqi":    None,
            "live_bucket": None,
            "station":     "",
            "message":     (
                "Live AQI verification is unavailable: no API key configured. "
                "Add AQI_API_KEY to your .env file or Streamlit secrets."
            ),
        }

    url = f"https://api.waqi.info/feed/{city_name.lower()}/?token={api_key}"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return {
                "success":     False,
                "live_aqi":    None,
                "live_bucket": None,
                "station":     "",
                "message":     f"WAQI API returned status: {data.get('status')}. "
                               "City may not be found or API key is invalid.",
            }

        aqi_val = data["data"]["aqi"]
        station = data["data"].get("city", {}).get("name", city_name)

        # Map numeric AQI to bucket (CPCB scale)
        live_bucket = _numeric_aqi_to_bucket(aqi_val)

        return {
            "success":     True,
            "live_aqi":    int(aqi_val),
            "live_bucket": live_bucket,
            "station":     station,
            "message":     f"Live data from WAQI station: {station}",
        }

    except requests.exceptions.ConnectionError:
        return {
            "success":     False,
            "live_aqi":    None,
            "live_bucket": None,
            "station":     "",
            "message":     "Could not connect to WAQI API. Check your internet connection.",
        }
    except requests.exceptions.Timeout:
        return {
            "success":     False,
            "live_aqi":    None,
            "live_bucket": None,
            "station":     "",
            "message":     "WAQI API request timed out. Using model prediction only.",
        }
    except Exception as exc:
        return {
            "success":     False,
            "live_aqi":    None,
            "live_bucket": None,
            "station":     "",
            "message":     f"Live API error: {exc}",
        }


def _numeric_aqi_to_bucket(aqi: float) -> str:
    """Convert numeric AQI to CPCB AQI_Bucket label."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"
