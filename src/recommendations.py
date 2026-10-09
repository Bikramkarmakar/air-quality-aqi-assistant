"""
recommendations.py
------------------
Safety recommendations engine for the Air Quality AQI Assistant.
Returns plain-language advice based on AQI bucket, dominant pollutant,
and user persona.
"""

from __future__ import annotations

# Canonical AQI bucket order (CPCB scale, least to most severe)
BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# ── AQI bucket metadata ──────────────────────────────────────────────────────
BUCKET_META: dict[str, dict] = {
    "Good": {
        "color": "#00C853",
        "emoji": "🟢",
        "health_msg": "Air quality is good. Enjoy your day outdoors!",
        "aqi_range": "0 – 50",
    },
    "Satisfactory": {
        "color": "#64DD17",
        "emoji": "🟡",
        "health_msg": "Air quality is acceptable. Sensitive individuals should take note.",
        "aqi_range": "51 – 100",
    },
    "Moderate": {
        "color": "#FFD600",
        "emoji": "🟠",
        "health_msg": "Moderate pollution. Prolonged outdoor exertion may cause discomfort.",
        "aqi_range": "101 – 200",
    },
    "Poor": {
        "color": "#FF6D00",
        "emoji": "🔴",
        "health_msg": "Poor air quality. Reduce outdoor activities, especially exercise.",
        "aqi_range": "201 – 300",
    },
    "Very Poor": {
        "color": "#DD2C00",
        "emoji": "🔴",
        "health_msg": "Very poor air. Avoid outdoor exposure as much as possible.",
        "aqi_range": "301 – 400",
    },
    "Severe": {
        "color": "#7B1FA2",
        "emoji": "🟣",
        "health_msg": "Severe pollution! Stay indoors, keep windows closed, use air purifier.",
        "aqi_range": "401+",
    },
}

# ── Pollutant thresholds for dominant-pollutant detection ──────────────────
# Values roughly based on CPCB standard breakpoints
POLLUTANT_THRESHOLDS: dict[str, float] = {
    "PM2.5":   60.0,
    "PM10":   100.0,
    "NO2":     80.0,
    "SO2":     80.0,
    "CO":      10.0,
    "O3":      100.0,
    "NH3":     400.0,
    "NO":      80.0,
    "NOx":     80.0,
    "Benzene":  5.0,
    "Toluene": 50.0,
}

POLLUTANT_DESCRIPTIONS: dict[str, str] = {
    "PM2.5":  "Fine particulate matter (PM2.5) — tiny particles from vehicles, fires, and industry that penetrate deep into lungs.",
    "PM10":   "Coarse dust particles (PM10) — from construction, roads, and soil. Irritate throat and lungs.",
    "NO2":    "Nitrogen dioxide (NO2) — from vehicle exhaust and power plants. Worsens asthma and reduces lung function.",
    "SO2":    "Sulfur dioxide (SO2) — from burning coal and industrial processes. Causes breathing difficulty.",
    "CO":     "Carbon monoxide (CO) — odourless gas from vehicles and combustion. Reduces oxygen in blood.",
    "O3":     "Ground-level ozone (O3) — formed by sunlight reacting with pollutants. Causes chest tightness.",
    "NH3":    "Ammonia (NH3) — from agriculture and waste. Irritates eyes, nose, and respiratory tract.",
    "NO":     "Nitric oxide (NO) — from combustion. Converts to NO2 in air.",
    "NOx":    "Nitrogen oxides (NOx) — vehicle and industrial emissions. Contribute to smog and acid rain.",
    "Benzene":"Benzene — from vehicle exhaust and petrol. A known carcinogen; prolonged exposure is harmful.",
    "Toluene":"Toluene — industrial solvent and fuel component. High levels cause headache and dizziness.",
}

# ── Persona-specific advice ──────────────────────────────────────────────────
PERSONA_ADVICE: dict[str, dict[str, str]] = {
    "Student": {
        "Good":          "Great day for outdoor sports and activities at school.",
        "Satisfactory":  "Normal outdoor activities are fine. Drink plenty of water.",
        "Moderate":      "Limit vigorous outdoor exercise. Take breaks indoors.",
        "Poor":          "Avoid outdoor PE/sports. Study or exercise indoors today.",
        "Very Poor":     "Stay indoors. If commuting, wear an N95 mask.",
        "Severe":        "Do not go to school if possible. Study online. Keep all windows shut.",
    },
    "Commuter": {
        "Good":          "Enjoy your commute — air is clean today.",
        "Satisfactory":  "Normal commute is fine. Windows can remain slightly open.",
        "Moderate":      "Keep car/bus windows closed. Use the air recirculation mode.",
        "Poor":          "Wear a mask during transit. Avoid cycling or walking on busy roads.",
        "Very Poor":     "Use public transport instead of cycling/walking. Wear an N95 mask.",
        "Severe":        "Work from home if possible. If commuting, wear N95, avoid rush hour.",
    },
    "Asthma Patient": {
        "Good":          "Air is clean — safe for normal activities.",
        "Satisfactory":  "Keep rescue inhaler handy as a precaution.",
        "Moderate":      "Take preventive inhaler before going outdoors. Monitor symptoms.",
        "Poor":          "Stay indoors. Use air purifier. Contact your doctor if symptoms worsen.",
        "Very Poor":     "Do not go outdoors. Keep bronchodilator accessible. Seek medical help if needed.",
        "Severe":        "MEDICAL ALERT: Stay indoors, use purifier at maximum, call doctor proactively.",
    },
    "Elderly Person": {
        "Good":          "Safe for morning walks and light outdoor activities.",
        "Satisfactory":  "Short outdoor walks are fine. Avoid peak traffic hours.",
        "Moderate":      "Keep outdoor time brief. Rest indoors after activity.",
        "Poor":          "Stay indoors. Light indoor exercises only. Hydrate well.",
        "Very Poor":     "Do not go outdoors. Keep medications near. Monitor blood pressure.",
        "Severe":        "HEALTH ALERT: Stay indoors completely. Alert caregivers or family members.",
    },
    "Outdoor Worker": {
        "Good":          "Normal work conditions. Stay hydrated.",
        "Satisfactory":  "Normal work. Take regular water breaks.",
        "Moderate":      "Take frequent breaks in shaded/sheltered areas. Wear a basic dust mask.",
        "Poor":          "Wear an N95 mask continuously. Limit heavy physical exertion outdoors.",
        "Very Poor":     "Wear full respiratory protection. Request indoor reassignment if possible.",
        "Severe":        "OCCUPATIONAL HAZARD: Work should be paused or moved indoors. Mandatory PPE required.",
    },
    "General Public": {
        "Good":          "Excellent air quality — great day for all outdoor activities.",
        "Satisfactory":  "Air quality is acceptable for most people.",
        "Moderate":      "Sensitive groups may feel mild irritation. Limit prolonged outdoor exposure.",
        "Poor":          "Everyone may experience effects. Reduce outdoor time.",
        "Very Poor":     "Health effects likely for all. Avoid outdoor activities.",
        "Severe":        "Health emergency conditions. Stay indoors. Seek medical advice if symptomatic.",
    },
}

# ── General protective tips by bucket ────────────────────────────────────────
GENERAL_TIPS: dict[str, list[str]] = {
    "Good":          ["Open windows to ventilate your home naturally.",
                      "Ideal time for physical exercise outdoors.",
                      "No special precautions needed."],
    "Satisfactory":  ["Light outdoor activity is fine.",
                      "Sensitive individuals should monitor symptoms.",
                      "Keep children's outdoor play to moderate levels."],
    "Moderate":      ["Wear a surgical mask outdoors for extended periods.",
                      "Close windows during peak traffic hours.",
                      "Use indoor air purifiers if available.",
                      "Avoid burning waste or candles indoors."],
    "Poor":          ["Wear N95/FFP2 mask if going outside.",
                      "Keep windows and doors closed.",
                      "Run air purifiers at medium/high setting.",
                      "Avoid strenuous outdoor physical activities.",
                      "Check on elderly neighbours and children."],
    "Very Poor":     ["Stay indoors as much as possible.",
                      "Seal window gaps with wet cloth if no purifier is available.",
                      "Do not smoke or use incense/candles indoors.",
                      "Inform local health authorities if levels persist.",
                      "Hydrate frequently — water helps flush pollutants."],
    "Severe":        ["EMERGENCY: Treat this as a health emergency.",
                      "Stay completely indoors with windows and doors sealed.",
                      "Use N95 masks even indoors if purifier is unavailable.",
                      "Evacuate to a cleaner area if medically advised.",
                      "Contact emergency medical services if you have breathing difficulty.",
                      "Do not allow children or elderly outside under any circumstances."],
}


def get_dominant_pollutant(pollutant_values: dict[str, float]) -> tuple[str, str]:
    """
    Return (pollutant_name, description) for the pollutant most exceeding
    its threshold. Falls back to PM2.5 if none exceed thresholds.
    """
    max_ratio  = -1.0
    dominant   = "PM2.5"
    for poll, val in pollutant_values.items():
        if poll in POLLUTANT_THRESHOLDS and val is not None:
            ratio = val / POLLUTANT_THRESHOLDS[poll]
            if ratio > max_ratio:
                max_ratio = ratio
                dominant  = poll
    return dominant, POLLUTANT_DESCRIPTIONS.get(dominant, "Unknown pollutant.")


def get_recommendations(
    bucket: str,
    persona: str,
    pollutant_values: dict[str, float],
) -> dict:
    """
    Return a structured recommendations dict.

    Parameters
    ----------
    bucket           : predicted AQI bucket label
    persona          : user persona string
    pollutant_values : dict of {feature_name: numeric_value}

    Returns
    -------
    dict with keys:
        bucket, color, emoji, health_msg, aqi_range,
        dominant_pollutant, pollutant_desc,
        persona_advice, general_tips
    """
    meta          = BUCKET_META.get(bucket, BUCKET_META["Moderate"])
    dom_poll, dom_desc = get_dominant_pollutant(pollutant_values)

    pers_advice   = (
        PERSONA_ADVICE
        .get(persona, PERSONA_ADVICE["General Public"])
        .get(bucket, "Follow general safety guidelines.")
    )
    tips = GENERAL_TIPS.get(bucket, [])

    return {
        "bucket":             bucket,
        "color":              meta["color"],
        "emoji":              meta["emoji"],
        "health_msg":         meta["health_msg"],
        "aqi_range":          meta["aqi_range"],
        "dominant_pollutant": dom_poll,
        "pollutant_desc":     dom_desc,
        "persona_advice":     pers_advice,
        "general_tips":       tips,
    }
