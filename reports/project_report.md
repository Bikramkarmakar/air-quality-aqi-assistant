# Air Quality AQI Assistant — Complete Project Report

**Project Title:** Air Quality AQI Assistant — Agentic AI + ML Web Application  
**Report Version:** 1.0  
**Date:** 2024  
**Target Audience:** IBM Internship Evaluation, Technical Reviewers, Academic Assessment  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [SDG Targets](#3-sdg-targets)
4. [Dataset Description](#4-dataset-description)
5. [Data Cleaning and Validation](#5-data-cleaning-and-validation)
6. [Machine Learning Model](#6-machine-learning-model)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Performance and Results](#8-model-performance-and-results)
9. [Application Architecture](#9-application-architecture)
10. [Safety Recommendation System](#10-safety-recommendation-system)
11. [Live API Integration](#11-live-api-integration)
12. [Security and API Key Handling](#12-security-and-api-key-handling)
13. [Novelty and Originality](#13-novelty-and-originality)
14. [Lean Canvas](#14-lean-canvas)
15. [Limitations](#15-limitations)
16. [Future Scope](#16-future-scope)
17. [References](#17-references)

---

## 1. Executive Summary

Air pollution is one of the most significant environmental health risks globally, responsible
for millions of premature deaths each year. In India alone, poor air quality affects hundreds
of millions of people daily, yet most citizens lack access to simple, actionable guidance on
how to protect their health on bad air days.

The **Air Quality AQI Assistant** is an original agentic AI web application that bridges this
gap. Using historical Indian city air quality data from 26 cities (2015-2020), the system
trains a lightweight Random Forest classifier to predict the CPCB AQI category from real-world
pollutant readings. The application identifies the dominant pollutant in plain language, provides
persona-specific safety advice tailored to the user's vulnerability profile, and optionally
cross-checks predictions against live real-time data from the WAQI public API.

The project achieves 80.26% test accuracy across six AQI categories on a 24,850-row dataset,
operates entirely on a standard laptop, and is deployable as a public web app on Streamlit Cloud
at zero cost.

---

## 2. Problem Statement

### The Challenge

Every day, millions of students, commuters, elderly residents, asthma patients, and outdoor
workers make decisions — whether to exercise outdoors, open a window, wear a mask, or stay
inside — without reliable, personalised air quality guidance.

Existing solutions either:
- Require expensive hardware or paid subscriptions
- Show raw numbers (AQI = 187) without meaningful context for ordinary users
- Provide generic national-level warnings rather than location-specific, person-specific advice
- Do not explain *why* the air is bad (which pollutant dominates)

### The Solution Gap

There is no widely accessible, free, AI-powered tool that:
1. Accepts local pollutant readings as inputs
2. Predicts risk category using trained ML
3. Identifies the primary pollution source
4. Gives different advice to a student vs. an asthma patient vs. an outdoor worker
5. Cross-validates with real-time sensor data
6. Works both offline (ML only) and online (with live API)

### Our Response

The Air Quality AQI Assistant addresses all five gaps in a single, deployable, open-source
Streamlit application trained on authentic Indian city data.

---

## 3. SDG Targets

### SDG 3: Good Health and Well-being

**Target 3.9:** By 2030, substantially reduce the number of deaths and illnesses from hazardous
chemicals and air, water and soil pollution and contamination.

**Contribution:** The application directly supports target 3.9 by:
- Providing personalised health risk assessments for vulnerable populations
- Enabling proactive protective behaviour before health effects occur
- Giving asthma patients and elderly people earlier warnings than generic AQI broadcasts
- Translating technical air quality indices into simple, actionable instructions

### SDG 11: Sustainable Cities and Communities

**Target 11.6:** By 2030, reduce the adverse per capita environmental impact of cities,
including by paying special attention to air quality.

**Contribution:**
- Builds urban air quality literacy among citizens
- Enables commuters and outdoor workers to make safer route and schedule choices
- Creates awareness that drives demand for cleaner transport and industrial practices

### SDG 13: Climate Action

**Target 13.3:** Improve education, awareness-raising, and human and institutional capacity
on climate change mitigation, adaptation, impact reduction, and early warning.

**Contribution:**
- Tracks how seasonal and climate-driven changes affect daily air quality
- Highlights which pollutants (CO, SO2, NOx) are directly tied to fossil fuel combustion
- Demonstrates that AI tools can serve as low-cost early warning systems for pollution events

---

## 4. Dataset Description

### Primary Dataset: city_day.csv

| Property | Value |
|----------|-------|
| Source | CPCB (Central Pollution Control Board, India) via Kaggle |
| Rows | 29,531 (24,850 after cleaning) |
| Columns | 16 (City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI, AQI_Bucket) |
| Cities | 26 major Indian cities |
| Date Range | January 2015 to July 2020 |
| Target Variable | AQI_Bucket (6 categories) |

### AQI Category Distribution (after cleaning)

| Category | Rows | Percentage |
|----------|------|------------|
| Moderate | 8,829 | 35.5% |
| Satisfactory | 8,224 | 33.1% |
| Poor | 2,781 | 11.2% |
| Very Poor | 2,337 | 9.4% |
| Severe | 1,338 | 5.4% |
| Good | 1,341 | 5.4% |

### Supporting Datasets

- **station_day.csv** (108,035 rows): Station-level daily readings. Used for context and
  potential future expansion to station-level predictions.
- **stations.csv** (230 rows): Station metadata including StationId, City, State, and Status.
  Used for geographic reference.

### Data Quality Issues Found

| Issue | Columns Affected | Resolution |
|-------|-----------------|------------|
| Missing target rows | AQI, AQI_Bucket — 4,681 rows | Dropped |
| High missingness (>60%) | Xylene | Column dropped |
| Moderate missingness | PM10, NH3, Benzene, Toluene | City-level median imputation |
| Low missingness (<5%) | PM2.5, NO, NO2, CO, SO2, O3 | City-level median imputation |
| Sensor outliers | All pollutant columns | Capped at 99.9th percentile |

---

## 5. Data Cleaning and Validation

### Pipeline Steps (src/data_cleaning.py)

1. **Load**: Read city_day.csv, parse Date column as datetime
2. **Drop missing targets**: Remove rows where AQI or AQI_Bucket is null (4,681 rows removed)
3. **Drop Xylene**: Over 62% missing — not useful for imputation; dropped entirely
4. **Validate labels**: Ensure AQI_Bucket values match the six known CPCB categories
5. **Cap outliers**: Clip each pollutant at its 99.9th percentile to remove sensor noise
6. **Impute missing values**: City-level median first; global median as fallback
7. **Add temporal features**: Extract Month and DayOfYear from Date for seasonal patterns
8. **Export**: Save to data/processed_city_day.csv (24,850 rows, 15 columns)

### Design Decisions

- **City-level imputation** rather than global imputation preserves local pollution patterns.
  Delhi and Ahmedabad have very different baseline PM2.5 levels; a global median would distort both.
- **Xylene was dropped** rather than imputed because over 60% of values are missing even after
  filtering to rows with known AQI_Bucket. Imputing >60% would introduce more noise than signal.
- **Outlier capping at 99.9%** rather than removal preserves the dataset size while eliminating
  implausible sensor spikes (e.g., PM2.5 = 4,200 µg/m³ in a single reading).

---

## 6. Machine Learning Model

### Model Selection

We evaluated the following candidates for this tabular multi-class classification task:

| Model | Reason Considered | Decision |
|-------|------------------|----------|
| Logistic Regression | Baseline, fast | Too simplistic for non-linear boundaries |
| Decision Tree | Interpretable | Overfits; replaced by ensemble |
| **Random Forest** | Ensemble, robust, handles imbalance | **Selected** |
| Gradient Boosting (XGBoost) | High accuracy | Too heavy for lightweight constraint |
| SVM | Good on tabular data | Slow training on 24K rows |

**Random Forest** was chosen because:
- Handles missing imputation effects gracefully
- Built-in feature importance (used for pollutant explanation)
- `class_weight="balanced"` mitigates the class imbalance in the dataset
- Trains in under 30 seconds on a standard laptop
- Requires no hyperparameter tuning to achieve good baseline performance

### Model Architecture

```
Pipeline:
  Step 1: StandardScaler  — normalise all numeric features
  Step 2: RandomForestClassifier
            n_estimators  = 200
            max_depth     = 18
            min_samples_leaf = 3
            class_weight  = "balanced"
            random_state  = 42
```

### Target Variable

- **AQI_Bucket** (6-class classification): Good, Satisfactory, Moderate, Poor, Very Poor, Severe
- We chose **classification over regression** because:
  - AQI_Bucket is directly meaningful to users (actionable categories)
  - Regression on AQI numeric value requires binning anyway for advice
  - Class-balanced training prevents the model from always predicting "Moderate"

---

## 7. Feature Engineering

### Input Features (13 total)

| Feature | Type | Rationale |
|---------|------|-----------|
| PM2.5 | Pollutant | Highest predictor (~28% importance) |
| PM10 | Pollutant | Second highest (~17%) |
| CO | Pollutant | Strong AQI driver (~15%) |
| NO | Pollutant | Combustion marker |
| O3 | Pollutant | Photochemical smog indicator |
| NO2 | Pollutant | Urban traffic marker |
| NOx | Pollutant | Combined nitrogen oxide |
| NH3 | Pollutant | Agricultural/industrial source |
| SO2 | Pollutant | Industrial/coal burning |
| Benzene | Pollutant | VOC / carcinogen |
| Toluene | Pollutant | Industrial solvent |
| Month | Temporal | Seasonal variation (winter fog, monsoon) |
| DayOfYear | Temporal | Finer seasonal resolution |

### Feature Importance (from trained model)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | PM2.5 | 28.4% |
| 2 | PM10 | 17.2% |
| 3 | CO | 14.9% |
| 4 | NO | 5.3% |
| 5 | O3 | 4.9% |
| 6 | NOx | 4.6% |
| 7 | DayOfYear | 4.2% |
| 8 | NH3 | 3.8% |
| 9 | NO2 | 3.7% |
| 10 | SO2 | 3.5% |

PM2.5, PM10, and CO together account for over 60% of model predictive power — consistent
with known air quality science where particulate matter drives AQI in Indian cities.

---

## 8. Model Performance and Results

### Evaluation on 20% Hold-Out Test Set (4,970 rows)

| Metric | Value |
|--------|-------|
| Test Accuracy | 80.26% |
| Weighted F1 Score | 80.43% |
| Macro F1 Score | 79.07% |
| 5-Fold Cross-Validation Accuracy | 79.99% ± 0.62% |

### Per-Class Performance

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Good | 0.715 | 0.832 | 0.769 | 268 |
| Satisfactory | 0.843 | 0.827 | 0.835 | 1,645 |
| Moderate | 0.843 | 0.780 | 0.810 | 1,766 |
| Poor | 0.639 | 0.804 | 0.712 | 556 |
| Very Poor | 0.804 | 0.775 | 0.790 | 467 |
| Severe | 0.839 | 0.817 | 0.828 | 268 |

### Analysis

- **Satisfactory and Moderate** categories achieve the best F1 scores (0.835, 0.810) as expected
  given their higher representation in the dataset.
- **Poor** has the lowest precision (0.639) — the model occasionally predicts "Poor" when the
  true label is "Very Poor" or "Moderate", suggesting these adjacent categories overlap in
  pollutant space.
- **Severe** achieves a strong F1 of 0.828 despite only 268 test samples, indicating the
  `class_weight="balanced"` parameter works well for minority classes.
- The 5-fold CV score of 79.99% is very close to the test accuracy, confirming the model
  generalises well and is not overfitting.

### Interpretation

An 80% accuracy on a 6-class problem with significant class imbalance and real-world
sensor noise is a strong result for a lightweight model. For the application's use case
— giving citizens category-level health advice — a one-category error (e.g., predicting
"Moderate" when true is "Poor") is acceptable; the advice given for adjacent categories
is broadly similar. The application displays confidence probabilities to help users
interpret borderline predictions.

---

## 9. Application Architecture

```
User (Browser)
     |
     v
Streamlit Frontend (app.py)
     |
     +----> src/predictor.py  ------> models/aqi_model.pkl
     |              |                 models/feature_names.pkl
     |              |                 models/label_encoder.pkl
     |              |
     |              +----> WAQI API (live check, optional)
     |
     +----> src/recommendations.py  (safety advice engine)
     |
     +----> data/processed_city_day.csv  (data explorer)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `app.py` | Streamlit UI, user inputs, results rendering |
| `src/predictor.py` | Load model, run predictions, live API handler |
| `src/recommendations.py` | Dominant pollutant detection, persona advice, tips |
| `src/data_cleaning.py` | One-time data pipeline (not called at runtime) |
| `src/train_model.py` | One-time training script (not called at runtime) |

### Tab Structure (app.py)

1. **Predict & Advise** — Main interaction: enter readings, get prediction + advice
2. **Data Explorer** — Interactive dataset visualisation
3. **Model Info** — Performance metrics and feature importances
4. **About & SDGs** — Project purpose, SDG alignment, API setup guide

---

## 10. Safety Recommendation System

The recommendation engine (`src/recommendations.py`) operates in three layers:

### Layer 1: Dominant Pollutant Identification

Each pollutant has a defined threshold (based on CPCB standard breakpoints). The engine
computes how many times each pollutant exceeds its threshold and selects the one with
the highest exceedance ratio as the "dominant pollutant." This is then explained in
plain, non-technical language.

### Layer 2: Persona-Specific Advice

Six user personas each have six bucket-specific advice strings (36 messages total):

| Persona | Key Concern |
|---------|------------|
| Student | Outdoor exercise, school commute |
| Commuter | Transit mode, mask usage |
| Asthma Patient | Inhaler use, indoor shelter, medical escalation |
| Elderly Person | Cardiovascular risk, caregiver alerts |
| Outdoor Worker | PPE requirements, work relocation |
| General Public | Broad population guidance |

### Layer 3: General Protective Tips

Each AQI bucket has a list of general protective actions (2–6 tips) covering ventilation,
mask types, indoor air quality, hydration, and emergency procedures.

---

## 11. Live API Integration

The application integrates with the **WAQI (World Air Quality Index) API** to fetch
real-time AQI readings for any city worldwide.

### How It Works

1. User enters a city name in the sidebar
2. On prediction, the app sends a GET request to `https://api.waqi.info/feed/{city}/?token={key}`
3. The API returns the current numeric AQI, which is mapped to a CPCB AQI_Bucket
4. The live result is displayed alongside the ML prediction with a match/differ indicator

### Fallback Behaviour

The live check is completely optional. If the API key is missing, invalid, or the API
is unavailable, the app:
- Shows a clear informational message explaining why live data is unavailable
- Continues to display the ML-based prediction without interruption
- Does not throw errors or block the user interface

### CPCB AQI Numeric-to-Bucket Mapping

| Range | Bucket |
|-------|--------|
| 0 – 50 | Good |
| 51 – 100 | Satisfactory |
| 101 – 200 | Moderate |
| 201 – 300 | Poor |
| 301 – 400 | Very Poor |
| 401+ | Severe |

---

## 12. Security and API Key Handling

API key security is a first-class concern in this project.

### The Problem

Hard-coding API keys in source code is a critical security vulnerability. Keys committed
to public GitHub repositories are routinely scraped by bots within minutes of being pushed.

### Our Solution (Defence-in-Depth)

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| `.env` file | Local development secret store | Never pushed to GitHub |
| `.gitignore` | Blocks `.env` from git staging | Prevents accidental commits |
| `.env.example` | Template with placeholder | Documents required variable safely |
| `python-dotenv` | Loads `.env` at runtime | Clean separation of code and secrets |
| Streamlit Secrets | Cloud deployment secret store | Secure injection in production |
| Code fallback | Graceful degradation if key absent | App works without key |

### Code Pattern

```python
def _get_api_key():
    # 1. Try Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        key = st.secrets.get("AQI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    # 2. Try environment variable / .env file (local dev)
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("AQI_API_KEY", None)
```

The real API key is **never present in any source file** committed to GitHub.

---

## 13. Novelty and Originality

This project is original in the following dimensions:

### Original Framing
Most air quality ML projects focus on AQI regression or simple prediction notebooks.
This project reframes the problem as a **health advisory system** — the output is not
a number but a personalised action plan.

### Original Safety Architecture
The persona-based recommendation engine (5 personas x 6 AQI buckets = 30 unique advice
messages, plus 6 general tip lists) is an original design not derived from any existing
public project.

### Original Pollutant Explanation Layer
The dominant pollutant detection system uses exceedance ratios against CPCB thresholds
and delivers plain-language explanations of the pollutant chemistry. This bridges the
gap between ML output and citizen understanding.

### Agentic Design
The app behaves agentically: given inputs, it autonomously selects the dominant pollutant,
chooses the right persona advice, decides whether to call the live API, handles failure
gracefully, and delivers a composed, multi-layer response — without requiring the user to
know anything about air quality science.

### Dual-Mode Operation
The offline-first (ML-only) + online-optional (live API) architecture ensures the app
works in all connectivity conditions, including areas with unreliable internet.

---

## 14. Lean Canvas

| Section | Content |
|---------|---------|
| **Problem** | Citizens lack personalised, actionable air quality guidance. Existing tools show raw numbers without context or person-specific advice. |
| **Customer Segments** | Students, daily commuters, asthma and respiratory patients, elderly residents, outdoor and construction workers, urban households. |
| **Unique Value Proposition** | "Know your risk, protect your health" — AI predicts your personal air quality risk and tells you exactly what to do, in your language, for your life situation. |
| **Solution** | ML-based AQI prediction from sensor readings + dominant pollutant explanation + personalised advice per user type + optional live data cross-check. |
| **Channels** | Streamlit Cloud (free, shareable URL), embedding in city health portals, integration with school or hospital dashboards. |
| **Revenue / Public Good Model** | Free public good tier (Streamlit deployment). Premium tier: API for NGOs, municipal corporations, or hospitals. Grant funding via SDG-aligned programmes. |
| **Key Metrics** | Daily active users, prediction requests per day, live API usage rate, user persona distribution, number of cities queried. |
| **Cost Structure** | Zero marginal cost (serverless Streamlit). WAQI API free tier sufficient for low-to-medium traffic. Hosting cost only if scaling beyond free tier. |
| **Unfair Advantage** | Trained on authentic CPCB data. Open-source with full reproducibility. Multi-persona safety system not present in any competing free tool. |

---

## 15. Limitations

### Data Limitations
- Training data covers India only (26 cities, 2015-2020). Accuracy on other geographies
  or post-2020 pollution patterns is unvalidated.
- The model was trained on daily averages. It cannot account for hour-by-hour spikes
  (e.g., rush-hour peaks or crop burning events).
- Xylene was dropped due to high missingness. Its inclusion could marginally improve accuracy.

### Model Limitations
- 80% accuracy means approximately 1 in 5 predictions is wrong by at least one category.
  Users should treat predictions as guidance, not medical certainty.
- The model does not account for weather (temperature, humidity, wind speed) which
  significantly affects actual AQI. Including weather features is a priority improvement.
- The Random Forest is not explainable at the individual prediction level (only global
  feature importances are available). SHAP values would improve per-prediction transparency.

### Application Limitations
- The live AQI API requires a free but externally registered key. Users who do not obtain
  a key will not benefit from real-time cross-checking.
- The app does not store or log user queries, so there is no feedback loop to detect when
  the model systematically fails.

---

## 16. Future Scope

### Short Term (1-3 months)
- Add weather features (temperature, humidity, wind speed) from OpenWeather API
- Implement SHAP values for per-prediction feature attribution
- Add a CSV upload feature so users can paste real sensor logs
- Support more cities and countries beyond India

### Medium Term (3-6 months)
- Integrate hourly data from station_day.csv for intra-day predictions
- Add push notification system for real-time pollution alerts
- Build a mobile-responsive PWA version
- Create school and hospital-specific dashboard templates

### Long Term (6+ months)
- Fine-tune on continuously updated API data for a living model
- Integrate satellite remote sensing data (NASA MODIS AOD) for gridded predictions
- Build a community reporting feature where users can flag sensor anomalies
- Partner with municipal corporations for official deployment

---

## 17. References

1. Central Pollution Control Board (CPCB). National Air Quality Index. Government of India. https://cpcb.nic.in/
2. Kaggle. India Air Quality Dataset. https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
3. World Air Quality Index (WAQI) API. https://aqicn.org/api/
4. World Health Organization. Ambient Air Pollution: Health Impacts. 2023. https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health
5. United Nations. Sustainable Development Goals — SDG 3, 11, 13. https://sdgs.un.org/goals
6. Scikit-learn Developers. RandomForestClassifier Documentation. https://scikit-learn.org/
7. Streamlit Inc. Streamlit Documentation. https://docs.streamlit.io/
8. Python-dotenv. Environment variable management. https://pypi.org/project/python-dotenv/
9. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. JMLR, 2011.
10. Breiman, L. Random Forests. Machine Learning, 45(1):5-32, 2001.

---

*End of Report*

---

**Prepared by:** IBM Internship Project Team  
**Project Name:** Air Quality AQI Assistant  
**Technology Stack:** Python, Streamlit, scikit-learn, WAQI API  
**Repository:** github.com/YOUR_USERNAME/air-quality-aqi-assistant  
