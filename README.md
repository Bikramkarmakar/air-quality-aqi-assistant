# 🌬️ Air Quality AQI Assistant

> An Agentic AI + Machine Learning web application that predicts air quality risk
> from pollutant sensor readings and delivers personalised health & safety advice.

[![SDG 3](https://img.shields.io/badge/SDG-3%20Good%20Health-blue)](https://sdgs.un.org/goals/goal3)
[![SDG 11](https://img.shields.io/badge/SDG-11%20Sustainable%20Cities-orange)](https://sdgs.un.org/goals/goal11)
[![SDG 13](https://img.shields.io/badge/SDG-13%20Climate%20Action-green)](https://sdgs.un.org/goals/goal13)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Quick Start — Run Locally](#quick-start--run-locally)
5. [API Key Setup (Live AQI)](#api-key-setup-live-aqi)
6. [Retrain the Model](#retrain-the-model)
7. [Push to GitHub](#push-to-github)
8. [Deploy on Streamlit Cloud](#deploy-on-streamlit-cloud)
9. [SDG Alignment](#sdg-alignment)
10. [Tech Stack](#tech-stack)
11. [License](#license)

---

## Project Overview

**Air Quality AQI Assistant** uses historical Indian city air quality data (2015–2020)
to train a Random Forest classifier that predicts the CPCB AQI category:
*Good → Satisfactory → Moderate → Poor → Very Poor → Severe*.

The app goes beyond prediction — it identifies the **dominant pollutant**, explains it
in plain language, and gives **persona-specific safety advice** for:
- Students, Commuters, Asthma Patients, Elderly People, Outdoor Workers, General Public

An optional **live AQI cross-check** via the [WAQI API](https://aqicn.org/api/) lets
users compare the model output with real-time sensor data for any city worldwide.

---

## Features

| Feature | Details |
|---------|---------|
| 🤖 ML Prediction | Random Forest, 80% accuracy, 6 AQI categories |
| 🧪 Pollutant Explanation | Plain-language description of dominant pollutant |
| 💡 Personalised Advice | 5 user personas with bucket-specific recommendations |
| 🌐 Live API Check | Real-time AQI from WAQI (needs free API key) |
| 📊 Data Explorer | Interactive charts for 26 Indian cities |
| 📈 Model Dashboard | Accuracy, F1, confusion data, feature importance |
| 🔐 Secure API Keys | .env locally, Streamlit Secrets in cloud — never hard-coded |

---

## Project Structure

```
Air Quality AQI Assistant/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for API key (copy to .env)
├── .gitignore                    # Prevents secrets and cache from being pushed
│
├── src/
│   ├── data_cleaning.py          # Data cleaning and validation pipeline
│   ├── train_model.py            # Model training script
│   ├── predictor.py              # Prediction service + live API handler
│   └── recommendations.py       # Health & safety recommendation engine
│
├── data/
│   ├── processed_city_day.csv    # Cleaned training dataset (auto-generated)
│   └── data_quality_report.txt  # Data quality summary (auto-generated)
│
├── models/
│   ├── aqi_model.pkl             # Trained Random Forest pipeline
│   ├── feature_names.pkl         # Feature list
│   ├── label_encoder.pkl         # LabelEncoder
│   └── model_metrics.json        # Evaluation metrics
│
├── reports/
│   └── project_report.md         # Full project report
│
├── .streamlit/
│   └── secrets.toml.example      # Streamlit secrets template
│
├── city_day.csv                  # Raw dataset (primary)
├── station_day.csv               # Raw dataset (supporting)
└── stations.csv                  # Station reference data
```

---

## Quick Start — Run Locally

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Clone or download the project

```bash
git clone https://github.com/YOUR_USERNAME/air-quality-aqi-assistant.git
cd air-quality-aqi-assistant
```

Or simply open the project folder you already have.

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Clean the data and train the model

```bash
python src/data_cleaning.py
python src/train_model.py
```

Expected output:
```
[INFO] Processed dataset saved -> data/processed_city_day.csv  (24850 rows)
[INFO] Model saved -> models/aqi_model.pkl
[RESULT] Test Accuracy : 0.8026
[CV]  5-fold CV accuracy: 0.7999 +/- 0.0062
[DONE] All artifacts saved successfully.
```

### Step 5 — Run the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## API Key Setup (Live AQI)

The live AQI cross-check is **optional**. The app works fully without it.

To enable it:

1. Get a **free API key** from [WAQI](https://aqicn.org/api/) (takes 1 minute)

2. Copy `.env.example` to `.env` in the project root:
   ```bash
   cp .env.example .env        # macOS/Linux
   copy .env.example .env      # Windows
   ```

3. Open `.env` and replace the placeholder:
   ```
   AQI_API_KEY=your_actual_key_here
   ```

4. The app auto-loads it. **Never commit `.env` to GitHub** — it is already in `.gitignore`.

---

## Retrain the Model

If you want to retrain with updated data or different parameters:

```bash
# Step 1: Clean data
python src/data_cleaning.py

# Step 2: Train
python src/train_model.py
```

The new model artifacts will overwrite files in `models/`.

---

## Push to GitHub

### First time setup

```bash
# Initialise git (if not already done)
git init

# Add remote (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/air-quality-aqi-assistant.git

# Stage all files (secrets are protected by .gitignore)
git add .

# Verify .env is NOT staged
git status
# You should NOT see .env in the list

# Commit
git commit -m "Initial commit: Air Quality AQI Assistant"

# Push
git push -u origin main
```

### What gets pushed

- ✅ `app.py`, `src/`, `models/*.pkl`, `data/processed_city_day.csv`
- ✅ `requirements.txt`, `.env.example`, `.gitignore`, `README.md`
- ✅ `city_day.csv`, `station_day.csv`, `stations.csv`
- ❌ `.env` (blocked by `.gitignore`)
- ❌ `.streamlit/secrets.toml` (blocked by `.gitignore`)

---

## Deploy on Streamlit Cloud

### Step 1 — Push to GitHub (see above)

### Step 2 — Create a Streamlit Cloud account

Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

### Step 3 — Deploy the app

1. Click **"New app"**
2. Select your repository and branch
3. Set **Main file path** to `app.py`
4. Click **"Deploy"**

### Step 4 — Add your API key as a Secret

1. In Streamlit Cloud, go to your app → **⋮ menu → Settings → Secrets**
2. Add:
   ```toml
   AQI_API_KEY = "your_actual_key_here"
   ```
3. Save. The app will automatically use it.

> **Note**: The `models/*.pkl` files must be present in the repository for the app to
> load the trained model. If the files are too large (>100 MB), consider using
> `git-lfs` or regenerating them in a Streamlit startup script.

---

## SDG Alignment

| Goal | Connection |
|------|-----------|
| **SDG 3** — Good Health & Well-being | Personalised health alerts protect vulnerable populations from air pollution |
| **SDG 11** — Sustainable Cities & Communities | Real-time city-level air quality awareness for safer urban living |
| **SDG 13** — Climate Action | Highlights pollution trends driven by industrial and transport emissions |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML Model | scikit-learn — Random Forest |
| Data Processing | pandas, NumPy |
| Live API | WAQI Air Quality API |
| Secret Management | python-dotenv (local), Streamlit Secrets (cloud) |
| Language | Python 3.10+ |

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built as part of an IBM AI Internship project · Targeting SDG 3, 11, 13 · 2024*
