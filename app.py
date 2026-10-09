"""
app.py  —  Air Quality AQI Assistant
--------------------------------------
Streamlit web application — fully upgraded with Gemini AI chat.

Tabs:
  1. Predict & Advise   — ML prediction + live AQI + Gemini AI insight
  2. AI Chat            — Full conversational Gemini advisor
  3. Data Explorer      — Interactive dataset charts
  4. Model Info         — Performance metrics
  5. About & SDGs       — Project info, API setup

Run locally:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import datetime
import streamlit as st
import pandas as pd
import numpy as np

from predictor import predict, fetch_live_aqi, get_feature_names, get_metrics
from recommendations import (
    get_recommendations,
    BUCKET_META,
    BUCKET_ORDER,
    POLLUTANT_DESCRIPTIONS,
)
from gemini_advisor import ask_gemini, gemini_available, get_aqi_insight

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality AQI Assistant",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .bucket-banner {
    text-align: center;
    padding: 1.4rem 1rem;
    border-radius: 14px;
    margin: 1rem 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.5px;
  }
  .advice-card {
    background: #f0f7ff;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 0.4rem 0;
    border-left: 5px solid #3b82f6;
    font-size: 0.95rem;
  }
  .gemini-card {
    background: linear-gradient(135deg, #e8f5e9 0%, #e3f2fd 100%);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.6rem 0;
    border-left: 5px solid #4caf50;
    font-size: 0.95rem;
  }
  .warning-box {
    background: #fff8e1;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    border-left: 5px solid #ffc107;
    margin: 0.4rem 0;
  }
  .live-box {
    background: #e8f5e9;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    border-left: 5px solid #4caf50;
    margin: 0.4rem 0;
  }
  .error-box {
    background: #fff5f5;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    border-left: 5px solid #f44336;
    margin: 0.4rem 0;
  }
  .chat-user {
    background: #e3f2fd;
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 0.4rem auto;
    max-width: 80%;
    font-size: 0.93rem;
    text-align: right;
  }
  .chat-ai {
    background: #f1f8e9;
    border-radius: 2px 12px 12px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem auto 0.4rem 0;
    max-width: 85%;
    font-size: 0.93rem;
    border-left: 3px solid #66bb6a;
  }
  .sdg-pill {
    display: inline-block;
    background: #1565c0;
    color: white;
    padding: 0.15rem 0.65rem;
    border-radius: 20px;
    font-size: 0.76rem;
    margin-right: 4px;
    font-weight: 600;
  }
  .gemini-badge {
    display: inline-block;
    background: #1a73e8;
    color: white;
    padding: 0.15rem 0.65rem;
    border-radius: 20px;
    font-size: 0.76rem;
    margin-left: 6px;
    font-weight: 600;
  }
  h1 { color: #1a237e; }
  h3 { color: #283593; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def bucket_color(bucket: str) -> str:
    return BUCKET_META.get(bucket, {}).get("color", "#888888")


def render_bucket_banner(bucket: str, emoji: str, health_msg: str) -> None:
    color = bucket_color(bucket)
    st.markdown(
        f'<div class="bucket-banner" style="background:{color};">'
        f'{emoji} &nbsp; {bucket.upper()} &nbsp; {emoji}<br>'
        f'<span style="font-size:0.9rem;font-weight:400;">{health_msg}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_prob_bar(probs: dict) -> None:
    ordered = [(b, probs.get(b, 0.0)) for b in BUCKET_ORDER]
    bar_html = '<div style="margin:0.4rem 0;">'
    for label, val in ordered:
        color = bucket_color(label)
        width = max(val * 90, 1)
        bar_html += (
            f'<div style="display:flex;align-items:center;margin:3px 0;">'
            f'<span style="width:115px;font-size:0.82rem;flex-shrink:0;">{label}</span>'
            f'<div style="flex:1;background:#f0f0f0;border-radius:4px;height:14px;">'
            f'<div style="width:{width:.1f}%;background:{color};height:14px;'
            f'border-radius:4px;min-width:4px;transition:width 0.3s;"></div></div>'
            f'<span style="margin-left:8px;font-size:0.78rem;width:42px;">{val*100:.1f}%</span>'
            f'</div>'
        )
    bar_html += '</div>'
    st.markdown(bar_html, unsafe_allow_html=True)


# ── session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_live" not in st.session_state:
    st.session_state.last_live = None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌬️ AQI Assistant")
    st.markdown(
        '<span class="sdg-pill">SDG 3</span>'
        '<span class="sdg-pill">SDG 11</span>'
        '<span class="sdg-pill">SDG 13</span>'
        + (' <span class="gemini-badge">✨ Gemini AI</span>' if gemini_available() else ''),
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 👤 Your Profile")
    persona = st.selectbox(
        "Who are you?",
        ["General Public", "Student", "Commuter",
         "Asthma Patient", "Elderly Person", "Outdoor Worker"],
        help="Personalised advice and AI responses will be tailored to your profile.",
    )

    st.markdown("---")
    st.markdown("### 🌍 Live AQI Check")
    city_name = st.text_input(
        "City name",
        value="Delhi",
        help="Fetches real-time AQI from WAQI API.",
    )
    check_live_btn = st.button("🔄 Fetch Live AQI", use_container_width=True)

    st.markdown("---")
    # Gemini status in sidebar
    if gemini_available():
        st.success("✨ **Gemini AI** is active\nUse the AI Chat tab to ask anything about air quality.")
    else:
        st.warning("⚠️ Gemini AI not configured.\nAdd `GEMINI_API_KEY` to your `.env` file.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "AI-powered assistant trained on Indian city data (2015–2020). "
        "26 cities · Random Forest · 80% accuracy · 6 AQI classes."
    )


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🌬️ Air Quality AQI Assistant")
st.markdown(
    "Enter pollutant readings to **predict air quality risk**, get **personalised safety advice**, "
    "and chat with your **Gemini AI advisor**."
)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_predict, tab_chat, tab_explore, tab_model, tab_about = st.tabs([
    "🔬 Predict & Advise",
    "✨ AI Chat",
    "📊 Data Explorer",
    "📈 Model Info",
    "📋 About & SDGs",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT & ADVISE
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### 🏭 Enter Pollutant Readings")
    st.caption("All values in µg/m³ except CO (mg/m³). Use defaults for a typical urban day.")

    c1, c2, c3 = st.columns(3)
    with c1:
        pm25    = st.number_input("PM2.5 (µg/m³)",  0.0, 600.0, 45.0,  0.5)
        pm10    = st.number_input("PM10 (µg/m³)",   0.0, 700.0, 80.0,  0.5)
        no      = st.number_input("NO (µg/m³)",     0.0, 200.0, 10.0,  0.5)
        no2     = st.number_input("NO2 (µg/m³)",    0.0, 250.0, 30.0,  0.5)
    with c2:
        nox     = st.number_input("NOx (µg/m³)",    0.0, 300.0, 40.0,  0.5)
        nh3     = st.number_input("NH3 (µg/m³)",    0.0, 500.0, 15.0,  0.5)
        co      = st.number_input("CO (mg/m³)",     0.0, 150.0, 1.0,   0.1)
        so2     = st.number_input("SO2 (µg/m³)",    0.0, 200.0, 12.0,  0.5)
    with c3:
        o3      = st.number_input("O3 (µg/m³)",     0.0, 200.0, 40.0,  0.5)
        benzene = st.number_input("Benzene (µg/m³)",0.0, 500.0, 2.0,   0.1)
        toluene = st.number_input("Toluene (µg/m³)",0.0, 500.0, 8.0,   0.1)
        today   = datetime.date.today()
        month   = st.number_input("Month", 1, 12, today.month)

    day_of_year = datetime.date(today.year, int(month), 15).timetuple().tm_yday

    st.markdown("---")
    predict_btn = st.button("🔍 Analyse Air Quality", type="primary", use_container_width=True)

    if predict_btn:
        input_vals = {
            "PM2.5":    pm25,  "PM10":    pm10,  "NO":      no,
            "NO2":      no2,   "NOx":     nox,   "NH3":     nh3,
            "CO":       co,    "SO2":     so2,   "O3":      o3,
            "Benzene":  benzene, "Toluene": toluene,
            "Month":    float(month), "DayOfYear": float(day_of_year),
        }

        with st.spinner("Analysing with AI model…"):
            result = predict(input_vals)

        bucket    = result["predicted_bucket"]
        probs     = result["probabilities"]
        top_feats = result["top_features"]
        reco      = get_recommendations(bucket, persona, input_vals)

        # Store for AI Chat context
        st.session_state.last_prediction = {
            "predicted_bucket":   bucket,
            "dominant_pollutant": reco["dominant_pollutant"],
            "persona":            persona,
            "pollutant_values":   input_vals,
        }

        # ── Banner ────────────────────────────────────────────────────────────
        st.markdown("## 🔎 Prediction Result")
        render_bucket_banner(bucket, reco["emoji"], reco["health_msg"])

        col1, col2, col3 = st.columns(3)
        col1.metric("AQI Category",   bucket)
        col2.metric("AQI Range",      reco["aqi_range"])
        col3.metric("Main Pollutant", reco["dominant_pollutant"])

        # ── Confidence bars ───────────────────────────────────────────────────
        st.markdown("#### Model Confidence")
        render_prob_bar(probs)

        st.markdown("---")

        # ── Gemini AI Insight ─────────────────────────────────────────────────
        if gemini_available():
            with st.spinner("✨ Getting AI insight from Gemini…"):
                ai_insight = get_aqi_insight(
                    bucket, reco["dominant_pollutant"], persona, input_vals
                )
            if ai_insight:
                st.markdown("### ✨ Gemini AI Insight")
                st.markdown(
                    f'<div class="gemini-card">🤖 <b>Gemini AI says:</b><br><br>{ai_insight}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("💬 Want more? Switch to the **AI Chat** tab to ask follow-up questions.")

        # ── Pollutant explanation ─────────────────────────────────────────────
        st.markdown("### 🧪 Main Pollution Source")
        st.markdown(
            f'<div class="advice-card"><b>{reco["dominant_pollutant"]}</b> — '
            f'{reco["pollutant_desc"]}</div>',
            unsafe_allow_html=True,
        )

        # ── Feature importance ────────────────────────────────────────────────
        st.markdown("#### Top Contributing Features")
        fi_df = pd.DataFrame(
            {"Feature": [f for f, _ in top_feats],
             "Importance (%)": [round(v * 100, 2) for _, v in top_feats]}
        ).sort_values("Importance (%)", ascending=True)
        st.bar_chart(fi_df.set_index("Feature"), height=220)

        st.markdown("---")

        # ── Persona advice ────────────────────────────────────────────────────
        st.markdown(f"### 💡 Advice for {persona}")
        st.markdown(
            f'<div class="advice-card">🧑 <b>{persona}</b>: {reco["persona_advice"]}</div>',
            unsafe_allow_html=True,
        )

        # ── General tips ──────────────────────────────────────────────────────
        st.markdown("### ✅ General Safety Tips")
        for tip in reco["general_tips"]:
            st.markdown(f"- {tip}")

        # ── Live AQI ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🌐 Live AQI Cross-Check")
        if city_name.strip():
            with st.spinner(f"Fetching live AQI for {city_name}…"):
                live = fetch_live_aqi(city_name.strip())
            st.session_state.last_live = live

            if live["success"]:
                live_bucket = live["live_bucket"]
                live_color  = bucket_color(live_bucket)
                match_icon  = "✅ Match" if live_bucket == bucket else "⚠️ Differs from input"
                st.markdown(
                    f'<div class="live-box">'
                    f'🌐 <b>Live Data</b> · <i>{live["station"]}</i><br>'
                    f'Real-Time AQI: <b>{live["live_aqi"]}</b> → '
                    f'<span style="color:{live_color};font-weight:700;">{live_bucket}</span>'
                    f' &nbsp; {match_icon}</div>',
                    unsafe_allow_html=True,
                )
                if live_bucket != bucket:
                    st.markdown(
                        '<div class="warning-box">⚠️ The live reading differs from your manual '
                        'input. Update the pollutant values above with current sensor readings '
                        'for a more accurate prediction.</div>',
                        unsafe_allow_html=True,
                    )
                # Update context with live data
                if st.session_state.last_prediction:
                    st.session_state.last_prediction["live_aqi"]  = live["live_aqi"]
                    st.session_state.last_prediction["city"]      = city_name
            else:
                st.markdown(
                    f'<div class="warning-box">ℹ️ {live["message"]}<br>'
                    f'The ML model prediction above is still valid.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Enter a city name in the sidebar to enable live AQI cross-check.")

    # Live-only check from sidebar button (outside predict block)
    if check_live_btn and not predict_btn:
        if city_name.strip():
            with st.spinner(f"Fetching live AQI for {city_name}…"):
                live = fetch_live_aqi(city_name.strip())
            if live["success"]:
                live_bucket = live["live_bucket"]
                live_color  = bucket_color(live_bucket)
                st.markdown(
                    f'<div class="live-box">'
                    f'🌐 <b>Live AQI · {city_name}</b><br>'
                    f'AQI: <b>{live["live_aqi"]}</b> → '
                    f'<span style="color:{live_color};font-weight:700;">{live_bucket}</span><br>'
                    f'<small>Source: {live["station"]}</small></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning(live["message"])
        else:
            st.info("Enter a city name in the sidebar first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEMINI AI CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### ✨ Gemini AI Air Quality Advisor")

    if not gemini_available():
        st.error(
            "**Gemini AI is not configured.**\n\n"
            "Add your Gemini API key to `.env`:\n```\nGEMINI_API_KEY=your_key_here\n```\n"
            "Then restart the app. Get a free key at https://aistudio.google.com/app/apikey"
        )
    else:
        st.markdown(
            "Ask anything about air quality, health effects, pollution sources, "
            "or safety precautions. The AI is aware of your latest prediction."
        )

        # Show context badge if prediction exists
        pred_ctx = st.session_state.last_prediction
        if pred_ctx:
            bucket_c = bucket_color(pred_ctx["predicted_bucket"])
            st.markdown(
                f'<div style="background:#f5f5f5;border-radius:8px;padding:0.5rem 1rem;'
                f'margin-bottom:0.8rem;font-size:0.85rem;">'
                f'🔗 Context: Prediction = '
                f'<span style="color:{bucket_c};font-weight:700;">{pred_ctx["predicted_bucket"]}</span>'
                f' · Pollutant: <b>{pred_ctx["dominant_pollutant"]}</b>'
                f' · Persona: <b>{pred_ctx["persona"]}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    '<div class="chat-ai">👋 Hi! I\'m your AQI Advisor powered by Gemini AI. '
                    'Ask me anything about air quality, pollution, or health protection. '
                    'I\'m aware of your latest prediction if you\'ve made one!</div>',
                    unsafe_allow_html=True,
                )
            for turn in st.session_state.chat_history:
                if turn["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user">👤 {turn["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Render markdown in AI responses safely
                    st.markdown(f'<div class="chat-ai">🤖 ', unsafe_allow_html=True)
                    st.markdown(turn["text"])
                    st.markdown('</div>', unsafe_allow_html=True)

        # Input row
        st.markdown("---")
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask your question",
                key="chat_input",
                placeholder="e.g. What does PM2.5 do to children's lungs? Is it safe to run outside?",
                label_visibility="collapsed",
            )
        with col_send:
            send_btn = st.button("Send 🚀", type="primary", use_container_width=True)

        # Quick-question chips
        st.markdown("**Quick questions:**")
        qcols = st.columns(4)
        quick_questions = [
            "What is AQI?",
            "How to protect myself from PM2.5?",
            "What causes air pollution in cities?",
            "When is it safe to exercise outdoors?",
        ]
        clicked_quick = None
        for i, (qcol, qq) in enumerate(zip(qcols, quick_questions)):
            if qcol.button(qq, key=f"qq_{i}", use_container_width=True):
                clicked_quick = qq

        # Decide what to send
        final_question = None
        if send_btn and user_input.strip():
            final_question = user_input.strip()
        elif clicked_quick:
            final_question = clicked_quick

        if final_question:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "text": final_question})

            # Build history for Gemini (exclude the just-added turn)
            history_for_gemini = st.session_state.chat_history[:-1]

            with st.spinner("✨ Gemini is thinking…"):
                response = ask_gemini(
                    final_question,
                    context=st.session_state.last_prediction,
                    chat_history=history_for_gemini,
                )

            if response["success"]:
                st.session_state.chat_history.append(
                    {"role": "model", "text": response["answer"]}
                )
            else:
                st.session_state.chat_history.append(
                    {"role": "model", "text": f"⚠️ {response['error']}"}
                )
            st.rerun()

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### 📊 Dataset Explorer")
    st.markdown("Training dataset — 26 Indian cities, 2015–2020.")

    @st.cache_data
    def load_processed():
        try:
            return pd.read_csv("data/processed_city_day.csv", parse_dates=["Date"])
        except FileNotFoundError:
            return None

    df_exp = load_processed()

    if df_exp is None:
        st.warning("Run `python src/data_cleaning.py` first to generate the processed dataset.")
    else:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            city_sel = st.selectbox(
                "Select City", ["All"] + sorted(df_exp["City"].unique().tolist())
            )
        with col_e2:
            bucket_sel = st.multiselect(
                "Filter by AQI Category", BUCKET_ORDER, default=BUCKET_ORDER
            )

        df_view = df_exp.copy()
        if city_sel != "All":
            df_view = df_view[df_view["City"] == city_sel]
        if bucket_sel:
            df_view = df_view[df_view["AQI_Bucket"].isin(bucket_sel)]

        st.markdown(f"**{len(df_view):,} records** shown")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### AQI Category Distribution")
            dist = df_view["AQI_Bucket"].value_counts().reindex(BUCKET_ORDER, fill_value=0)
            st.bar_chart(dist, height=260)

        with col_c2:
            st.markdown("#### AQI Trend Over Time")
            aqi_time = df_view.groupby("Date")["AQI"].mean().dropna()
            if len(aqi_time) > 0:
                st.line_chart(aqi_time, height=260)
            else:
                st.info("No date data for current filter.")

        st.markdown("#### PM2.5 vs AQI")
        sample = (
            df_view[["PM2.5", "AQI"]].dropna()
            .sample(min(1000, len(df_view)), random_state=1)
        )
        st.scatter_chart(sample, x="PM2.5", y="AQI", height=280)

        st.markdown("#### City-wise Average AQI")
        city_aqi = (
            df_view.groupby("City")["AQI"].mean()
            .sort_values(ascending=False)
            .round(1)
        )
        st.bar_chart(city_aqi, height=300)

        with st.expander("📋 Raw Data Table (first 200 rows)"):
            st.dataframe(df_view.head(200))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### 📈 Model Performance")

    try:
        metrics = get_metrics()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Test Accuracy",  f"{metrics['accuracy']*100:.2f}%")
        col_m2.metric("Weighted F1",    f"{metrics['f1_weighted']*100:.2f}%")
        col_m3.metric("Macro F1",       f"{metrics['f1_macro']*100:.2f}%")
        col_m4.metric("5-Fold CV",      f"{metrics.get('cv_accuracy_mean',0)*100:.2f}%")

        st.markdown("#### Per-Class Performance")
        rows = []
        for b in BUCKET_ORDER:
            if b in metrics.get("class_report", {}):
                r = metrics["class_report"][b]
                rows.append({
                    "Category":  b,
                    "Precision": f"{r['precision']:.3f}",
                    "Recall":    f"{r['recall']:.3f}",
                    "F1-Score":  f"{r['f1-score']:.3f}",
                    "Support":   int(r["support"]),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)

        st.markdown("#### Feature Importances")
        fi_list = metrics.get("feature_importances", [])
        fi_df   = pd.DataFrame(fi_list).sort_values("importance", ascending=False)
        fi_df["importance_pct"] = (fi_df["importance"] * 100).round(2)
        st.bar_chart(fi_df.set_index("feature")["importance_pct"], height=300)

        st.markdown("#### Model Details")
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Trees | 200 |
| Max Depth | 18 |
| Class Weighting | Balanced |
| Train / Test Split | 80% / 20% |
| Target | AQI_Bucket (6 classes) |
| Training Rows | 24,850 |
| Imputation | City-level median |
| Scaling | StandardScaler |
| AI Advisor | Gemini flash-lite-latest |
""")

    except Exception as e:
        st.error(f"Could not load metrics: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT & SDGs
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### 📋 About This Application")
    st.markdown("""
**Air Quality AQI Assistant** is an agentic AI + machine learning web app that predicts
air quality risk from pollutant sensor readings and provides personalised health &
safety recommendations, now enhanced with **Gemini AI** for intelligent conversation.

**Features:**
- 🤖 ML-based AQI category prediction (6 classes, 80% accuracy)
- ✨ Gemini AI advisor — ask anything about air quality in natural language
- 🧪 Dominant pollutant identification with plain-language explanations
- 💡 Personalised safety advice for 5 user personas
- 🌐 Live AQI cross-check via WAQI API
- 📊 Interactive data explorer for 26 Indian cities (2015–2020)
""")

    st.markdown("### 🌍 SDG Alignment")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div style="background:#1565c0;color:white;padding:1rem;
border-radius:10px;text-align:center;"><h4 style="color:white;">SDG 3</h4>
<b>Good Health & Well-being</b><br><br>
Personalised health alerts protect vulnerable people from air pollution.
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div style="background:#e65100;color:white;padding:1rem;
border-radius:10px;text-align:center;"><h4 style="color:white;">SDG 11</h4>
<b>Sustainable Cities</b><br><br>
Real-time city-level awareness helps urban residents make safer daily decisions.
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div style="background:#2e7d32;color:white;padding:1rem;
border-radius:10px;text-align:center;"><h4 style="color:white;">SDG 13</h4>
<b>Climate Action</b><br><br>
Tracks pollution trends linked to industrial emissions and climate change.
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔑 API Key Setup")
    st.markdown("""
Your `.env` file (in the project root) should contain:

```
AQI_API_KEY=your_waqi_key_here
GEMINI_API_KEY=your_gemini_key_here
```

| Key | Where to get it | Used for |
|-----|----------------|---------|
| `AQI_API_KEY` | [aqicn.org/api](https://aqicn.org/api/) — free | Live real-time AQI |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) — free | AI Chat advisor |

For **Streamlit Cloud** deployment: add both keys in **Settings → Secrets**.

⚠️ Never commit `.env` to GitHub — it is already protected by `.gitignore`.
""")

    st.markdown("### 🗃️ Data Sources")
    st.markdown("""
- **Primary**: `city_day.csv` — Daily city air quality (CPCB India / Kaggle)
- **Period**: January 2015 – July 2020 · **26 cities** · 11 pollutants
- **Supporting**: `station_day.csv`, `stations.csv` — station reference data
""")

    st.markdown("---")
    st.caption(
        "Air Quality AQI Assistant · ML + Gemini AI · "
        "Built with Streamlit & scikit-learn · SDG 3 · 11 · 13"
    )
