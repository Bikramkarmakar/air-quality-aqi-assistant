"""
gemini_advisor.py
-----------------
Gemini AI integration for the Air Quality AQI Assistant.
Provides an intelligent conversational advisor that answers air quality
questions, interprets predictions, and gives context-aware health guidance.

The module gracefully degrades if the API key is missing or the API is
unavailable — the app continues working using the static recommendation engine.
"""

from __future__ import annotations

import os
from typing import Optional

# ── model to use ──────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-flash-lite-latest"

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are AQI Advisor, an expert AI assistant embedded in the
Air Quality AQI Assistant web application. Your role is to help users understand
air quality, interpret AQI predictions, and make safe decisions.

Guidelines:
- Keep answers concise (2-4 short paragraphs max) unless the user asks for detail
- Always use plain language — no jargon without explanation
- Tailor advice to the user's stated persona when known
- When AQI context is provided, reference it specifically
- Focus on practical, actionable health and safety guidance
- Align advice with SDG 3 (Health), SDG 11 (Sustainable Cities), SDG 13 (Climate)
- If a question is unrelated to air quality or health, politely redirect
- Never diagnose medical conditions — always recommend consulting a doctor for symptoms
- Use bullet points for lists of tips; keep tone friendly and supportive
"""


def _get_gemini_key() -> Optional[str]:
    """Load Gemini API key from Streamlit secrets or .env file."""
    # 1. Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    # 2. Environment / .env (local dev)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY", None)


def _build_client():
    """Return a configured Gemini client or None if key is unavailable."""
    key = _get_gemini_key()
    if not key:
        return None, "GEMINI_API_KEY not configured."
    try:
        from google import genai
        client = genai.Client(api_key=key)
        return client, None
    except ImportError:
        return None, "google-genai package not installed. Run: pip install google-genai"
    except Exception as exc:
        return None, f"Gemini client error: {exc}"


def ask_gemini(
    question: str,
    context: Optional[dict] = None,
    chat_history: Optional[list] = None,
) -> dict:
    """
    Send a question to Gemini with optional AQI context.

    Parameters
    ----------
    question      : The user's question string
    context       : Optional dict with keys like:
                      predicted_bucket, dominant_pollutant,
                      persona, live_aqi, city
    chat_history  : List of previous {"role": "user"/"model", "text": "..."} dicts

    Returns
    -------
    dict:
        success  : bool
        answer   : str   (Gemini response or fallback message)
        error    : str   (empty string if success)
    """
    client, err = _build_client()
    if client is None:
        return {"success": False, "answer": "", "error": err}

    # Build a context-enriched prompt
    full_prompt = _build_prompt(question, context, chat_history)

    try:
        from google import genai as _genai
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        answer = response.text.strip() if response.text else "No response generated."
        return {"success": True, "answer": answer, "error": ""}

    except Exception as exc:
        err_msg = str(exc)
        # Friendly user-facing messages for common errors
        if "503" in err_msg or "UNAVAILABLE" in err_msg:
            friendly = "Gemini is experiencing high demand right now. Please try again in a moment."
        elif "API_KEY" in err_msg or "401" in err_msg or "403" in err_msg:
            friendly = "Invalid or expired Gemini API key. Please check your GEMINI_API_KEY in .env."
        elif "quota" in err_msg.lower():
            friendly = "Gemini API quota exceeded. Please wait a bit or check your API plan."
        else:
            friendly = f"Gemini API error: {err_msg[:120]}"
        return {"success": False, "answer": "", "error": friendly}


def _build_prompt(
    question: str,
    context: Optional[dict],
    chat_history: Optional[list],
) -> str:
    """Assemble a full prompt with system instructions, context, history, and question."""
    parts = [SYSTEM_PROMPT, ""]

    # Inject current AQI context if available
    if context:
        ctx_lines = ["=== Current Air Quality Context ==="]
        if context.get("predicted_bucket"):
            ctx_lines.append(f"Predicted AQI Category : {context['predicted_bucket']}")
        if context.get("dominant_pollutant"):
            ctx_lines.append(f"Dominant Pollutant     : {context['dominant_pollutant']}")
        if context.get("persona"):
            ctx_lines.append(f"User Persona           : {context['persona']}")
        if context.get("live_aqi"):
            ctx_lines.append(f"Live AQI (real-time)   : {context['live_aqi']} ({context.get('city', '')})")
        if context.get("pollutant_values"):
            vals = context["pollutant_values"]
            top = sorted(vals.items(), key=lambda x: x[1], reverse=True)[:4]
            ctx_lines.append("Top Pollutant Readings : " +
                             ", ".join(f"{k}={v:.1f}" for k, v in top))
        ctx_lines.append("===================================")
        parts.append("\n".join(ctx_lines))
        parts.append("")

    # Include last few exchanges for conversational context
    if chat_history:
        parts.append("=== Conversation History ===")
        for turn in chat_history[-6:]:   # last 3 exchanges
            role  = "User" if turn["role"] == "user" else "AQI Advisor"
            parts.append(f"{role}: {turn['text']}")
        parts.append("===========================")
        parts.append("")

    parts.append(f"User: {question}")
    parts.append("AQI Advisor:")

    return "\n".join(parts)


def gemini_available() -> bool:
    """Quick check — returns True if Gemini key is present and client initialises."""
    client, _ = _build_client()
    return client is not None


def get_aqi_insight(
    predicted_bucket: str,
    dominant_pollutant: str,
    persona: str,
    pollutant_values: dict,
) -> str:
    """
    Generate a one-paragraph AI insight for the prediction result panel.
    Returns empty string if Gemini is unavailable.
    """
    question = (
        f"The AQI model just predicted '{predicted_bucket}' air quality. "
        f"The dominant pollutant is {dominant_pollutant}. "
        f"The user is a {persona}. "
        f"In 2-3 sentences, give a specific, practical insight tailored to this person "
        f"about what this prediction means for their day and one key action they should take."
    )
    context = {
        "predicted_bucket":  predicted_bucket,
        "dominant_pollutant": dominant_pollutant,
        "persona":           persona,
        "pollutant_values":  pollutant_values,
    }
    result = ask_gemini(question, context=context)
    return result["answer"] if result["success"] else ""
