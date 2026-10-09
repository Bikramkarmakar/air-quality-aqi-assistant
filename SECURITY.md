# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest (main branch) | ✅ Yes |

---

## Reporting a Vulnerability

If you discover a security vulnerability in this project, **please do not open a public GitHub issue**.

Instead, report it privately by emailing the repository owner directly (see GitHub profile).

We will acknowledge the report within 48 hours and aim to release a fix within 7 days for critical issues.

---

## Secret and API Key Handling

This project uses two external API keys. **Neither key is hard-coded anywhere in the source code.**

| Key | Purpose | Where it must live |
|-----|---------|-------------------|
| `AQI_API_KEY` | Live AQI from WAQI API | `.env` (local) · Streamlit Secrets (cloud) |
| `GEMINI_API_KEY` | Gemini AI advisor | `.env` (local) · Streamlit Secrets (cloud) |

### Rules enforced in this repo

1. **`.env` is blocked by `.gitignore`** — the real keys file can never be staged or pushed.
2. **`.env.example` contains only placeholders** — safe to commit, shows required variable names only.
3. **`.streamlit/secrets.toml` is blocked by `.gitignore`** — Streamlit's local secret store never leaves your machine.
4. **No API key appears in any `.py`, `.md`, `.json`, `.toml`, or `.bat` file** — verified by pre-push audit.
5. The app **gracefully degrades** if either key is absent — it continues working with the ML model only.

### How keys are loaded in code

Both modules (`src/predictor.py` and `src/gemini_advisor.py`) use this two-stage pattern:

```python
# 1. Streamlit Secrets (cloud deployment via st.secrets)
key = st.secrets.get("KEY_NAME", None)

# 2. Environment variable loaded from .env (local development)
from dotenv import load_dotenv
load_dotenv()
key = os.environ.get("KEY_NAME", None)
```

This pattern means:
- In **local development**: keys come from `.env` via `python-dotenv`
- In **Streamlit Cloud**: keys come from the Secrets panel, never from files
- If neither is set: the feature is disabled with a user-friendly message

---

## What to do before pushing to GitHub

Run the pre-push security check included in this repo:

```bash
python check_secrets.py
```

This script scans all tracked files for patterns matching real API keys and blocks the push if any are found.

---

## What NOT to do

- ❌ Never copy your `.env` file into the project folder with a different name and commit it
- ❌ Never paste an API key into `app.py`, `README.md`, or any other tracked file
- ❌ Never commit `secrets.toml` — this file is for local Streamlit testing only
- ❌ Never log API keys to console output or write them to log files
- ❌ Never share your `.env` file via email, Slack, or chat

---

## Dependencies

This project depends on the following third-party packages. Keep them updated to avoid known vulnerabilities:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `scikit-learn` | ML model |
| `pandas` / `numpy` | Data processing |
| `requests` | HTTP calls to WAQI API |
| `python-dotenv` | Secure local env loading |
| `google-genai` | Gemini AI API client |

Run `pip install --upgrade -r requirements.txt` periodically to stay current.
