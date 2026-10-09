"""
check_secrets.py
----------------
Pre-push security guard for the Air Quality AQI Assistant repo.

Scans every file that would be committed to GitHub and exits with
code 1 (blocking the push) if any real API key or secret pattern
is detected.

Usage:
    python check_secrets.py

Returns:
    exit 0  — clean, safe to push
    exit 1  — secret detected, push BLOCKED
"""

import os
import re
import sys

# ── Patterns that indicate a REAL secret ─────────────────────────────────────
# These are regex patterns. Placeholders like "your_key_here" are fine.
SECRET_PATTERNS = [
    # Google / Gemini API keys (real format: AIzaSy... OR AQ. prefixed)
    (r"AIzaSy[A-Za-z0-9_\-]{35}", "Google/Gemini API key (AIzaSy...)"),
    (r"AQ\.[A-Za-z0-9_\-]{40,}", "WAQI / Gemini API key (AQ.xxx...)"),
    # Generic patterns: key = "real_value" (not a placeholder)
    (r"""(?i)(api_key|apikey|api_token|secret_key|auth_token)\s*=\s*["'][A-Za-z0-9]{20,}["']""",
     "Hardcoded API key assignment"),
    # Long hex strings typical of WAQI tokens (40 char hex)
    (r"\b[0-9a-f]{40}\b", "40-char hex token (WAQI-style)"),
]

# ── Files / directories to skip ───────────────────────────────────────────────
SKIP_PATHS = {
    ".git", "__pycache__", ".env", ".env_backup_DO_NOT_COMMIT",
    "venv", ".venv", "node_modules",
}

# ── Extensions to scan ────────────────────────────────────────────────────────
SCAN_EXTENSIONS = {
    ".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml",
    ".cfg", ".ini", ".bat", ".sh", ".env.example",
}

# ── Known-safe placeholder strings (allowed even if they match shape) ─────────
PLACEHOLDER_FRAGMENTS = [
    "your_api_key_here", "your_waqi_key_here", "your_gemini_key_here",
    "your_actual_key_here", "your_key_here", "paste_your_token_here",
    "YOUR_USERNAME", "your_token_from_email",
]


def is_placeholder(line: str) -> bool:
    """Return True if the line contains only a placeholder, not a real key."""
    line_lower = line.lower()
    return any(p.lower() in line_lower for p in PLACEHOLDER_FRAGMENTS)


def scan_file(filepath: str) -> list[tuple[int, str, str]]:
    """
    Scan a single file for secret patterns.
    Returns list of (line_number, matched_text, pattern_description).
    """
    hits = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                if is_placeholder(line):
                    continue
                for pattern, description in SECRET_PATTERNS:
                    match = re.search(pattern, line)
                    if match:
                        hits.append((lineno, match.group(0), description))
    except (OSError, PermissionError):
        pass
    return hits


# Files that are allowed to contain key-shaped patterns (scanner itself, docs)
EXEMPT_FILES = {"check_secrets.py", "SECURITY.md"}


def collect_files(root: str) -> list[str]:
    """Walk the project tree and return files eligible for scanning."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_PATHS and not d.startswith(".")
        ]
        for fname in filenames:
            # Skip this scanner itself and exempt docs
            if fname in EXEMPT_FILES:
                continue
            # Skip hidden files and .env (never committed anyway)
            if fname.startswith(".") and fname not in {".env.example", ".gitattributes"}:
                continue
            ext = os.path.splitext(fname)[1].lower()
            # Include files with known extensions OR no extension
            if ext in SCAN_EXTENSIONS or ext == "":
                files.append(os.path.join(dirpath, fname))
    return files


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    print("=" * 60)
    print("  Air Quality AQI Assistant — Pre-Push Security Check")
    print("=" * 60)
    print(f"Scanning: {root}\n")

    files = collect_files(root)
    total_issues = 0

    for filepath in files:
        rel = os.path.relpath(filepath, root)
        hits = scan_file(filepath)
        if hits:
            print(f"[ALERT] {rel}")
            for lineno, text, desc in hits:
                # Redact middle of matched value for safe display
                redacted = text[:6] + "*" * max(0, len(text) - 10) + text[-4:] if len(text) > 12 else "***"
                print(f"        Line {lineno}: {desc}")
                print(f"        Matched: {redacted}")
            total_issues += len(hits)
            print()

    print("-" * 60)
    if total_issues == 0:
        print("RESULT: CLEAN — no secrets detected.")
        print("Safe to push to GitHub.")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"RESULT: BLOCKED — {total_issues} potential secret(s) found.")
        print("")
        print("ACTION REQUIRED:")
        print("  1. Remove or rotate the exposed key immediately.")
        print("  2. Move the key to your .env file.")
        print("  3. Run this script again to confirm it is clean.")
        print("  4. If a key was already pushed, revoke it NOW and generate a new one.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
