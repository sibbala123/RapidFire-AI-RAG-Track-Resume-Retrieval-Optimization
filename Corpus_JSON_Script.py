import re
import json
import pandas as pd
from pathlib import Path
import html as html_lib

# -----------------------------
# Text cleaning helpers
# -----------------------------

def clean_text(text: str) -> str:
    """
    Cleans plain resume text:
    - fixes encoding issues
    - normalizes whitespace
    - keeps paragraph breaks (important for chunking)
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (if any)
    text = html_lib.unescape(text)

    # Fix common encoding junk seen in your dataset
    text = (
        text.replace("\u00a0", " ")
            .replace("Â", " ")
            .replace("ï¼", "-")
    )

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def strip_html(html: str) -> str:
    """
    Converts HTML resume into readable plain text.
    Used only as a fallback if Resume_str is missing.
    """
    if not isinstance(html, str):
        return ""

    html = html_lib.unescape(html)

    # Remove scripts/styles
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)

    # Preserve line breaks
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p>", "\n\n", html)

    # Remove all remaining tags
    html = re.sub(r"(?s)<.*?>", " ", html)

    # Normalize whitespace
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)

    return html.strip()


# -----------------------------
# Paths
# -----------------------------

CSV_PATH = Path("Resume.csv")             # <-- your input CSV
OUTPUT_PATH = Path("dataset/corpus.jsonl")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load CSV
# -----------------------------

df = pd.read_csv(CSV_PATH)
print("Loaded CSV with columns:", list(df.columns))

written = 0
skipped = 0

# -----------------------------
# Convert rows → JSONL
# -----------------------------

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        resume_id = str(row["ID"]).strip()
        if not resume_id:
            skipped += 1
            continue

        # Clean plain text resume
        text_plain = clean_text(row.get("Resume_str", ""))

        # Clean HTML resume (fallback)
        text_html = clean_text(strip_html(row.get("Resume_html", "")))

        # Prefer plain text, fallback to HTML
        final_text = text_plain if len(text_plain) >= 50 else text_html

        if len(final_text) < 50:
            skipped += 1
            continue

        record = {
            "doc_id": resume_id,
            "person_id": resume_id,
            "doc_type": "resume",
            "category": str(row.get("Category", "")).strip(),
            "text": final_text,
            "text_plain": text_plain,
            "text_html_stripped": text_html,
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

print(f"✅ Wrote {written} resumes to {OUTPUT_PATH}")
print(f"⚠️ Skipped {skipped} resumes (empty or too short)")
