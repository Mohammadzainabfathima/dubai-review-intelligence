"""Text pre-processing for mixed Arabic / English reviews.

Two cleaning levels are used on purpose:
  * "light"      - for transformer models (XLM-R etc.). Keeps word forms and
                   emojis because the sub-word tokenizer already handles them.
  * "aggressive" - for the TF-IDF baseline, keyword matching and topic terms.
                   Normalises Arabic letter variants so that spelling
                   differences map to one token.
Personal data (emails, phone numbers, URLs, @mentions) is masked in both
levels - this is part of the privacy mitigation described in the report.
"""
import re

# --- regular expressions -------------------------------------------------
AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+")
MENTION_RE = re.compile(r"@\w+")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s\-]{7,}\d(?!\w)")
# Arabic-Indic and Persian digits -> ASCII digits
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
REPEAT_LETTER = re.compile(r"([^\W\d_])\1{2,}")  # 3+ repeated letters
AR_LETTERS = re.compile(r"[؀-ۿݐ-ݿ]")
LAT_LETTERS = re.compile(r"[A-Za-z]")

# Splits a review into clauses at punctuation and contrast words so that
# "الخدمة كانت great بس الأكل cold" -> ["الخدمة كانت great", "الأكل cold"].
CLAUSE_SPLIT = re.compile(
    r"[.!?؟،,;:\n]+|\s(?:but|however|although|though|whereas|while|"
    r"بس|لكن|ولكن|لاكن|بينما|الا ان|إلا أن)\s",
    re.I,
)


def mask_pii(text: str) -> str:
    """Replace personal identifiers with neutral placeholders."""
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = PHONE_RE.sub(" <PHONE> ", text)
    text = MENTION_RE.sub("@user", text)
    return text


def clean_text(text, level: str = "light") -> str:
    """Clean one review. level is 'light' or 'aggressive'."""
    if text is None:
        return ""
    text = str(text)
    text = mask_pii(text)
    text = AR_DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = text.translate(DIGITS)

    if level == "aggressive":
        text = text.lower()
        text = re.sub("[إأآا]", "ا", text)
        text = text.replace("ى", "ي").replace("ة", "ه")
        text = text.replace("ؤ", "ء").replace("ئ", "ء")
        text = REPEAT_LETTER.sub(r"\1", text)
        text = re.sub(r"[^\w\s<>]", " ", text)
    else:
        text = REPEAT_LETTER.sub(r"\1\1", text)

    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """Script-ratio language detection: 'ar', 'en', 'mixed' or 'other'.

    Reviews in Arabic script only -> 'ar'; Latin only -> 'en'; a blend ->
    'mixed' (code-switching). Fast, dependency free and good enough because
    the two scripts are visually distinct.
    """
    text = mask_pii(str(text))
    ar = len(AR_LETTERS.findall(text))
    lat = len(LAT_LETTERS.findall(text))
    total = ar + lat
    if total == 0:
        return "other"
    ratio = ar / total
    if ratio >= 0.85:
        return "ar"
    if ratio <= 0.15:
        return "en"
    return "mixed"


def split_clauses(text: str, min_chars: int = 3):
    """Split a review into short clauses for aspect-level analysis."""
    parts = [p.strip() for p in CLAUSE_SPLIT.split(str(text)) if p and p.strip()]
    parts = [p for p in parts if len(p) >= min_chars]
    return parts or [str(text).strip()]


def preprocess_dataframe(df, text_col: str = "text"):
    """Add clean_text / clean_text_agg / lang columns and drop empty rows."""
    out = df.copy()
    out["clean_text"] = out[text_col].map(lambda t: clean_text(t, "light"))
    out["clean_text_agg"] = out[text_col].map(lambda t: clean_text(t, "aggressive"))
    out["lang"] = out[text_col].map(detect_language)
    out = out[out["clean_text"].str.len() > 2]
    out = out.drop_duplicates(subset="clean_text").reset_index(drop=True)
    return out
