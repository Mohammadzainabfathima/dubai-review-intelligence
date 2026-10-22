"""Lexicon-guided aspect-based sentiment analysis (ABSA).

Idea: split every review into clauses, find which business aspect(s) a
clause talks about using a bilingual keyword lexicon, then score the clause
with the sentiment model. This gives "service: positive, food: negative"
for ONE review without needing aspect-labelled training data, which does not
exist for Arabic/English code-switched text.

Limitation (discussed in the report): keyword matching misses implicit
aspects ("we waited an hour" = service). Extending the lexicon or using a
zero-shot NLI classifier are the natural next steps.
"""
import pandas as pd

from .preprocess import clean_text, split_clauses, detect_language

_RAW_LEXICON = {
    "service": [
        "service", "services", "waiting", "waited", "slow", "reception",
        "check-in", "checkin", "checkout", "check-out",
        "خدمة", "الخدمة", "استقبال", "انتظار", "بطيء", "بطيئه", "تعامل",
    ],
    "staff": [
        "staff", "employee", "employees", "manager", "waiter", "waitress",
        "receptionist", "rude", "friendly", "helpful",
        "موظف", "موظفين", "موظفات", "طاقم", "مدير", "نادل", "عامل", "عمال",
        "ودود", "وقح",
    ],
    "food": [
        "food", "breakfast", "dinner", "lunch", "buffet", "meal", "taste",
        "restaurant", "menu", "cold", "dish",
        "اكل", "الاكل", "طعام", "الطعام", "وجبة", "وجبات", "فطور", "الفطور",
        "عشاء", "غداء", "بوفيه", "طعم", "مطعم", "بارد",
    ],
    "price": [
        "price", "prices", "expensive", "cheap", "overpriced", "value",
        "cost", "money", "worth", "fees",
        "سعر", "اسعار", "الاسعار", "غالي", "غاليه", "رخيص", "تكلفة", "قيمة",
        "مبالغ",
    ],
    "cleanliness": [
        "clean", "dirty", "smell", "smelly", "dust", "stain", "hygiene",
        "cleanliness", "mold", "cockroach",
        "نظافة", "النظافة", "نظيف", "نظيفة", "وسخ", "متسخ", "قذر", "رائحة",
        "روائح", "غبار",
    ],
    "room": [
        "room", "rooms", "bed", "bathroom", "shower", "view", "balcony",
        "pillow", "suite",
        "غرفة", "الغرفة", "غرف", "سرير", "حمام", "اطلالة", "بلكونة", "جناح",
    ],
    "location": [
        "location", "near", "close", "far", "metro", "walking", "marina",
        "downtown", "beach", "mall",
        "موقع", "الموقع", "قريب", "بعيد", "مترو", "شاطئ", "مول", "وسط",
    ],
    "facilities": [
        "pool", "gym", "wifi", "wi-fi", "internet", "parking", "elevator",
        "lift", "ac", "aircon", "air-conditioning", "spa", "noise", "noisy",
        "مسبح", "المسبح", "جيم", "نادي", "واي", "انترنت", "موقف", "مواقف",
        "مصعد", "مكيف", "تكييف", "سبا", "ضوضاء", "ازعاج",
    ],
}

ARABIC_PREFIXES = ("وال", "بال", "فال", "كال", "ال", "لل", "و", "ب", "ل", "ف")


def _norm(word: str) -> str:
    return clean_text(word, "aggressive")


# Build a lookup {normalised keyword -> aspect}
KEYWORD_TO_ASPECT = {}
for _aspect, _words in _RAW_LEXICON.items():
    for _w in _words:
        KEYWORD_TO_ASPECT.setdefault(_norm(_w).replace(" ", ""), _aspect)
ASPECTS = list(_RAW_LEXICON.keys())


def _variants(token: str):
    """Yield the token plus forms with Arabic prefixes / English plural removed."""
    yield token
    for p in ARABIC_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            yield token[len(p):]
    if token.endswith("s") and len(token) > 3:
        yield token[:-1]


def find_aspects(clause: str):
    """Return the set of aspects mentioned in a clause."""
    found = set()
    for token in _norm(clause).split():
        for v in _variants(token):
            if v in KEYWORD_TO_ASPECT:
                found.add(KEYWORD_TO_ASPECT[v])
                break
    return found


def extract_aspect_sentiments(df, sentiment_fn, text_col="text", id_col="review_id"):
    """Return one row per (review, aspect) with the clause and its sentiment.

    sentiment_fn: callable(list[str]) -> list[(label_id, confidence)] where
    label ids are 0=negative, 1=neutral, 2=positive.
    Reviews in which no clause matches any aspect are kept under the
    'overall' aspect using the whole review text.
    """
    records = []
    for rid, text in zip(df[id_col], df[text_col]):
        matched_any = False
        for clause in split_clauses(clean_text(text, "light")):
            for aspect in find_aspects(clause):
                records.append((rid, aspect, clause))
                matched_any = True
        if not matched_any:
            records.append((rid, "overall", clean_text(text, "light")))

    out = pd.DataFrame(records, columns=[id_col, "aspect", "clause"])
    if out.empty:
        out["sentiment"], out["confidence"], out["lang"] = [], [], []
        return out

    unique = out["clause"].unique().tolist()
    preds = dict(zip(unique, sentiment_fn(unique)))
    out["sentiment"] = out["clause"].map(lambda c: preds[c][0])
    out["confidence"] = out["clause"].map(lambda c: preds[c][1])
    out["lang"] = out["clause"].map(detect_language)
    return out


def aspect_summary(aspect_df):
    """Counts of negative / neutral / positive mentions per aspect."""
    if aspect_df.empty:
        return pd.DataFrame(columns=["aspect", "mentions", "negative", "neutral",
                                     "positive", "negative_pct"])
    t = (aspect_df.groupby(["aspect", "sentiment"]).size()
         .unstack(fill_value=0)
         .reindex(columns=[0, 1, 2], fill_value=0))
    t.columns = ["negative", "neutral", "positive"]
    t["mentions"] = t.sum(axis=1)
    t["negative_pct"] = (100 * t["negative"] / t["mentions"]).round(1)
    return t.sort_values("negative", ascending=False).reset_index()[
        ["aspect", "mentions", "negative", "neutral", "positive", "negative_pct"]]
