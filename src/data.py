"""Dataset loading, label harmonisation and code-switch augmentation.

Target schema: review_id, text, label (0 neg / 1 neutral / 2 positive),
source, domain.

You need two public review datasets (see README for exact download steps):
  * Arabic: e.g. "Arabic 100k Reviews" (Kaggle, labels Positive/Negative/Mixed)
            or HARD (hotel reviews, 1-5 stars) or LABR.
  * English: e.g. Yelp reviews (HF 'Yelp/yelp_review_full') or any hotel
            review CSV from Kaggle.
Put the CSV files in data/ and call load_csv() - dataset names on the web
change, so loading from your own CSV is the most reliable route.
"""
import random

import pandas as pd

POS_WORDS = {"positive", "pos", "good"}
NEG_WORDS = {"negative", "neg", "bad"}
NEU_WORDS = {"neutral", "neu", "mixed", "mix", "average"}


def harmonise_label(v, star_scale=True):
    """Map stars (1-5), words or 0/1/2 ids to 0/1/2. Returns None if unknown."""
    s = str(v).strip().lower()
    if s in NEU_WORDS:
        return 1
    if s in POS_WORDS:
        return 2
    if s in NEG_WORDS:
        return 0
    try:
        x = float(s)
    except ValueError:
        return None
    if not star_scale:  # already 0/1/2 ids
        return int(x) if int(x) in (0, 1, 2) else None
    if x <= 2:
        return 0
    if x == 3:
        return 1
    return 2  # 4-5 stars


def load_csv(path, text_col, label_col, source, domain="hospitality",
             star_scale=True, max_rows=None, seed=42):
    """Load a CSV of reviews into the project schema.

    star_scale=True  -> label_col holds 1-5 stars (<=2 neg, 3 neu, >=4 pos)
    star_scale=False -> label_col holds words / 0-1-2 ids (use for binary 0/1
                        datasets: set label to 0 neg / 2 pos beforehand).
    """
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna().rename(
        columns={text_col: "text", label_col: "raw_label"})
    df["label"] = df["raw_label"].map(lambda v: harmonise_label(v, star_scale))
    df = df.dropna(subset=["label"]).drop(columns="raw_label")
    df["label"] = df["label"].astype(int)
    df["source"], df["domain"] = source, domain
    if max_rows and len(df) > max_rows:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max_rows // 3), random_state=seed))
    return df.reset_index(drop=True)


def balance_by_label(df, per_class, seed=42):
    """Down-sample to at most per_class reviews per label."""
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
              .reset_index(drop=True))


CONNECTIVES_POS = ["والله", "and honestly", "كمان", "plus", "و"]
CONNECTIVES_NEG = ["بس", "but", "للأسف", "unfortunately", "و"]


def make_code_switched(df_ar, df_en, n=2000, seed=42, max_len=160):
    """Create synthetic code-switched reviews (Arabic clause + English clause).

    Only pairs two reviews with the SAME label so the combined label stays
    correct. This simulates how people in Dubai write ("الخدمة ممتازة and the
    room was great") and lets us test the model on mixed text. Clearly
    documented as synthetic augmentation in the report.
    """
    rnd = random.Random(seed)
    rows = []
    for label in (0, 1, 2):
        ar = df_ar[df_ar.label == label]["text"].tolist()
        en = df_en[df_en.label == label]["text"].tolist()
        ar = [t for t in ar if 15 <= len(t) <= max_len]
        en = [t for t in en if 15 <= len(t) <= max_len]
        if not ar or not en:
            continue
        for _ in range(n // 3):
            a, e = rnd.choice(ar), rnd.choice(en)
            con = rnd.choice(CONNECTIVES_NEG if label == 0 else CONNECTIVES_POS)
            parts = [a, con, e]
            if rnd.random() < 0.5:
                parts = [e, con, a]
            rows.append({"text": " ".join(parts), "label": label,
                         "source": "synthetic_code_switch", "domain": "hospitality"})
    return pd.DataFrame(rows)


def stratified_split(df, val=0.1, test=0.1, seed=42):
    """Train/val/test split stratified by label."""
    from sklearn.model_selection import train_test_split
    train, rest = train_test_split(df, test_size=val + test, random_state=seed,
                                   stratify=df["label"])
    v, t = train_test_split(rest, test_size=test / (val + test), random_state=seed,
                            stratify=rest["label"])
    return (train.reset_index(drop=True), v.reset_index(drop=True),
            t.reset_index(drop=True))
