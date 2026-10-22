"""Model-free tests. Run:  python tests/test_core.py   (or: pytest tests)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.aspects import aspect_summary, extract_aspect_sentiments, find_aspects
from src.cluster import cluster_complaints
from src.preprocess import clean_text, detect_language, preprocess_dataframe, split_clauses
from src.summarize import TemplateBackend, build_evidence, verify_summary


def fake_sentiment(texts):
    """Tiny rule-based stand-in for the transformer, only for testing the plumbing."""
    neg = ["cold", "slow", "rude", "dirty", "بارد", "بطيء", "وسخ", "خربان", "غالية",
           "nightmare", "overpriced", "waited", "ignored", "far", "بعيد", "رائحة", "قليلة"]
    out = []
    for t in texts:
        out.append((0, 0.9) if any(w in t.lower() for w in neg) else (2, 0.8))
    return out


def fake_embed(texts):
    """TF-IDF + SVD embeddings (offline stand-in for sentence-transformers)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts)
    d = max(2, min(20, X.shape[1] - 1, X.shape[0] - 1))
    Z = TruncatedSVD(d, random_state=0).fit_transform(X)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def test_cleaning():
    assert clean_text("جميييييل", "light") == "جمييل" or clean_text("جميييييل", "light") == "جمييل"
    t = clean_text("Call me +971 50 123 4567 or mail a.b@x.com http://x.co @bob")
    assert "<PHONE>" in t and "<EMAIL>" in t and "<URL>" in t and "@user" in t
    assert "ً" not in clean_text("مَرْحَباً")
    assert clean_text("الغُرْفَةُ", "aggressive") == "الغرفه"
    assert clean_text("Nooooo waaay", "light") == "Noo waay"
    assert clean_text("٣ نجوم", "light").startswith("3")


def test_language():
    assert detect_language("The room was great") == "en"
    assert detect_language("الغرفة كانت ممتازة") == "ar"
    assert detect_language("الخدمة كانت great بس الأكل cold") == "mixed"
    assert detect_language("12345 !!!") == "other"


def test_clauses():
    c = split_clauses("الخدمة كانت great بس الأكل cold")
    assert len(c) == 2, c
    c = split_clauses("Nice room but terrible breakfast. Staff were rude!")
    assert len(c) == 3, c


def test_aspects():
    assert find_aspects("The breakfast was cold") == {"food"}
    assert "service" in find_aspects("الخدمة بطيئة")
    assert find_aspects("مواقف السيارات قليلة") == {"facilities"}
    assert "price" in find_aspects("Overpriced for what you get")
    assert find_aspects("Nothing to say") == set()


def test_full_flow():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "sample_reviews.csv"))
    rev = preprocess_dataframe(df)
    assert len(rev) == len(df) and set(rev.lang) >= {"ar", "en"}
    asp = extract_aspect_sentiments(rev, fake_sentiment, text_col="clean_text")
    summ = aspect_summary(asp)
    assert {"aspect", "negative", "negative_pct"} <= set(summ.columns)
    neg, clusters = cluster_complaints(asp, fake_embed, k=3)
    assert clusters and sum(c["size"] for c in clusters) == len(neg)
    ev = build_evidence(summ, clusters, len(rev))
    text = TemplateBackend().generate(ev)
    check = verify_summary(text, ev)
    assert check["passed"], (text, check)


def test_verifier_catches_hallucination():
    ev = {"total_reviews": 100,
          "aspects": [{"aspect": "food", "mentions": 40, "negative": 20, "neutral": 5,
                       "positive": 15, "negative_pct": 50.0}],
          "complaint_clusters": [{"cluster": 0, "size": 20, "share_pct": 100.0, "top_aspect": "food",
                                  "terms": ["cold"], "examples": [{"id": "R7", "text": "cold food"}]}]}
    good = "Food is the top issue: 20 complaints [R7]."
    bad = "Food caused 87% of complaints [R99] across 5000 reviews."
    assert verify_summary(good, ev)["passed"]
    r = verify_summary(bad, ev)
    assert not r["passed"] and "R99" in r["invalid_citations"] and r["unsupported_numbers"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all tests passed")
