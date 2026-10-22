"""End-to-end pipeline: reviews in -> manager briefing out."""
import os

import pandas as pd

from .aspects import aspect_summary, extract_aspect_sentiments
from .cluster import cluster_complaints
from .preprocess import preprocess_dataframe
from .summarize import build_evidence, get_backend, verify_summary
from .models import ZERO_SHOT_MODEL, HFSentiment


def load_sentiment_fn(path=None):
    """Use the fine-tuned model if present, else the multilingual zero-shot one."""
    path = path or ("outputs/xlmr-sentiment" if os.path.isdir("outputs/xlmr-sentiment")
                    else ZERO_SHOT_MODEL)
    print(f"[pipeline] sentiment model: {path}")
    return HFSentiment(path)


def run_pipeline(df, sentiment_fn, embed_fn, backend="auto", language="English",
                 text_col="text", k=None):
    """Run every stage and return all intermediate results.

    df needs a text column; a review_id column is created if absent.
    """
    df = df.copy()
    if "review_id" not in df.columns:
        df["review_id"] = range(1, len(df) + 1)
    reviews = preprocess_dataframe(df, text_col=text_col)

    # 1) review-level sentiment on the whole text
    preds = sentiment_fn(reviews["clean_text"].tolist())
    reviews["sentiment"] = [p[0] for p in preds]
    reviews["confidence"] = [p[1] for p in preds]

    # 2) aspect-level sentiment on clauses
    aspects = extract_aspect_sentiments(reviews, sentiment_fn, text_col="clean_text")
    a_sum = aspect_summary(aspects)

    # 3) cluster the negative clauses into recurring complaint themes
    neg, clusters = cluster_complaints(aspects, embed_fn, k=k)

    # 4) grounded LLM briefing + automatic faithfulness check
    evidence = build_evidence(a_sum, clusters, len(reviews))
    bk = get_backend(backend)
    summary = bk.generate(evidence, language)
    check = verify_summary(summary, evidence)

    return {"reviews": reviews, "aspects": aspects, "aspect_summary": a_sum,
            "negative_clauses": neg, "clusters": clusters, "evidence": evidence,
            "summary": summary, "verification": check, "backend": bk.name}
