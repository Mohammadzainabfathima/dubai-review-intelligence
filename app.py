"""Gradio demo: upload reviews (CSV) or paste text -> aspects, complaints, briefing.

Run:  python app.py        (opens a local web page; on Colab it prints a public link)
"""
import io

import gradio as gr
import pandas as pd

from src.cluster import ReviewIndex, default_embed_fn
from src.pipeline import load_sentiment_fn, run_pipeline
from src.summarize import get_backend

SENTIMENT_FN = None
EMBED_FN = None
LAST = {}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def _models():
    global SENTIMENT_FN, EMBED_FN
    if SENTIMENT_FN is None:
        SENTIMENT_FN = load_sentiment_fn()
        EMBED_FN = default_embed_fn()
    return SENTIMENT_FN, EMBED_FN


def analyse(csv_file, pasted, backend, language):
    if csv_file is not None:
        df = pd.read_csv(csv_file.name if hasattr(csv_file, "name") else csv_file)
        text_col = "text" if "text" in df.columns else df.columns[0]
        df = df.rename(columns={text_col: "text"})
    elif pasted and pasted.strip():
        df = pd.DataFrame({"text": [l.strip() for l in pasted.splitlines() if l.strip()]})
    else:
        raise gr.Error("Upload a CSV with a 'text' column or paste reviews (one per line).")

    sentiment_fn, embed_fn = _models()
    res = run_pipeline(df, sentiment_fn, embed_fn, backend=backend, language=language)
    LAST["res"], LAST["index"] = res, None

    rev = res["reviews"][["review_id", "text", "lang", "sentiment", "confidence"]].copy()
    rev["sentiment"] = rev["sentiment"].map(LABEL_NAMES)
    rev["confidence"] = rev["confidence"].round(2)

    clusters = pd.DataFrame([{
        "theme": ", ".join(c["terms"][:3]) or c["top_aspect"], "main aspect": c["top_aspect"],
        "complaints": c["size"], "% of negatives": c["share_pct"],
        "example": f"[{c['examples'][0]['id']}] {c['examples'][0]['text']}"}
        for c in res["clusters"]])

    v = res["verification"]
    check = (f"Backend: {res['backend']} | citations: {v['citations']} "
             f"(invalid: {v['invalid_citations'] or 'none'}) | numbers checked: "
             f"{v['numbers_checked']} (unsupported: {v['unsupported_numbers'] or 'none'}) "
             f"| faithfulness: {v['faithfulness']}")
    plot_df = res["aspect_summary"].set_index("aspect")[["negative", "neutral", "positive"]]
    return res["summary"], check, res["aspect_summary"], plot_df.reset_index(), clusters, rev


def ask(question):
    res = LAST.get("res")
    if not res:
        raise gr.Error("Run the analysis first.")
    _, embed_fn = _models()
    if LAST.get("index") is None:
        r = res["reviews"]
        LAST["index"] = ReviewIndex(r["clean_text"], r["review_id"], embed_fn)
    hits = LAST["index"].search(question, top_k=5)
    return pd.DataFrame(hits)


with gr.Blocks(title="Dubai Review Intelligence") as demo:
    gr.Markdown("# Dubai Review Intelligence\nBilingual (Arabic / English / mixed) review "
                "analysis: aspect sentiment, complaint themes and a fact-checked briefing.")
    with gr.Row():
        csv_in = gr.File(label="CSV with a 'text' column", file_types=[".csv"])
        pasted = gr.Textbox(label="...or paste reviews (one per line)", lines=6)
    with gr.Row():
        backend = gr.Radio(["auto", "template", "claude", "local"], value="auto",
                           label="Summary backend")
        language = gr.Radio(["English", "Arabic"], value="English", label="Briefing language")
    go = gr.Button("Analyse", variant="primary")
    summary = gr.Textbox(label="Manager briefing", lines=6)
    check = gr.Textbox(label="Faithfulness check")
    aspects = gr.Dataframe(label="Aspect sentiment (negative first)")
    chart = gr.BarPlot(x="aspect", y="negative", label="Negative mentions per aspect")
    clusters = gr.Dataframe(label="Complaint themes")
    reviews = gr.Dataframe(label="Per-review results")
    go.click(analyse, [csv_in, pasted, backend, language],
             [summary, check, aspects, chart, clusters, reviews])

    gr.Markdown("### Ask your reviews (semantic search)")
    q = gr.Textbox(label="e.g. 'problems with parking' or 'مشاكل المكيف'")
    hits = gr.Dataframe(label="Most relevant reviews")
    q.submit(ask, q, hits)

if __name__ == "__main__":
    demo.launch(share=True)
