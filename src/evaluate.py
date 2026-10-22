"""Evaluation utilities: overall + per-language metrics, CIs, plots, errors."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import LABELS


def bootstrap_ci(y_true, y_pred, n=500, seed=42):
    """95% bootstrap confidence interval for macro-F1."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_predictions(df, y_pred, name, label_col="label", lang_col="lang"):
    """Return a dict of metrics, overall and for each language group."""
    y_true = df[label_col].astype(int).to_numpy()
    y_pred = np.asarray(y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred)
    res = {"model": name, "n": int(len(df)),
           "accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro"),
           "macro_f1_ci_low": lo, "macro_f1_ci_high": hi}
    for lang in ["ar", "en", "mixed"]:
        mask = (df[lang_col] == lang).to_numpy()
        if mask.sum() >= 10:
            res[f"f1_{lang}"] = f1_score(y_true[mask], y_pred[mask], average="macro")
            res[f"n_{lang}"] = int(mask.sum())
    return res


def comparison_table(results):
    return pd.DataFrame(results).set_index("model").round(3)


def plot_confusion(y_true, y_pred, title, path=None):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), LABELS, rotation=30)
    ax.set_yticks(range(3), LABELS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def error_analysis(df, y_pred, n=40, label_col="label", seed=42):
    """Random sample of misclassified reviews for the qualitative error analysis."""
    d = df.copy()
    d["pred"] = np.asarray(y_pred)
    wrong = d[d[label_col].astype(int) != d["pred"]]
    return wrong.sample(min(n, len(wrong)), random_state=seed)[
        ["text", "lang", label_col, "pred"]]


def save_results(obj, path="results/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
