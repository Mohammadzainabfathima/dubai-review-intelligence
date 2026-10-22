"""Training code: TF-IDF baseline and XLM-RoBERTa fine-tuning.

Fine-tuning uses a plain PyTorch loop (instead of the HF Trainer) so it works
across library versions and every step is visible/explainable in the report:
  tokenise -> forward -> weighted cross-entropy -> backward -> AdamW step
  with linear warm-up/decay, gradient clipping and mixed precision on GPU.
The checkpoint with the best validation macro-F1 is kept.
"""
import os
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, make_union


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- baseline
def train_tfidf_baseline(train_df, text_col="clean_text_agg", label_col="label"):
    """Word + character n-gram TF-IDF with logistic regression (strong classic baseline)."""
    model = make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                            sublinear_tf=True, max_features=200_000),
        ),
        LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced"),
    )
    model.fit(train_df[text_col], train_df[label_col])
    return model


# ------------------------------------------------------------ fine-tuning
def finetune(train_df, val_df, base_model="xlm-roberta-base",
             out_dir="outputs/xlmr-sentiment", text_col="clean_text",
             label_col="label", epochs=2, lr=2e-5, batch_size=32,
             max_len=128, seed=42, use_class_weights=True):
    """Fine-tune a multilingual transformer for 3-class sentiment.

    Returns the training history (list of dicts) and saves the best model to out_dir.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}).to(device)

    def collate(rows):
        texts, labels = zip(*rows)
        enc = tok(list(texts), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc

    train_rows = list(zip(train_df[text_col], train_df[label_col].astype(int)))
    val_rows = list(zip(val_df[text_col], val_df[label_col].astype(int)))
    train_dl = DataLoader(train_rows, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_rows, batch_size=batch_size * 2, collate_fn=collate)

    if use_class_weights:
        counts = np.bincount(train_df[label_col].astype(int), minlength=3).astype(float)
        weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)),
                               dtype=torch.float, device=device)
    else:
        weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    history, best_f1 = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device, dtype=torch.float16,
                                enabled=(device == "cuda")):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                preds += model(**batch).logits.argmax(-1).cpu().tolist()
                golds += labels.cpu().tolist()
        f1 = f1_score(golds, preds, average="macro")
        history.append({"epoch": epoch, "train_loss": running / len(train_dl),
                        "val_macro_f1": f1})
        print(history[-1])
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(out_dir, exist_ok=True)
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    return history
