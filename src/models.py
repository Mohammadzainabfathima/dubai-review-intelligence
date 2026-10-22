"""Sentiment model wrapper used by the pipeline, evaluation and demo."""
import numpy as np

ZERO_SHOT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # multilingual, no training


class HFSentiment:
    """Callable: list[str] -> list[(label_id, confidence)].

    label ids: 0 = negative, 1 = neutral, 2 = positive. Works with any
    3-class sequence-classification checkpoint (our fine-tuned model or the
    zero-shot multilingual baseline).
    """

    def __init__(self, model_name_or_path, batch_size=64, max_len=128, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path).to(self.device).eval()
        self.batch_size, self.max_len = batch_size, max_len
        self.id_map = self._build_id_map(self.model.config.id2label)

    @staticmethod
    def _build_id_map(id2label):
        """Map the checkpoint's label ids to our 0/1/2 convention."""
        mapping = {}
        for i, name in id2label.items():
            n = str(name).lower()
            if "neg" in n:
                mapping[int(i)] = 0
            elif "neu" in n:
                mapping[int(i)] = 1
            elif "pos" in n:
                mapping[int(i)] = 2
        # Generic names (LABEL_0..2): assume order negative, neutral, positive
        return mapping if len(mapping) == len(id2label) else {i: i for i in range(len(id2label))}

    def __call__(self, texts):
        torch, out = self.torch, []
        texts = list(texts)
        for s in range(0, len(texts), self.batch_size):
            batch = texts[s:s + self.batch_size]
            enc = self.tok(batch, truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            for p in probs:
                k = int(np.argmax(p))
                out.append((self.id_map[k], float(p[k])))
        return out

    def predict_labels(self, texts):
        return [lab for lab, _ in self(texts)]
