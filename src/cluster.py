"""Complaint clustering and review retrieval.

Negative clauses are embedded with a multilingual sentence-transformer and
grouped with k-means (k chosen by silhouette score). Each cluster gets
readable top terms (c-TF-IDF style) and representative example reviews.
The same embeddings power a small semantic-search index (FAISS if
installed, plain NumPy otherwise) used for retrieval-grounded answers.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .preprocess import clean_text

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

STOPWORDS = set("""
the a an and or of to in is was were are it this that for with on at as be we i my
very so but not no they them our us you your had has have there
في من على الى عن هذا هذه ذلك كان كانت و ان انه انها ما لا لم مع كل جدا وما وفي ومن والي
""".split())


def default_embed_fn(model_name: str = EMBED_MODEL):
    """Return fn(list[str]) -> L2-normalised embeddings (multilingual)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def embed(texts):
        return model.encode(list(texts), batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True)
    return embed


def _choose_k(emb, k_range, random_state):
    best_k, best_score = k_range[0], -1.0
    n = len(emb)
    for k in range(k_range[0], min(k_range[1], n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=random_state).fit_predict(emb)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_complaints(aspect_df, embed_fn, k=None, k_range=(3, 8),
                       n_examples=3, random_state=42, id_col="review_id"):
    """Cluster negative clauses. Returns (negative_df_with_cluster, clusters)."""
    neg = aspect_df[aspect_df["sentiment"] == 0].drop_duplicates("clause").reset_index(drop=True)
    if len(neg) == 0:
        return neg.assign(cluster=[]), []

    emb = np.asarray(embed_fn(neg["clause"].tolist()))
    if len(neg) < max(6, k_range[0] * 2):
        k = 1
    elif k is None:
        k = _choose_k(emb, k_range, random_state)

    if k == 1:
        labels = np.zeros(len(neg), dtype=int)
        centroids = emb.mean(axis=0, keepdims=True)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(emb)
        labels, centroids = km.labels_, km.cluster_centers_
    neg = neg.assign(cluster=labels)

    # c-TF-IDF style terms: one document per cluster
    docs = [" ".join(neg.loc[neg.cluster == c, "clause"]) for c in range(k)]
    vec = TfidfVectorizer(preprocessor=lambda t: clean_text(t, "aggressive"),
                          stop_words=sorted({clean_text(w, "aggressive") for w in STOPWORDS}), ngram_range=(1, 2),
                          max_df=0.8 if k >= 3 else 1.0)
    try:
        tfidf = vec.fit_transform(docs)
        terms = np.array(vec.get_feature_names_out())
    except ValueError:  # empty vocabulary
        tfidf, terms = None, np.array([])

    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        d = np.linalg.norm(emb[idx] - centroids[c], axis=1)
        nearest = idx[np.argsort(d)[:n_examples]]
        top_terms = []
        if tfidf is not None and len(terms):
            row = tfidf[c].toarray().ravel()
            top_terms = [terms[i] for i in row.argsort()[::-1][:4] if row[i] > 0]
        clusters.append({
            "cluster": int(c),
            "size": int(len(idx)),
            "share_pct": round(100 * len(idx) / len(neg), 1),
            "top_aspect": neg.loc[idx, "aspect"].mode().iloc[0],
            "terms": top_terms,
            "examples": [{"id": f"R{neg.loc[i, id_col]}", "text": neg.loc[i, "clause"]}
                         for i in nearest],
        })
    clusters.sort(key=lambda x: x["size"], reverse=True)
    return neg, clusters


class ReviewIndex:
    """Semantic search over reviews (retrieval step of the RAG pipeline)."""

    def __init__(self, texts, ids, embed_fn):
        self.texts, self.ids, self.embed_fn = list(texts), list(ids), embed_fn
        self.emb = np.asarray(embed_fn(self.texts), dtype="float32")
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        except ImportError:
            self.index = None  # NumPy fallback

    def search(self, query, top_k=5):
        q = np.asarray(self.embed_fn([query]), dtype="float32")
        if self.index is not None:
            scores, idx = self.index.search(q, min(top_k, len(self.texts)))
            scores, idx = scores[0], idx[0]
        else:
            sims = (self.emb @ q.T).ravel()
            idx = np.argsort(-sims)[:top_k]
            scores = sims[idx]
        return [{"id": f"R{self.ids[i]}", "text": self.texts[i], "score": float(s)}
                for i, s in zip(idx, scores)]
