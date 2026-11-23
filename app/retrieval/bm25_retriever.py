from rank_bm25 import BM25Okapi

from app.utils.text import tokenize


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.bm25 = BM25Okapi([tokenize(d["content"]) for d in documents]) if documents else None

    def retrieve(self, query: str, top_k: int = 5):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query, drop_stopwords=True))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # Skip zero-score hits: they are not lexical matches at all.
        return [
            {**self.documents[i], "bm25_score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]
