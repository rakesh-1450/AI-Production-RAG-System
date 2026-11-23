import json

import faiss
import numpy as np

from app.config.settings import settings
from app.core.embeddings import get_embedding_model


class VectorStore:
    """FAISS inner-product index over L2-normalised embeddings (= cosine similarity)."""

    def __init__(self):
        self.index = None
        self.documents = []

    # -- paths ---------------------------------------------------------
    @staticmethod
    def _paths():
        base = settings.VECTOR_DB_PATH
        return base / "faiss.index", base / "documents.json", base / "meta.json"

    @classmethod
    def exists(cls) -> bool:
        return all(p.exists() for p in cls._paths())

    # -- build / persist -----------------------------------------------
    def build(self, chunks):
        texts = [c["content"] for c in chunks]
        embeddings = get_embedding_model().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents = list(chunks)

    def save(self, fingerprint: str = ""):
        index_path, docs_path, meta_path = self._paths()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        docs_path.write_text(json.dumps(self.documents), encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "chunks": len(self.documents),
                }
            ),
            encoding="utf-8",
        )

    def load(self):
        index_path, docs_path, _ = self._paths()
        self.index = faiss.read_index(str(index_path))
        self.documents = json.loads(docs_path.read_text(encoding="utf-8"))

    # -- query ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        if self.index is None or not self.documents:
            return []
        k = min(top_k, len(self.documents))
        q = get_embedding_model().encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        scores, indices = self.index.search(np.asarray(q, dtype="float32"), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer hits exist
                continue
            doc = self.documents[idx]
            results.append({**doc, "dense_score": float(score)})
        return results
