class RRFFusion:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across ranked lists."""

    def __init__(self, k: int = 60):
        self.k = k

    @staticmethod
    def _key(doc):
        return (doc.get("source"), doc.get("chunk_id"), doc["content"])

    def fuse(self, *ranked_lists):
        scores, docs = {}, {}
        for results in ranked_lists:
            for rank, doc in enumerate(results):
                key = self._key(doc)
                docs[key] = {**docs.get(key, {}), **doc}
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.k + rank + 1)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [{**docs[key], "rrf_score": score} for key, score in ranked]
