from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.rrf_fusion import RRFFusion


class HybridRetriever:
    """Dense (FAISS) + sparse (BM25) retrieval merged with Reciprocal Rank Fusion."""

    def __init__(self, vector_store):
        self.dense = DenseRetriever(vector_store)
        self.sparse = BM25Retriever(vector_store.documents)
        self.rrf = RRFFusion()

    def retrieve(self, query: str, top_k: int = 5):
        dense_results = self.dense.retrieve(query, top_k)
        sparse_results = self.sparse.retrieve(query, top_k)
        return self.rrf.fuse(dense_results, sparse_results)[:top_k]
