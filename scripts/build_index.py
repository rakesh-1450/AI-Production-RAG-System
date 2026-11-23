"""Rebuild the FAISS index from data/documents. Run from the project root:
    python -m scripts.build_index
"""
from app.ingestion.indexer import build_index

if __name__ == "__main__":
    stats = build_index()
    print(f"Indexed {stats['documents']} documents into {stats['chunks']} chunks.")
