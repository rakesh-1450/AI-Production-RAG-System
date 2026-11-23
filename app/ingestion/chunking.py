from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings


class ChunkingService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def create_chunks(self, documents):
        chunks = []
        chunk_id = 0
        for doc in documents:
            for piece in self.text_splitter.split_text(doc["content"]):
                chunks.append(
                    {"content": piece, "source": doc["source"], "chunk_id": chunk_id}
                )
                chunk_id += 1
        return chunks
