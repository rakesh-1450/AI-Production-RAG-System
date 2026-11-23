from pathlib import Path

from app.config.settings import settings

SUPPORTED_SUFFIXES = {".txt", ".md"}


def list_document_files(data_path=None):
    path = Path(data_path or settings.DOCUMENTS_PATH)
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


class DocumentLoader:
    def __init__(self, data_path=None):
        self.data_path = Path(data_path or settings.DOCUMENTS_PATH)

    def load_documents(self):
        documents = []
        for file_path in list_document_files(self.data_path):
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append({"content": text, "source": file_path.name})
        return documents
