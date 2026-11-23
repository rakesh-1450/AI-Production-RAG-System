from app.generation.extractive import extractive_answer
from app.generation.prompt_builder import NOT_FOUND
from app.utils.text import tokenize

DOCS = [
    {"source": "hr_policy.txt", "chunk_id": 0,
     "content": "Office timings are from 9:30 AM to 6:00 PM, Monday through Friday.\n\n"
                "Employees are entitled to 24 annual paid leaves per calendar year."},
]


def test_tokenize_stems_plurals():
    assert tokenize("Leaves and timings") == ["leave", "and", "timing"]


def test_extractive_finds_the_right_sentence():
    answer = extractive_answer("What are the office timings?", DOCS, max_sentences=1)
    assert "9:30 AM" in answer


def test_extractive_matches_plural_forms():
    answer = extractive_answer("How many annual leaves?", DOCS, max_sentences=1)
    assert "24" in answer


def test_extractive_not_found():
    assert extractive_answer("quantum chromodynamics", DOCS) == NOT_FOUND
    assert extractive_answer("anything", []) == NOT_FOUND


def test_extractive_skips_weakly_related_sentences():
    docs = [{"source": "t.txt", "chunk_id": 0,
             "content": "Cloud deployments are performed on AWS.\n\n"
                        "FAISS is used for vector similarity search in retrieval systems."}]
    answer = extractive_answer("Which cloud platform is used for deployments?", docs)
    assert "AWS" in answer and "FAISS" not in answer
