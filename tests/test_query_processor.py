from app.retrieval.query_processor import QueryProcessor

qp = QueryProcessor()


def test_decomposes_on_the_word_and_only():
    parts = qp.process_query("What are office timings and which cloud platform is used?")
    assert len(parts) == 2


def test_does_not_split_inside_words():
    # 'standard' and 'candidate' contain 'and' but must not be split
    assert len(qp.process_query("What is the standard candidate process")) == 1


def test_expands_known_terms():
    (expanded,) = qp.process_query("What is the remote work policy?")
    assert "work from home" in expanded


def test_deduplicates_and_normalises_whitespace():
    assert qp.process_query("  hello   world  ") == ["hello world"]
