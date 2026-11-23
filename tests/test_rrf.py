from app.retrieval.rrf_fusion import RRFFusion


def doc(i):
    return {"source": "a.txt", "chunk_id": i, "content": f"chunk {i}"}


def test_documents_in_both_lists_rank_first():
    fused = RRFFusion().fuse([doc(1), doc(2), doc(3)], [doc(3), doc(4)])
    assert fused[0]["chunk_id"] == 3          # appears in both lists
    assert {d["chunk_id"] for d in fused} == {1, 2, 3, 4}  # no duplicates


def test_scores_are_attached_and_sorted():
    fused = RRFFusion().fuse([doc(1), doc(2)], [doc(2), doc(1)])
    scores = [d["rrf_score"] for d in fused]
    assert scores == sorted(scores, reverse=True)
