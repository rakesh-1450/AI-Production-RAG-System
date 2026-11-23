"""Run the benchmark in benchmark/questions.json:
    python -m scripts.run_evaluation
"""
from app.evaluation.evaluator import RAGEvaluator
from app.ingestion.indexer import ensure_index

if __name__ == "__main__":
    ensure_index()
    report = RAGEvaluator().evaluate()
    print(f"\nAnswer accuracy : {report['accuracy']:.0%}")
    print(f"Retrieval hits  : {report['retrieval_hit_rate']:.0%}")
    print(f"Average latency : {report['average_latency']}s\n")
    for r in report["results"]:
        mark = "PASS" if r["answer_correct"] else "FAIL"
        print(f"[{mark}] {r['question']}\n       -> {r['answer']}")
