import json
import time

from app.config.settings import settings
from app.generation.rag_pipeline import RAGPipeline


class RAGEvaluator:
    """Runs benchmark questions and reports answer accuracy, retrieval hit-rate and latency."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or RAGPipeline()

    def load_questions(self, path=None):
        with open(path or settings.BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, api_key=None, progress=None):
        questions = self.load_questions()
        results = []

        for i, item in enumerate(questions, start=1):
            started = time.perf_counter()
            result = self.pipeline.run(item["question"], api_key=api_key)
            latency = time.perf_counter() - started

            answer = result["answer"].lower()
            keywords = item["expected_keywords"]
            answer_correct = all(k.lower() in answer for k in keywords)

            retrieved = {s["source"] for s in result["sources"]}
            expected_sources = item.get("expected_sources", [])
            retrieval_hit = all(s in retrieved for s in expected_sources)

            results.append(
                {
                    "question": item["question"],
                    "expected": ", ".join(keywords),
                    "answer": result["answer"],
                    "answer_correct": answer_correct,
                    "retrieval_hit": retrieval_hit,
                    "latency": round(latency, 2),
                    "mode": result["mode"],
                }
            )
            if progress:
                progress(i, len(questions))

        n = max(len(results), 1)
        return {
            "accuracy": round(sum(r["answer_correct"] for r in results) / n, 2),
            "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in results) / n, 2),
            "average_latency": round(sum(r["latency"] for r in results) / n, 2),
            "results": results,
        }
