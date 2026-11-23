import time

from app.config.settings import settings
from app.core.vector_store import VectorStore
from app.evaluation.metrics_logger import MetricsLogger
from app.generation.extractive import extractive_answer
from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import NOT_FOUND, PromptBuilder
from app.guardrails.input_guardrails import InputGuardrails
from app.guardrails.output_guardrails import OutputGuardrails
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_processor import QueryProcessor
from app.utils.logger import logger


class RAGPipeline:
    """query -> guardrails -> query processing -> hybrid retrieval -> rerank -> answer."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.load()

        self.query_processor = QueryProcessor()
        self.retriever = HybridRetriever(self.vector_store)
        self.reranker = CrossEncoderReranker()
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.metrics_logger = MetricsLogger()

        self.reranker.warmup()  # load models now so the first query is fast

    @staticmethod
    def _result(query, answer, model, mode, sources, latency, stages, blocked=False, notice=None):
        return {
            "query": query,
            "answer": answer,
            "model": model,
            "mode": mode,
            "sources": sources,
            "latency": latency,
            "stages": stages,
            "blocked": blocked,
            "notice": notice,
        }

    def run(self, query, top_k=None, context_k=None, api_key=None):
        top_k = top_k or settings.TOP_K_RETRIEVAL
        context_k = context_k or settings.TOP_K_CONTEXT
        started = time.perf_counter()
        stages = {"retrieval": 0.0, "rerank": 0.0, "generation": 0.0}

        validation = self.input_guardrails.validate_query(query)
        if not validation["allowed"]:
            return self._result(
                query, "Query blocked by security guardrails.", None, "blocked", [], 0.0,
                stages, blocked=True, notice=validation["reason"],
            )

        # 1) hybrid retrieval over every processed sub-query
        t = time.perf_counter()
        candidates, seen = [], set()
        for sub_query in self.query_processor.process_query(query):
            for doc in self.retriever.retrieve(sub_query, top_k):
                key = (doc["source"], doc["chunk_id"])
                if key not in seen:
                    seen.add(key)
                    candidates.append(doc)
        stages["retrieval"] = time.perf_counter() - t

        # 2) cross-encoder rerank against the ORIGINAL question
        t = time.perf_counter()
        final_docs = self.reranker.rerank(query, candidates, top_k=context_k)
        stages["rerank"] = time.perf_counter() - t

        # 3) answer: LLM when available, extractive fallback otherwise
        t = time.perf_counter()
        answer, model, mode, notice = None, None, "extractive", None

        if not final_docs:
            answer = NOT_FOUND
        else:
            if self.llm_client.is_configured(api_key):
                prompt = self.prompt_builder.build_prompt(query, final_docs)
                response = self.llm_client.generate(prompt, api_key=api_key)
                if response["answer"]:
                    check = self.output_guardrails.validate_response(response["answer"], final_docs)
                    if check["valid"]:
                        answer, model, mode = response["answer"], response["model"], "llm"
                    else:
                        notice = "LLM answer failed the grounding check; showing extracted passages instead."
                else:
                    notice = f"LLM unavailable ({response['error']}); showing extracted passages instead."

            if answer is None:
                answer = extractive_answer(query, final_docs)
                model = "extractive"
        stages["generation"] = time.perf_counter() - t

        latency = round(time.perf_counter() - started, 4)
        logger.info("RAG pipeline completed in %s s (mode=%s)", latency, mode)
        self.metrics_logger.log(query=query, model=model, latency=latency, mode=mode)

        return self._result(
            query, answer, model, mode, final_docs, latency,
            {k: round(v, 4) for k, v in stages.items()}, notice=notice,
        )
