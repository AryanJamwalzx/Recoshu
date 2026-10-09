import os

import pytest
from langchain_core.documents import Document

from app.Pipeline import pipeline
from app.Pipeline.pipeline import run_pipeline

requires_live_services = pytest.mark.skipif(
    not (os.getenv("MISTRAL_API_KEY") and os.getenv("PINECONE_API_KEY")),
    reason="MISTRAL_API_KEY and/or PINECONE_API_KEY not set; skipping live pipeline test.",
)


# ---------------------------------------------------------------------------
# Input validation -- no dependencies required
# ---------------------------------------------------------------------------

def test_run_pipeline_empty_query_raises():
    with pytest.raises(ValueError):
        run_pipeline("")


def test_run_pipeline_whitespace_query_raises():
    with pytest.raises(ValueError):
        run_pipeline("   ")


# ---------------------------------------------------------------------------
# Orchestration logic -- every stage mocked, so this tests only that
# pipeline.py wires the four stages together correctly: right call order,
# right arguments passed between stages, right final return value. No
# network, no API keys, no real models required.
# ---------------------------------------------------------------------------

class FakeAnalysis:
    def __init__(self, semantic_query, filters):
        self.semantic_query = semantic_query
        self.filters = filters


def test_run_pipeline_calls_stages_in_correct_order(monkeypatch):
    call_order = []
    fake_analysis = FakeAnalysis(semantic_query="running shoes", filters="fake_filters")
    fake_candidates = [Document(page_content=f"doc{i}") for i in range(20)]
    fake_top_docs = fake_candidates[:5]

    def fake_analyze_query(query):
        call_order.append("analyze_query")
        assert query == "Nike running shoes"
        return fake_analysis

    def fake_build_pinecone_filter(filters):
        call_order.append("build_pinecone_filter")
        assert filters == "fake_filters"
        return {"brand": "Nike"}

    def fake_retrieve(query, top_k, filter):
        call_order.append("retrieve")
        assert query == "running shoes"
        assert top_k == pipeline.DEFAULT_INITIAL_K
        assert filter == {"brand": "Nike"}
        return fake_candidates

    def fake_rerank_documents(query, documents, top_n):
        call_order.append("rerank_documents")
        assert query == "running shoes"
        assert documents == fake_candidates
        assert top_n == pipeline.DEFAULT_FINAL_K
        return fake_top_docs

    def fake_build_context(documents):
        call_order.append("build_context")
        assert documents == fake_top_docs
        return "FINAL CONTEXT STRING"

    monkeypatch.setattr(pipeline, "analyze_query", fake_analyze_query)
    monkeypatch.setattr(pipeline, "build_pinecone_filter", fake_build_pinecone_filter)
    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "rerank_documents", fake_rerank_documents)
    monkeypatch.setattr(pipeline, "build_context", fake_build_context)

    result = run_pipeline("Nike running shoes")

    assert result == "FINAL CONTEXT STRING"
    assert call_order == [
        "analyze_query",
        "build_pinecone_filter",
        "retrieve",
        "rerank_documents",
        "build_context",
    ]


def test_run_pipeline_respects_custom_k_values(monkeypatch):
    fake_analysis = FakeAnalysis(semantic_query="shoes", filters=None)
    fake_candidates = [Document(page_content="doc")]

    monkeypatch.setattr(pipeline, "analyze_query", lambda q: fake_analysis)
    monkeypatch.setattr(pipeline, "build_pinecone_filter", lambda f: None)

    captured = {}

    def fake_retrieve(query, top_k, filter):
        captured["top_k"] = top_k
        return fake_candidates

    def fake_rerank_documents(query, documents, top_n):
        captured["top_n"] = top_n
        return documents

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "rerank_documents", fake_rerank_documents)
    monkeypatch.setattr(pipeline, "build_context", lambda docs: "context")

    run_pipeline("shoes", initial_k=50, final_k=3)

    assert captured["top_k"] == 50
    assert captured["top_n"] == 3


def test_run_pipeline_no_candidates_raises_clear_error(monkeypatch):
    fake_analysis = FakeAnalysis(semantic_query="obscure query", filters=None)

    monkeypatch.setattr(pipeline, "analyze_query", lambda q: fake_analysis)
    monkeypatch.setattr(pipeline, "build_pinecone_filter", lambda f: None)
    monkeypatch.setattr(pipeline, "retrieve", lambda query, top_k, filter: [])

    with pytest.raises(ValueError, match="No relevant results"):
        run_pipeline("some obscure query")


# ---------------------------------------------------------------------------
# Live end-to-end test -- requires real MISTRAL_API_KEY, PINECONE_API_KEY,
# and data already ingested via app/Ingestion/ingest.py.
# ---------------------------------------------------------------------------

@requires_live_services
def test_run_pipeline_end_to_end_real():
    context = run_pipeline("What Nike shoes are good for running?")

    print("\n--- Final pipeline context ---")
    print(context)

    assert isinstance(context, str)
    assert len(context) > 0