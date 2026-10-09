import json

import pytest

from app.Retrieval.query_analyzer import (
    QueryAnalysis,
    QueryFilters,
    analyze_query,
    build_pinecone_filter,
)
import app.Retrieval.query_analyzer as query_analyzer_module


# ---------------------------------------------------------------------------
# Test doubles -- no live API calls, no real quota spent.
# ---------------------------------------------------------------------------

class FakeAnalyzer:
    """Stand-in for the Mistral structured-output Runnable."""

    def __init__(self, result: QueryAnalysis | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.call_count = 0

    def invoke(self, prompt):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """
    Point the module's on-disk cache at a throwaway path for every test, so
    tests never read/write the real cache used during actual development
    (which would either pollute it or make tests order-dependent on
    whatever's already cached from real usage).
    """
    cache_path = tmp_path / "query_analysis_cache.json"
    monkeypatch.setattr(query_analyzer_module, "CACHE_PATH", cache_path)
    monkeypatch.setattr(query_analyzer_module, "_last_call_time", 0.0)
    yield cache_path


@pytest.fixture
def fake_result():
    return QueryAnalysis(
        semantic_query="Nike running shoes",
        filters=QueryFilters(brand="Nike", sub_category="Running", final_price_lte=100.0),
    )


def _patch_analyzer(monkeypatch, fake):
    monkeypatch.setattr(query_analyzer_module, "get_query_analyzer", lambda model="mistral-small-latest": fake)


# ---------------------------------------------------------------------------
# Input validation -- no network required.
# ---------------------------------------------------------------------------

def test_analyze_query_empty_raises():
    with pytest.raises(ValueError):
        analyze_query("")


def test_analyze_query_whitespace_raises():
    with pytest.raises(ValueError):
        analyze_query("   ")


# ---------------------------------------------------------------------------
# Core extraction behavior, via FakeAnalyzer (no live API calls).
# ---------------------------------------------------------------------------

def test_analyze_query_returns_extracted_filters(monkeypatch, fake_result):
    fake = FakeAnalyzer(result=fake_result)
    _patch_analyzer(monkeypatch, fake)

    result = analyze_query("Nike running shoes under $100", use_cache=False)

    assert result.semantic_query == "Nike running shoes"
    assert result.filters.brand == "Nike"
    assert result.filters.sub_category == "Running"
    assert result.filters.final_price_lte == 100.0
    assert fake.call_count == 1


def test_analyze_query_caches_result(monkeypatch, fake_result):
    """Second call with the same query should hit the cache, not the API."""
    fake = FakeAnalyzer(result=fake_result)
    _patch_analyzer(monkeypatch, fake)

    first = analyze_query("Nike running shoes under $100")
    second = analyze_query("Nike running shoes under $100")

    assert fake.call_count == 1  # only the first call actually invoked the analyzer
    assert first == second


def test_analyze_query_cache_is_case_and_whitespace_insensitive(monkeypatch, fake_result):
    fake = FakeAnalyzer(result=fake_result)
    _patch_analyzer(monkeypatch, fake)

    analyze_query("Nike running shoes under $100")
    analyze_query("  NIKE running shoes under $100  ")

    assert fake.call_count == 1


def test_analyze_query_use_cache_false_forces_fresh_call(monkeypatch, fake_result):
    fake = FakeAnalyzer(result=fake_result)
    _patch_analyzer(monkeypatch, fake)

    analyze_query("Nike running shoes under $100")
    analyze_query("Nike running shoes under $100", use_cache=False)

    assert fake.call_count == 2


def test_analyze_query_cache_persists_to_disk(monkeypatch, fake_result, isolated_cache):
    fake = FakeAnalyzer(result=fake_result)
    _patch_analyzer(monkeypatch, fake)

    analyze_query("Nike running shoes under $100")

    assert isolated_cache.exists()
    saved = json.loads(isolated_cache.read_text())
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# Fail-soft behavior -- this is the important one given the free-tier 429s.
# ---------------------------------------------------------------------------

def test_analyze_query_falls_back_on_api_error(monkeypatch):
    """
    A rate-limit or network error should never propagate out of
    analyze_query -- it should degrade to plain semantic search instead.
    """
    fake = FakeAnalyzer(error=RuntimeError("429 Too Many Requests"))
    _patch_analyzer(monkeypatch, fake)

    result = analyze_query("Nike running shoes under $100", use_cache=False)

    assert result.semantic_query == "Nike running shoes under $100"
    assert result.filters == QueryFilters()  # no filters extracted, but no exception either


def test_analyze_query_does_not_cache_failed_calls(monkeypatch, isolated_cache):
    """A failed call shouldn't be cached as if it were a real (empty) result."""
    fake = FakeAnalyzer(error=RuntimeError("429 Too Many Requests"))
    _patch_analyzer(monkeypatch, fake)

    analyze_query("Nike running shoes under $100")

    assert not isolated_cache.exists() or json.loads(isolated_cache.read_text()) == {}


# ---------------------------------------------------------------------------
# build_pinecone_filter -- pure function, no mocking needed.
# ---------------------------------------------------------------------------

def test_build_pinecone_filter_empty_returns_none():
    assert build_pinecone_filter(QueryFilters()) is None


def test_build_pinecone_filter_single_field():
    result = build_pinecone_filter(QueryFilters(brand="Nike"))
    assert result == {"brand": "Nike"}


def test_build_pinecone_filter_combines_all_fields():
    filters = QueryFilters(
        brand="Nike",
        sub_category="Running",
        gender="Men",
        availability="In Stock",
        final_price_gte=50.0,
        final_price_lte=100.0,
    )
    result = build_pinecone_filter(filters)

    assert result == {
        "brand": "Nike",
        "sub_category": "Running",
        "gender": "Men",
        "availability": "In Stock",
        "final_price": {"$gte": 50.0, "$lte": 100.0},
    }


def test_build_pinecone_filter_price_bounds_share_one_key():
    """Both bounds must land under the same 'final_price' key, not overwrite each other."""
    filters = QueryFilters(final_price_gte=20.0, final_price_lte=80.0)
    result = build_pinecone_filter(filters)

    assert result["final_price"] == {"$gte": 20.0, "$lte": 80.0}


def test_build_pinecone_filter_only_upper_bound():
    result = build_pinecone_filter(QueryFilters(final_price_lte=100.0))
    assert result == {"final_price": {"$lte": 100.0}}


def test_build_pinecone_filter_only_lower_bound():
    result = build_pinecone_filter(QueryFilters(final_price_gte=20.0))
    assert result == {"final_price": {"$gte": 20.0}}


# ---------------------------------------------------------------------------
# Live test -- costs ONE real API call. Opt-in only, skipped by default.
# Run explicitly with: pytest -m live_llm
# ---------------------------------------------------------------------------

@pytest.mark.live_llm
@pytest.mark.skipif(
    not __import__("os").getenv("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set; skipping live extraction test.",
)
def test_analyze_query_live_extraction():
    result = analyze_query("Show me Nike running shoes under $100 for men", use_cache=False)

    print(f"\nsemantic_query: {result.semantic_query}")
    print(f"filters: {result.filters}")

    assert result.filters.brand == "Nike"
    assert result.filters.final_price_lte == 100.0