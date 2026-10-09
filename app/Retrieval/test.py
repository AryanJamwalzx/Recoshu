import pytest

from app.Retrieval.retrieval import retrieve, retrieve_with_scores


# ---------------------------------------------------------------------------
# Live retrieval tests -- require PINECONE_API_KEY set and data already
# ingested via app/Ingestion/ingest.py. Run these after a successful ingest.
# ---------------------------------------------------------------------------

def test_retrieve_returns_results():
    query = "What Nike shoes are good for running?"

    results = retrieve(query=query, top_k=5)

    print(f"\nQuery: {query}")
    print(f"Number of results: {len(results)}")

    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(document.page_content[:200])
        print(document.metadata)

    assert len(results) > 0
    assert len(results) <= 5


def test_retrieve_with_scores_returns_results():
    query = "What is your return policy?"

    results = retrieve_with_scores(query=query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Number of results: {len(results)}")

    for i, (document, score) in enumerate(results, start=1):
        print(f"\n--- Result {i} (score={score:.4f}) ---")
        print(document.page_content[:200])

    assert len(results) > 0
    for document, score in results:
        assert isinstance(score, float)


def test_retrieve_with_metadata_filter():
    results = retrieve(
        query="comfortable shoes",
        top_k=5,
        filter={"document_type": "faq"},
    )

    print(f"\nFiltered results (document_type=faq): {len(results)}")

    for document in results:
        assert document.metadata.get("document_type") == "faq"


# ---------------------------------------------------------------------------
# Input validation tests -- no network required, should pass regardless of
# Pinecone connectivity since validation happens before any API call.
# ---------------------------------------------------------------------------

def test_retrieve_empty_query_raises():
    with pytest.raises(ValueError):
        retrieve(query="")


def test_retrieve_whitespace_query_raises():
    with pytest.raises(ValueError):
        retrieve(query="   ")


def test_retrieve_invalid_top_k_raises():
    with pytest.raises(ValueError):
        retrieve(query="running shoes", top_k=0)


def test_retrieve_negative_top_k_raises():
    with pytest.raises(ValueError):
        retrieve(query="running shoes", top_k=-1)


def test_retrieve_with_scores_empty_query_raises():
    with pytest.raises(ValueError):
        retrieve_with_scores(query="")


def test_retrieve_with_scores_invalid_top_k_raises():
    with pytest.raises(ValueError):
        retrieve_with_scores(query="running shoes", top_k=0)