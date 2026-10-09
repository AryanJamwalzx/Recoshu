import pytest

from langchain_core.documents import Document

from app.Retrieval import reranker
from app.Retrieval.reranker import (
    rerank_documents,
    rerank_with_scores,
)


# ---------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------

def test_rerank_documents_empty_query_raises():
    documents = [Document(page_content="x")]

    with pytest.raises(ValueError):
        rerank_documents("", documents)


def test_rerank_documents_whitespace_query_raises():
    documents = [Document(page_content="x")]

    with pytest.raises(ValueError):
        rerank_documents("   ", documents)


def test_rerank_documents_empty_documents_raises():
    with pytest.raises(ValueError):
        rerank_documents("running shoes", [])


def test_rerank_documents_invalid_top_n_raises():
    documents = [Document(page_content="x")]

    with pytest.raises(ValueError):
        rerank_documents(
            "running shoes",
            documents,
            top_n=0,
        )


def test_rerank_with_scores_same_guards():
    with pytest.raises(ValueError):
        rerank_with_scores(
            "",
            [Document(page_content="x")],
        )

    with pytest.raises(ValueError):
        rerank_with_scores(
            "query",
            [],
        )

    with pytest.raises(ValueError):
        rerank_with_scores(
            "query",
            [Document(page_content="x")],
            top_n=-1,
        )


# ---------------------------------------------------------
# Fake reranker for unit tests
# ---------------------------------------------------------

class FakeCrossEncoder:
    """
    Fake CrossEncoder used for unit tests.

    It returns predefined scores instead of loading
    the real model.
    """

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores


def test_rerank_documents_reorders_by_score(monkeypatch):
    documents = [
        Document(
            page_content="low relevance",
            metadata={"id": "A"},
        ),
        Document(
            page_content="high relevance",
            metadata={"id": "B"},
        ),
        Document(
            page_content="medium relevance",
            metadata={"id": "C"},
        ),
    ]

    fake = FakeCrossEncoder(
        scores=[0.1, 0.9, 0.5]
    )

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda *args, **kwargs: fake,
    )

    result = rerank_documents(
        "some query",
        documents,
        top_n=3,
    )

    assert [
        document.metadata["id"]
        for document in result
    ] == ["B", "C", "A"]


def test_rerank_documents_respects_top_n(monkeypatch):
    documents = [
        Document(
            page_content="a",
            metadata={"id": "A"},
        ),
        Document(
            page_content="b",
            metadata={"id": "B"},
        ),
        Document(
            page_content="c",
            metadata={"id": "C"},
        ),
    ]

    fake = FakeCrossEncoder(
        scores=[0.3, 0.9, 0.6]
    )

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda *args, **kwargs: fake,
    )

    result = rerank_documents(
        "some query",
        documents,
        top_n=2,
    )

    assert len(result) == 2

    assert [
        document.metadata["id"]
        for document in result
    ] == ["B", "C"]


def test_rerank_with_scores_returns_correct_scores(monkeypatch):
    documents = [
        Document(
            page_content="a",
            metadata={"id": "A"},
        ),
        Document(
            page_content="b",
            metadata={"id": "B"},
        ),
    ]

    fake = FakeCrossEncoder(
        scores=[0.2, 0.8]
    )

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda *args, **kwargs: fake,
    )

    result = rerank_with_scores(
        "some query",
        documents,
        top_n=2,
    )

    assert result[0][0].metadata["id"] == "B"
    assert result[0][1] == pytest.approx(0.8)

    assert result[1][0].metadata["id"] == "A"
    assert result[1][1] == pytest.approx(0.2)


def test_rerank_documents_reranker_failure_raises_runtime_error(
    monkeypatch,
):
    class BrokenCrossEncoder:

        def predict(self, pairs):
            raise RuntimeError("model exploded")

    fake = BrokenCrossEncoder()

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda *args, **kwargs: fake,
    )

    with pytest.raises(RuntimeError):
        rerank_documents(
            "query",
            [Document(page_content="x")],
        )


# ---------------------------------------------------------
# Live test using the real CrossEncoder
# ---------------------------------------------------------

def test_rerank_documents_with_real_model():
    documents = [
        Document(
            page_content=(
                "Return windows vary by brand "
                "and product type."
            )
        ),
        Document(
            page_content=(
                "The Nike Pegasus is a lightweight "
                "running shoe with responsive cushioning."
            )
        ),
        Document(
            page_content=(
                "Payment methods accepted include "
                "Visa, Mastercard, and PayPal."
            )
        ),
    ]

    result = rerank_documents(
        "best shoes for running",
        documents,
        top_n=1,
    )

    print(
        f"\nTop result: "
        f"{result[0].page_content}"
    )

    assert "running" in result[0].page_content.lower()