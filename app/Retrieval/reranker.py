import logging

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ms-marco-MiniLM-L-6-v2: small, fast, free, runs locally on CPU. Trained
# specifically for query-passage relevance ranking (the exact task here),
# and is one of the most commonly used cross-encoders for RAG reranking.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# The "retrieve wide, rerank narrow" pattern: pull more candidates from
# Pinecone than you actually need, then let the reranker -- which is far
# more precise than embedding similarity alone, since it reads the query
# and each document together instead of comparing two separate vectors --
# narrow that down to the genuinely best few.
DEFAULT_INITIAL_K = 20
DEFAULT_FINAL_K = 5

# Created once and reused -- reloading the model on every call adds
# unnecessary latency per query, same lesson as the embedding model.
_reranker: CrossEncoder | None = None


def get_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoder:
    """
    Return a cached cross-encoder reranker. The first call downloads and
    loads the model (~80MB, cached locally after); later calls reuse the
    same instance.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(model_name)
    return _reranker


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = DEFAULT_FINAL_K,
) -> list[Document]:
    """
    Re-score and reorder retrieved documents by true relevance to the
    query, then return only the top_n most relevant.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not documents:
        raise ValueError("No documents provided to rerank.")
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in documents]

    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        raise RuntimeError(f"Reranking failed for query {query!r}: {exc}") from exc

    scored_docs = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)

    if not scored_docs:
        logger.warning("Reranking produced no results for query: %r", query)

    return [doc for doc, _ in scored_docs[:top_n]]


def rerank_with_scores(
    query: str,
    documents: list[Document],
    top_n: int = DEFAULT_FINAL_K,
) -> list[tuple[Document, float]]:
    """
    Same as rerank_documents, but also returns each result's relevance
    score -- useful for debugging or applying a confidence threshold before
    handing results to the LLM.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not documents:
        raise ValueError("No documents provided to rerank.")
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in documents]

    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        raise RuntimeError(f"Reranking failed for query {query!r}: {exc}") from exc

    scored_docs = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)

    return [(doc, float(score)) for doc, score in scored_docs[:top_n]]