import logging

from app.context_builder.context import build_context
from app.Retrieval.query_analyzer import (
    analyze_query,
    build_pinecone_filter,
)
from app.Retrieval.retrieval import retrieve
from app.Retrieval.reranker import rerank_documents

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_K = 20
DEFAULT_FINAL_K = 5


def retrieve_with_pipeline(
    query: str,
    initial_k: int = DEFAULT_INITIAL_K,
    final_k: int = DEFAULT_FINAL_K,
):
    """
    Query understanding → metadata filtering → retrieval → reranking.

    Returns the final Documents.
    """

    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    analysis = analyze_query(query)

    pinecone_filter = build_pinecone_filter(
        analysis.filters
    )

    candidates = retrieve(
        query=analysis.semantic_query,
        top_k=initial_k,
        filter=pinecone_filter,
    )

    if not candidates:
        return []

    return rerank_documents(
        analysis.semantic_query,
        candidates,
        top_n=final_k,
    )


def run_pipeline(
    query: str,
    initial_k: int = DEFAULT_INITIAL_K,
    final_k: int = DEFAULT_FINAL_K,
) -> str:
    """
    Full retrieval pipeline returning the final context.
    """

    documents = retrieve_with_pipeline(
        query=query,
        initial_k=initial_k,
        final_k=final_k,
    )

    if not documents:
        raise ValueError(
            f"No relevant results found for query: {query!r}"
        )

    return build_context(documents)