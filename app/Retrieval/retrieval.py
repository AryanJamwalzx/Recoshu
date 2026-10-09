import logging

from langchain_core.documents import Document

from app.Vector_Store.pinecone import (
    DEFAULT_INDEX_NAME,
    get_vector_store,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    index_name: str = DEFAULT_INDEX_NAME,
    filter: dict | None = None,
) -> list[Document]:
    """
    Retrieve the most relevant documents from Pinecone.

    Args:
        query: User's search question.
        top_k: Number of results to return.
        index_name: Pinecone index to search.
        filter: Optional Pinecone metadata filter.

    Returns:
        List of relevant LangChain Documents.
    """
    # Validate query
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Validate top_k
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Connect to Pinecone vector store
    vector_store = get_vector_store(index_name)

    try:
        results = vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=filter,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Retrieval failed for query {query!r}: {exc}"
        ) from exc

    if not results:
        logger.warning(
            "No results found for query: %r",
            query,
        )

    return results


def retrieve_with_scores(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    index_name: str = DEFAULT_INDEX_NAME,
    filter: dict | None = None,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents from Pinecone along with their similarity scores.

    Useful for:
    - Debugging retrieval
    - Evaluating retrieval quality
    - Understanding similarity scores
    - Applying a score threshold later
    """
    # Validate query
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Validate top_k
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Connect to Pinecone vector store
    vector_store = get_vector_store(index_name)

    try:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Retrieval with scores failed for query {query!r}: {exc}"
        ) from exc

    if not results:
        logger.warning(
            "No results found for query: %r",
            query,
        )

    return results