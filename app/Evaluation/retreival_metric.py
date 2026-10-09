import logging
import math
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.documents import Document


logger = logging.getLogger(__name__)

DEFAULT_K = 5


# =========================================================
# Core metrics
# =========================================================

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Recall@K:
    Fraction of relevant documents retrieved in the top K.
    """

    if not relevant_ids:
        raise ValueError(
            "relevant_ids must be non-empty."
        )

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    top_k = set(retrieved_ids[:k])

    hits = top_k & relevant_ids

    return len(hits) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Precision@K:
    Fraction of retrieved documents in the top K
    that are relevant.
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    hits = sum(
        doc_id in relevant_ids
        for doc_id in top_k
    )

    return hits / len(top_k)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Reciprocal Rank@K:
    1 / rank of the first relevant result
    within the top K results.

    Returns 0 if no relevant result appears
    within the top K.
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    for rank, doc_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def _dcg_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    """
    Calculate Discounted Cumulative Gain@K.
    """

    dcg = 0.0

    for rank, doc_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        rel = relevance.get(
            doc_id,
            0.0,
        )

        dcg += (
            (2 ** rel - 1)
            / math.log2(rank + 1)
        )

    return dcg


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain@K.

    Supports graded relevance:
        3 = highly relevant
        2 = relevant
        1 = mildly relevant
        0 = irrelevant
    """

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    if not relevance:
        return 0.0

    dcg = _dcg_at_k(
        retrieved_ids,
        relevance,
        k,
    )

    ideal_order = sorted(
        relevance.keys(),
        key=lambda doc_id: relevance[doc_id],
        reverse=True,
    )

    idcg = _dcg_at_k(
        ideal_order,
        relevance,
        k,
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =========================================================
# Helper
# =========================================================

def deduplicate_ids(
    retrieved_ids: list[str],
) -> list[str]:
    """
    Remove duplicate IDs while preserving ranking order.

    Used for source-level evaluation because multiple
    chunks can come from the same source file.
    """

    seen = set()
    unique_ids = []

    for doc_id in retrieved_ids:

        if doc_id in seen:
            continue

        seen.add(doc_id)
        unique_ids.append(doc_id)

    return unique_ids


# =========================================================
# Evaluation data structures
# =========================================================

@dataclass
class RetrievalTestCase:
    query: str

    # Metadata field containing the ground-truth IDs.
    #
    # Examples:
    #   product_id
    #   faq_id
    #   source_file
    id_field: str

    relevant_ids: set[str]

    # Optional graded relevance.
    relevance: dict[str, float] = field(
        default_factory=dict
    )


@dataclass
class RetrievalEvalResult:
    query: str
    id_field: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


# =========================================================
# Evaluation orchestrator
# =========================================================

def evaluate_retrieval(
    test_cases: list[RetrievalTestCase],
    retrieve_fn: Callable[[str], list[Document]],
    k: int = DEFAULT_K,
) -> dict:
    """
    Evaluate a retriever using:

        Recall@K
        Precision@K
        MRR@K
        nDCG@K

    Each test case specifies its own metadata field,
    so products, FAQs, and policies can be evaluated.
    """

    if not test_cases:
        raise ValueError(
            "No test cases provided."
        )

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    results: list[RetrievalEvalResult] = []

    for case in test_cases:

        try:
            documents = retrieve_fn(
                case.query
            )

        except Exception as exc:
            raise RuntimeError(
                f"Retrieval failed for "
                f"query {case.query!r}: {exc}"
            ) from exc

        retrieved_ids = []

        # -------------------------------------------------
        # Extract the correct ID field
        # -------------------------------------------------

        for document in documents:

            value = document.metadata.get(
                case.id_field
            )

            # A retrieved document may belong to a
            # different document type and therefore not
            # contain this test case's ID field.
            #
            # Example:
            # FAQ query → PDF/product documents are ignored
            # Policy query → FAQ/product documents are ignored
            if value is None:
                continue

            retrieved_ids.append(
                str(value)
            )

        # -------------------------------------------------
        # Source-level deduplication
        # -------------------------------------------------

        # PDFs are chunked, so multiple retrieved chunks
        # can have the same source_file. For source-level
        # evaluation, count the source only once.
        if case.id_field == "source_file":

            retrieved_ids = deduplicate_ids(
                retrieved_ids
            )

        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        relevant_ids = {
            str(doc_id)
            for doc_id in case.relevant_ids
        }

        # -------------------------------------------------
        # Relevance scores
        # -------------------------------------------------

        if case.relevance:

            relevance = {
                str(doc_id): float(score)
                for doc_id, score
                in case.relevance.items()
            }

        else:

            # Binary relevance:
            # relevant = 1
            # everything else = 0
            relevance = {
                doc_id: 1.0
                for doc_id in relevant_ids
            }

        # -------------------------------------------------
        # Calculate metrics
        # -------------------------------------------------

        results.append(
            RetrievalEvalResult(
                query=case.query,
                id_field=case.id_field,

                recall_at_k=recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    k,
                ),

                precision_at_k=precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    k,
                ),

                reciprocal_rank=reciprocal_rank(
                    retrieved_ids,
                    relevant_ids,
                    k,
                ),

                ndcg_at_k=ndcg_at_k(
                    retrieved_ids,
                    relevance,
                    k,
                ),
            )
        )

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    def average(attribute: str) -> float:

        return sum(
            getattr(result, attribute)
            for result in results
        ) / len(results)

    return {
        "per_query": results,

        "recall_at_k": average(
            "recall_at_k"
        ),

        "precision_at_k": average(
            "precision_at_k"
        ),

        "mrr": average(
            "reciprocal_rank"
        ),

        "ndcg_at_k": average(
            "ndcg_at_k"
        ),

        "k": k,

        "num_queries": len(results),
    }