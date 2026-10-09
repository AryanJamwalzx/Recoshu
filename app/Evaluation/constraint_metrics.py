from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document


@dataclass
class ConstraintEvalResult:
    query: str
    total_results: int
    fully_matching_results: int
    constraint_satisfaction_rate: float


def value_matches_constraint(
    actual_value: Any,
    expected_value: Any,
) -> bool:
    """
    Check whether a metadata value satisfies a constraint.

    Supports:
        exact match
        $eq
        $lt
        $lte
        $gt
        $gte
    """

    if isinstance(expected_value, dict):

        for operator, expected in expected_value.items():

            if operator == "$eq":
                if actual_value != expected:
                    return False

            elif operator == "$lt":
                if actual_value is None or actual_value >= expected:
                    return False

            elif operator == "$lte":
                if actual_value is None or actual_value > expected:
                    return False

            elif operator == "$gt":
                if actual_value is None or actual_value <= expected:
                    return False

            elif operator == "$gte":
                if actual_value is None or actual_value < expected:
                    return False

            else:
                raise ValueError(
                    f"Unsupported constraint operator: {operator}"
                )

        return True

    return actual_value == expected_value


def document_matches_constraints(
    document: Document,
    constraints: dict[str, Any],
) -> bool:
    """
    Check whether a document satisfies all constraints.
    """

    for field, expected_value in constraints.items():

        if field not in document.metadata:
            return False

        actual_value = document.metadata[field]

        if not value_matches_constraint(
            actual_value,
            expected_value,
        ):
            return False

    return True


def evaluate_constraints(
    query: str,
    documents: list[Document],
    constraints: dict[str, Any],
) -> ConstraintEvalResult:
    """
    Evaluate how many retrieved documents satisfy all
    requested constraints.
    """

    if not query or not query.strip():
        raise ValueError(
            "query must be a non-empty string."
        )

    if not constraints:
        raise ValueError(
            "constraints must not be empty."
        )

    if not documents:
        return ConstraintEvalResult(
            query=query,
            total_results=0,
            fully_matching_results=0,
            constraint_satisfaction_rate=0.0,
        )

    matching_results = sum(
        document_matches_constraints(
            document,
            constraints,
        )
        for document in documents
    )

    return ConstraintEvalResult(
        query=query,
        total_results=len(documents),
        fully_matching_results=matching_results,
        constraint_satisfaction_rate=(
            matching_results / len(documents)
        ),
    )


def evaluate_constraint_dataset(
    test_cases: list[dict[str, Any]],
    retrieve_fn: Callable[..., list[Document]],
    top_k: int = 5,
) -> dict:
    """
    Evaluate a retriever against a product-constraint dataset.
    """

    if not test_cases:
        raise ValueError(
            "No constraint test cases provided."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be positive."
        )

    results: list[ConstraintEvalResult] = []

    for case in test_cases:

        query = case.get("query")
        constraints = case.get("constraints")

        if not query or not query.strip():
            raise ValueError(
                "Each test case must contain a non-empty query."
            )

        if not isinstance(constraints, dict) or not constraints:
            raise ValueError(
                f"Invalid constraints for query: {query!r}"
            )

        try:
            documents = retrieve_fn(
                query=query,
                top_k=top_k,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Retrieval failed for query {query!r}: {exc}"
            ) from exc

        result = evaluate_constraints(
            query=query,
            documents=documents,
            constraints=constraints,
        )

        results.append(result)

    average_csr = sum(
        result.constraint_satisfaction_rate
        for result in results
    ) / len(results)

    return {
        "per_query": results,
        "constraint_satisfaction_rate": average_csr,
        "top_k": top_k,
        "num_queries": len(results),
    }