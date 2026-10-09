import json
from pathlib import Path

from app.Evaluation.constraint_metrics import (
    evaluate_constraint_dataset,
)
from app.Pipeline.pipeline import retrieve_with_pipeline
from app.Retrieval.retrieval import retrieve


TOP_K = 5
INITIAL_K = 20


def load_constraint_dataset():
    """
    Load the product constraint evaluation dataset.
    """

    path = (
        Path(__file__).parent
        / "rag_constraint_eval_dataset.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Constraint dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Constraint evaluation dataset must "
            "contain a JSON list."
        )

    return data


def baseline_retrieve(
    query: str,
    top_k: int = TOP_K,
):
    """
    Baseline:
    semantic retrieval only.
    """

    return retrieve(
        query=query,
        top_k=top_k,
    )


def improved_retrieve(
    query: str,
    top_k: int = TOP_K,
):
    """
    Improved pipeline:
    query understanding
    -> metadata filtering
    -> Pinecone retrieval
    -> reranking
    """

    return retrieve_with_pipeline(
        query=query,
        initial_k=INITIAL_K,
        final_k=top_k,
    )


def print_results(
    title: str,
    results: dict,
):
    """
    Print per-query and overall constraint results.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for result in results["per_query"]:

        print("\n" + "-" * 70)

        print(
            f"Query: {result.query}"
        )

        print(
            f"Retrieved results: "
            f"{result.total_results}"
        )

        print(
            f"Fully matching results: "
            f"{result.fully_matching_results}"
        )

        print(
            f"Constraint Satisfaction Rate: "
            f"{result.constraint_satisfaction_rate:.4f}"
        )

    print("\n" + "-" * 70)

    print(
        f"Queries evaluated: "
        f"{results['num_queries']}"
    )

    print(
        f"Top-K: "
        f"{results['top_k']}"
    )

    print(
        f"Overall Constraint Satisfaction Rate: "
        f"{results['constraint_satisfaction_rate']:.4f}"
    )


def main():

    dataset = load_constraint_dataset()

    print("\n" + "=" * 70)
    print("PRODUCT CONSTRAINT EVALUATION")
    print("=" * 70)

    print(
        f"\nDataset queries: {len(dataset)}"
    )

    # -----------------------------------------------------
    # Baseline evaluation
    # -----------------------------------------------------

    baseline_results = evaluate_constraint_dataset(
        test_cases=dataset,
        retrieve_fn=baseline_retrieve,
        top_k=TOP_K,
    )

    print_results(
        "BASELINE RETRIEVAL",
        baseline_results,
    )

    # -----------------------------------------------------
    # Improved pipeline evaluation
    # -----------------------------------------------------

    improved_results = evaluate_constraint_dataset(
        test_cases=dataset,
        retrieve_fn=improved_retrieve,
        top_k=TOP_K,
    )

    print_results(
        "IMPROVED PIPELINE",
        improved_results,
    )

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    baseline_csr = (
        baseline_results[
            "constraint_satisfaction_rate"
        ]
    )

    improved_csr = (
        improved_results[
            "constraint_satisfaction_rate"
        ]
    )

    improvement = improved_csr - baseline_csr

    print("\n" + "=" * 70)
    print("BASELINE VS IMPROVED")
    print("=" * 70)

    print(
        f"\nBaseline CSR:  {baseline_csr:.4f}"
    )

    print(
        f"Improved CSR:  {improved_csr:.4f}"
    )

    print(
        f"Improvement:    {improvement:+.4f}"
    )


if __name__ == "__main__":
    main()