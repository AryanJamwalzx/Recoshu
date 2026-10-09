from app.Evaluation.dataset import (
    load_evaluation_dataset,
)

from app.Evaluation.retreival_metric import (
    RetrievalTestCase,
    evaluate_retrieval,
)

from app.Retrieval.retrieval import retrieve


TOP_K = 5

EXACT_ID_FIELDS = {
    "product_id",
    "faq_id",
    "source_file",
}


def build_test_cases(
    dataset: list[dict],
) -> list[RetrievalTestCase]:
    """
    Convert JSON evaluation data into RetrievalTestCase objects.

    Each test case keeps its own metadata field.
    """

    test_cases = []

    for item in dataset:

        for definition in item["relevant"]:

            field = definition["field"]

            # Only use exact identifiers for the classical
            # retrieval metrics.
            if field not in EXACT_ID_FIELDS:
                continue

            relevant_ids = {
                str(value)
                for value in definition["values"]
            }

            relevance = {
                str(value): 1.0
                for value in definition["values"]
            }

            test_cases.append(
                RetrievalTestCase(
                    query=item["query"],
                    id_field=field,
                    relevant_ids=relevant_ids,
                    relevance=relevance,
                )
            )

            # One exact ground-truth definition per query
            # for this first evaluation.
            break

    return test_cases


def main():

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    dataset = load_evaluation_dataset()

    print(
        f"\nDataset queries: {len(dataset)}"
    )

    test_cases = build_test_cases(
        dataset
    )

    print(
        f"Queries with exact ground truth: "
        f"{len(test_cases)}"
    )

    if not test_cases:
        raise RuntimeError(
            "No valid evaluation cases found."
        )

    results = evaluate_retrieval(
        test_cases=test_cases,

        retrieve_fn=lambda query: retrieve(
            query=query,
            top_k=TOP_K,
        ),

        k=TOP_K,
    )

    # -----------------------------------------------------
    # Per-query results
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("PER-QUERY RESULTS")
    print("-" * 70)

    for result in results["per_query"]:

        print(
            f"\nQuery: {result.query}"
        )

        print(
            f"Ground-truth field: "
            f"{result.id_field}"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{result.recall_at_k:.4f}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{result.precision_at_k:.4f}"
        )

        print(
            f"MRR@{TOP_K}: "
            f"{result.reciprocal_rank:.4f}"
        )

        print(
            f"nDCG@{TOP_K}: "
            f"{result.ndcg_at_k:.4f}"
        )

    # -----------------------------------------------------
    # Overall results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"\nQueries evaluated: "
        f"{results['num_queries']}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{results['recall_at_k']:.4f}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{results['precision_at_k']:.4f}"
    )

    print(
        f"MRR@{TOP_K}: "
        f"{results['mrr']:.4f}"
    )

    print(
        f"nDCG@{TOP_K}: "
        f"{results['ndcg_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()