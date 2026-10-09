import json
from pathlib import Path
from typing import Any


DATASET_PATH = (
    Path(__file__).parent / "rag_retrieval_eval_dataset.json"
)


def load_evaluation_dataset() -> list[dict[str, Any]]:
    """
    Load and validate the retrieval evaluation dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    try:
        with DATASET_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in evaluation dataset: {DATASET_PATH}"
        ) from exc

    if not isinstance(dataset, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    for index, item in enumerate(dataset):

        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset item {index} must be an object."
            )

        query = item.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Dataset item {index} has an invalid query."
            )

        relevant = item.get("relevant")

        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Dataset item {index} must contain "
                "a non-empty 'relevant' list."
            )

        for definition in relevant:

            if not isinstance(definition, dict):
                raise ValueError(
                    f"Invalid relevance definition "
                    f"in dataset item {index}."
                )

            field = definition.get("field")
            values = definition.get("values")

            if not isinstance(field, str) or not field:
                raise ValueError(
                    f"Invalid 'field' in dataset item {index}."
                )

            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"'values' must be a non-empty list "
                    f"in dataset item {index}."
                )

    return dataset