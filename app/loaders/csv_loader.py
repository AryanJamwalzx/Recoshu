import csv
import logging
from pathlib import Path

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


PRODUCT_METADATA_FIELDS = {
    "product_id",
    "sku",
    "brand",
    "product_name",
    "model",
    "collection",
    "gender",
    "age_group",
    "category",
    "sub_category",
    "sport",
    "shoe_type",
    "terrain",
    "usage",
    "color",
    "size",
    "width",
    "availability",
    "launch_year",
    "bestseller",
    "trending",
    "price",
    "final_price",
}


def convert_value(value: str):
    """
    Convert CSV string values into useful Python types.

    Examples:
        "True"  -> True
        "2022"  -> 2022
        "99.95" -> 99.95
        "Nike"  -> "Nike"
    """

    value = value.strip()

    if value == "":
        return None

    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def load_csv(file_path: str) -> list[Document]:
    """
    Load a product CSV.

    Each CSV row becomes one LangChain Document.

    Product information is stored in page_content for semantic retrieval,
    while important structured fields are stored in metadata for
    Pinecone filtering.
    """

    path = Path(file_path)

    # -----------------------------
    # Validate file
    # -----------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {path.suffix}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"CSV file is empty: {path}"
        )

    documents: list[Document] = []

    # -----------------------------
    # Read CSV
    # -----------------------------

    try:
        file = path.open(
            "r",
            encoding="utf-8",
            newline="",
        )

    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 failed for %s. Retrying with latin-1.",
            path,
        )

        file = path.open(
            "r",
            encoding="latin-1",
            newline="",
        )

    try:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                f"CSV has no header row: {path}"
            )

        # -----------------------------
        # Convert every row
        # -----------------------------

        for row_number, row in enumerate(reader):

            # Remove completely empty rows
            if not any(
                value and value.strip()
                for value in row.values()
            ):
                continue

            # -------------------------
            # Build searchable content
            # -------------------------

            content_lines = []

            for key, value in row.items():

                if key is None or value is None:
                    continue

                value = value.strip()

                if value:
                    content_lines.append(
                        f"{key}: {value}"
                    )

            page_content = "\n".join(content_lines)

            # -------------------------
            # Build metadata
            # -------------------------

            metadata = {
                "source": str(path),
                "source_file": path.name,
                "document_type": "csv",
                "row": row_number,
            }

            for field in PRODUCT_METADATA_FIELDS:

                raw_value = row.get(field)

                if raw_value is None:
                    continue

                converted_value = convert_value(
                    raw_value
                )

                if converted_value is not None:
                    metadata[field] = converted_value

            # -------------------------
            # Create Document
            # -------------------------

            document = Document(
                page_content=page_content,
                metadata=metadata,
            )

            documents.append(document)

    except Exception as exc:
        raise ValueError(
            f"Failed to parse CSV file {path}: {exc}"
        ) from exc

    finally:
        file.close()

    # -----------------------------
    # Final validation
    # -----------------------------

    if not documents:
        raise ValueError(
            f"CSV cpython -m app.loaders.test_csv_loaderontained no product rows: {path}"
        )

    return documents