import logging
from pathlib import Path

from langchain_core.documents import Document

from .csv_loader import load_csv
from .json_loader import load_json
from .pdf_loader import load_pdf

logger = logging.getLogger(__name__)


def load_all_data(data_dir: str, on_error: str = "skip") -> list[Document]:
    """
    Load all supported files (CSV, JSON, PDF) from the data directory, including
    subfolders (e.g. data/catalog, data/faq, data/policies).

    Args:
        data_dir: Root folder to scan.
        on_error: "skip"  -> log the failure and continue with the next file (default).
                  "raise" -> stop the whole run on the first file that fails to load.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got: {on_error}")

    documents: list[Document] = []
    files_failed = 0

    loaders_by_extension = {
        "*.csv": load_csv,
        "*.json": load_json,
        "*.pdf": load_pdf,
    }

    for pattern, load_fn in loaders_by_extension.items():
        for file_path in sorted(data_path.rglob(pattern)):
            print(f"Loading {file_path.suffix.upper().lstrip('.')}: {file_path.relative_to(data_path)}")
            try:
                documents.extend(load_fn(str(file_path)))
            except (FileNotFoundError, ValueError) as exc:
                files_failed += 1
                logger.error("Failed to load %s: %s", file_path, exc)
                if on_error == "raise":
                    raise

    print(f"\nTotal documents loaded: {len(documents)}")
    if files_failed:
        print(f"Files skipped due to errors: {files_failed}")

    if not documents:
        raise ValueError(f"No documents could be loaded from {data_path}")

    return documents