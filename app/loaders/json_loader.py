import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_json(file_path: str) -> list[Document]:
    """
    Load FAQ JSON data and convert each FAQ into a LangChain Document.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file, got: {path.suffix}")
    if path.stat().st_size == 0:
        raise ValueError(f"JSON file is empty: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path} (line {exc.lineno}, col {exc.colno}): {exc.msg}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not decode {path} as UTF-8: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of FAQ objects.")
    if not data:
        raise ValueError(f"JSON file contained an empty list: {path}")

    documents = []
    skipped = 0

    for i, faq in enumerate(data):
        if not isinstance(faq, dict):
            skipped += 1
            logger.warning("Skipping entry %d in %s: not an object.", i, path)
            continue

        question = faq.get("question", "").strip()
        answer = faq.get("answer", "").strip()
        if not question or not answer:
            skipped += 1
            logger.warning("Skipping entry %d in %s: missing question/answer.", i, path)
            continue

        content = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Keywords: {', '.join(faq.get('keywords', []))}"
        )
        metadata = {
            "source": path.name,
            "faq_id": faq.get("faq_id"),
            "category": faq.get("category"),
            "priority": faq.get("priority"),
            "last_updated": faq.get("last_updated"),
            "related_documents": faq.get("related_documents", []),
            "document_type": "faq",
        }
        documents.append(Document(page_content=content, metadata=metadata))

    if not documents:
        raise ValueError(f"No usable FAQ entries found in {path} (all {skipped} skipped).")

    if skipped:
        logger.warning("Loaded %d FAQ(s) from %s, skipped %d malformed entr(ies).", len(documents), path, skipped)

    return documents