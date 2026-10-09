import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF and return its pages as LangChain Documents.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
    if path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty: {path}")

    try:
        loader = PyPDFLoader(str(path))
        documents = loader.load()
    except Exception as exc:
        # Covers corrupt files, password-protected PDFs PyPDFLoader can't open,
        # and any other pypdf-level read failure.
        raise ValueError(f"Failed to parse PDF file {path}: {exc}") from exc

    if not documents:
        raise ValueError(f"PDF file has no pages: {path}")

    # PyPDFLoader still yields a Document per page even when a page has no
    # extractable text (e.g. scanned images); drop those rather than passing
    # empty content downstream.
    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        raise ValueError(
            f"PDF file {path} produced no extractable text. It may be a "
            f"scanned/image-only PDF that needs OCR before it can be indexed."
        )

    for document in documents:
        document.metadata["document_type"] = "pdf"
        document.metadata["source_file"] = path.name

    return documents