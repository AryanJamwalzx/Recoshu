import hashlib
import tiktoken

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


NO_SPLIT_DOCUMENT_TYPES = {"csv", "faq"}

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60

ENCODING = tiktoken.get_encoding("cl100k_base")


def length_function(text: str) -> int:
    return len(ENCODING.encode(text))


def create_chunk_id(doc: Document, index: int) -> str:
    content = f"{doc.metadata.get('source', '')}_{index}_{doc.page_content}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:

    if not documents:
        raise ValueError("No documents provided.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_documents = []

    for doc in documents:
        document_type = doc.metadata.get("document_type", "unknown")

        if document_type in NO_SPLIT_DOCUMENT_TYPES:
            doc.metadata["chunk_id"] = create_chunk_id(doc, 0)
            chunked_documents.append(doc)
            continue

        if not doc.page_content.strip():
            continue

        chunks = splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["chunk_id"] = create_chunk_id(chunk, i)
            chunked_documents.append(chunk)

    if not chunked_documents:
        raise ValueError("No chunks were created.")

    return chunked_documents