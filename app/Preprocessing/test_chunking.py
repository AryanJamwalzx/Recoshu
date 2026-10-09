from langchain_core.documents import Document

from app.Preprocessing.Chunking import chunk_documents


def test_chunk_documents():
    documents = [
        Document(
            page_content="This is a long document. " * 100,
            metadata={
                "document_type": "pdf",
                "source_file": "return_policy.pdf",
            },
        )
    ]

    chunks = chunk_documents(documents)

    print(f"Number of chunks: {len(chunks)}")
    print("\nFirst chunk:")
    print(chunks[0].page_content)

    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert "total_chunks" in chunks[0].metadata


def test_csv_documents_are_not_split():
    documents = [
        Document(
            page_content="Nike Air Max | Nike | Running Shoes",
            metadata={
                "document_type": "csv",
                "source_file": "catalog.csv",
            },
        )
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 1