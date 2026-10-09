import pytest
from langchain_core.documents import Document

from app.context_builder.context import build_context


def test_build_context_empty_raises():
    with pytest.raises(ValueError):
        build_context([])


def test_build_context_faq_includes_real_question_and_answer():
    # Guards against the metadata.get("question"/"answer") bug -- those
    # fields live in page_content, not metadata, for FAQ documents.
    faq_doc = Document(
        page_content=(
            "Question: Do Nike shoes run large or small?\n"
            "Answer: Nike shoes generally run true to size.\n"
            "Keywords: sizing"
        ),
        metadata={"document_type": "faq", "category": "Size & Fit", "faq_id": "FAQ0124"},
    )

    context = build_context([faq_doc])

    print("\n--- FAQ context ---")
    print(context)

    assert "Do Nike shoes run large or small?" in context
    assert "Nike shoes generally run true to size" in context
    assert "N/A" not in context


def test_build_context_csv_includes_description():
    csv_doc = Document(
        page_content="The Nike Air Max 90 is a black low-top sneaker built for casual wear.",
        metadata={
            "document_type": "csv",
            "brand": "Nike",
            "product_name": "Nike Air Max 90",
            "sub_category": "Lifestyle",
            "sport": "Casual",
            "color": "Black",
            "size": 9.5,
            "final_price": 129.99,
            "availability": "In Stock",
        },
    )

    context = build_context([csv_doc])

    print("\n--- CSV context ---")
    print(context)

    assert "Nike" in context
    assert "Air Max 90" in context
    assert "black low-top sneaker" in context
    assert "129.99" in context


def test_build_context_pdf_uses_source_file():
    pdf_doc = Document(
        page_content="Return Policy\nItems can be returned within 30 days.",
        metadata={
            "document_type": "pdf",
            "source": "app/data/policies/return_policy.pdf",
            "source_file": "return_policy.pdf",
            "page": 0,
        },
    )

    context = build_context([pdf_doc])

    print("\n--- PDF context ---")
    print(context)

    assert "return_policy.pdf" in context
    assert "Items can be returned within 30 days" in context


def test_build_context_multiple_documents_are_separated():
    docs = [
        Document(page_content="desc", metadata={"document_type": "csv", "brand": "Nike"}),
        Document(
            page_content="Question: x?\nAnswer: y",
            metadata={"document_type": "faq", "category": "Orders"},
        ),
    ]

    context = build_context(docs)

    assert "PRODUCT 1" in context
    assert "FAQ 2" in context
    assert "---" in context


def test_build_context_unknown_type_falls_back_gracefully():
    doc = Document(
        page_content="Some miscellaneous content.",
        metadata={"document_type": "something_else", "source_file": "misc.txt"},
    )

    context = build_context([doc])

    assert "misc.txt" in context
    assert "Some miscellaneous content." in context


def test_build_context_missing_metadata_fields_degrade_to_na():
    doc = Document(page_content="desc", metadata={"document_type": "csv"})

    context = build_context([doc])

    assert "N/A" in context


def test_build_context_respects_max_chars():
    docs = [
        Document(page_content="x" * 3000, metadata={"document_type": "pdf", "source_file": f"doc{i}.pdf"})
        for i in range(5)
    ]

    context = build_context(docs, max_chars=6000)

    assert context.count("POLICY DOCUMENT") < 5


def test_build_context_never_drops_first_document_even_if_oversized():
    doc = Document(page_content="y" * 10000, metadata={"document_type": "pdf", "source_file": "huge.pdf"})

    context = build_context([doc], max_chars=100)

    assert "huge.pdf" in context
    assert len(context) > 100


def test_build_context_with_real_pipeline_data():
    """Integration test: run real loaded + chunked project data through build_context."""
    from app.loaders.loader import load_all_data
    from app.Preprocessing.Chunking import chunk_documents

    documents = load_all_data("app/data")
    chunks = chunk_documents(documents)

    csv_chunk = next(c for c in chunks if c.metadata.get("document_type") == "csv")
    faq_chunk = next(c for c in chunks if c.metadata.get("document_type") == "faq")
    pdf_chunk = next(c for c in chunks if c.metadata.get("document_type") == "pdf")

    context = build_context([csv_chunk, faq_chunk, pdf_chunk])

    print("\n--- Real pipeline context ---")
    print(context)

    assert "PRODUCT 1" in context
    assert "FAQ 2" in context
    assert "POLICY DOCUMENT 3" in context
