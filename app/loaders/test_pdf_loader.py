import pytest

from app.loaders.pdf_loader import load_pdf


def test_pdf_loader():
    file_path = "app/data/policies/return_policy.pdf"

    documents = load_pdf(file_path)

    print(f"Number of documents: {len(documents)}")

    print("\nFirst document:")
    print(documents[0])

    assert len(documents) > 0
    assert documents[0].metadata["document_type"] == "pdf"
    assert documents[0].metadata["source_file"] == "return_policy.pdf"


def test_pdf_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf("app/data/policies/does_not_exist.pdf")


def test_pdf_loader_wrong_extension_raises():
    with pytest.raises(ValueError):
        load_pdf("app/data/catalog/nike_adidas_shoes_catalog.csv")