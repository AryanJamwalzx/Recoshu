from app.loaders.csv_loader import load_csv


def test_csv_loader():
    file_path = "app/data/catalog/nike_adidas_shoes_catalog.csv"

    documents = load_csv(file_path)

    print(f"Number of documents: {len(documents)}")

    print("\nFirst document:")
    print(documents[0].page_content)

    print("\nMetadata:")
    print(documents[0].metadata)

    assert len(documents) > 0
    assert documents[0].metadata["document_type"] == "csv"
    assert "source_file" in documents[0].metadata
    assert "brand" in documents[0].metadata
    assert "product_id" in documents[0].metadata


if __name__ == "__main__":
    test_csv_loader()
    