from app.Vector_Store.pinecone import (
    get_pinecone_client,
    ensure_index_exists,
    DEFAULT_INDEX_NAME,
)


def test_pinecone():

    print("Connecting to Pinecone...")

    pc = get_pinecone_client()

    print("✅ Connected successfully.")

    print("Checking index...")

    ensure_index_exists(DEFAULT_INDEX_NAME)

    print(f"✅ Index '{DEFAULT_INDEX_NAME}' exists.")

    print("\nIndexes:")

    for index in pc.list_indexes():
        print(index)


if __name__ == "__main__":
    test_pinecone()