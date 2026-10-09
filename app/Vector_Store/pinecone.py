import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.Embeddings.embeddings import get_embedding_model


# Load variables from .env
load_dotenv()


# -----------------------------
# Pinecone configuration
# -----------------------------

DEFAULT_INDEX_NAME = "rag-asst-index"
EMBEDDING_DIMENSION = 384
METRIC = "cosine"

CLOUD = "aws"
REGION = "us-east-1"


# -----------------------------
# Create Pinecone client
# -----------------------------

def get_pinecone_client() -> Pinecone:
    """Create and return a Pinecone client."""

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env file."
        )

    return Pinecone(api_key=api_key)


# -----------------------------
# Create Pinecone index
# -----------------------------

def ensure_index_exists(
    index_name: str = DEFAULT_INDEX_NAME,
    dimension: int = EMBEDDING_DIMENSION,
    metric: str = METRIC,
) -> None:
    """
    Create the Pinecone index if it does not already exist.
    """

    pc = get_pinecone_client()

    if pc.has_index(index_name):
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION,
        ),
    )


# -----------------------------
# Get LangChain vector store
# -----------------------------

def get_vector_store(
    index_name: str = DEFAULT_INDEX_NAME,
) -> PineconeVectorStore:
    """
    Return a LangChain Pinecone vector store.
    """

    ensure_index_exists(index_name)

    embedding_model = get_embedding_model()

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model,
    )


# -----------------------------
# Upsert documents
# -----------------------------

def upsert_documents(
    documents: list[Document],
    index_name: str = DEFAULT_INDEX_NAME,
    batch_size: int = 100,
) -> int:
    """
    Embed and upsert documents into Pinecone in batches.

    Each document must contain a unique metadata['chunk_id'].
    """

    if not documents:
        raise ValueError(
            "No documents provided to upsert."
        )

    # Collect chunk IDs
    ids = []

    for i, doc in enumerate(documents):

        chunk_id = doc.metadata.get("chunk_id")

        if not chunk_id:
            raise ValueError(
                f"Document at index {i} is missing "
                "metadata['chunk_id']."
            )

        ids.append(chunk_id)

    # Check for duplicate IDs
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk_id values found. "
            "Each document must have a unique ID."
        )

    # Connect to Pinecone
    vector_store = get_vector_store(index_name)

    total_upserted = 0

    # Upload in batches
    for start in range(
        0,
        len(documents),
        batch_size,
    ):

        batch_documents = documents[
            start:start + batch_size
        ]

        batch_ids = ids[
            start:start + batch_size
        ]

        try:

            vector_store.add_documents(
                documents=batch_documents,
                ids=batch_ids,
            )

        except Exception as exc:

            raise RuntimeError(
                f"Failed to upsert batch "
                f"{start}-{start + len(batch_documents)}: "
                f"{exc}"
            ) from exc

        total_upserted += len(batch_documents)

    return total_upserted