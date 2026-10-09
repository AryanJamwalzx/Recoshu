from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


def get_embedding_model(
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Create and return a Hugging Face embedding model.

    The model runs locally on CPU and produces normalized embeddings.
    """

    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
        query_encode_kwargs={
            "prompt": QUERY_INSTRUCTION,
        },
    )