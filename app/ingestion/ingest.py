import argparse
import logging
import time

from app.loaders.loader import load_all_data
from app.Preprocessing.Chunking import chunk_documents
from app.Vector_Store.pinecone import DEFAULT_INDEX_NAME, upsert_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "app/data"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60


def run_ingestion(
    data_dir: str = DEFAULT_DATA_DIR,
    index_name: str = DEFAULT_INDEX_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """
    Run the full ingestion pipeline: load all supported files from data_dir,
    chunk them, and upsert into the Pinecone index. Returns the number of
    chunks upserted.
    """
    start_time = time.time()

    logger.info("Step 1/3: Loading documents from '%s'", data_dir)
    documents = load_all_data(data_dir)
    logger.info("Loaded %d documents.", len(documents))

    logger.info(
        "Step 2/3: Chunking documents (chunk_size=%d, chunk_overlap=%d)",
        chunk_size, chunk_overlap,
    )
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("Produced %d chunks.", len(chunks))

    logger.info("Step 3/3: Upserting chunks into Pinecone index '%s'", index_name)
    upserted_count = upsert_documents(chunks, index_name=index_name)
    logger.info("Upserted %d chunks.", upserted_count)

    elapsed = time.time() - start_time
    logger.info("Ingestion complete in %.1fs.", elapsed)

    return upserted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest project data into the Pinecone vector store.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root data directory to ingest.")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help="Pinecone index name.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    try:
        run_ingestion(
            data_dir=args.data_dir,
            index_name=args.index_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()