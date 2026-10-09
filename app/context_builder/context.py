import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Rough safety cap on total context size passed to the LLM. If retrieval
# ever returns more/larger documents than expected (e.g. top_k set too
# high), this stops the context from silently blowing past the model's
# context window instead of failing loudly or truncating mid-sentence.
DEFAULT_MAX_CHARS = 6000


def build_context(documents: list[Document], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """
    Convert retrieved documents into a structured context string for the LLM.
    """
    if not documents:
        raise ValueError("No documents provided.")

    context_parts = []
    total_chars = 0
    included = 0

    for i, doc in enumerate(documents, start=1):
        block = _format_document(doc, i)

        if included > 0 and total_chars + len(block) > max_chars:
            logger.warning(
                "build_context(): stopped after %d/%d documents to stay under max_chars=%d.",
                included, len(documents), max_chars,
            )
            break

        context_parts.append(block)
        total_chars += len(block)
        included += 1

    return "\n\n---\n\n".join(context_parts)


def _format_document(doc: Document, index: int) -> str:
    metadata = doc.metadata
    document_type = metadata.get("document_type", "unknown")

    if document_type == "csv":
        context = f"""PRODUCT {index}
Brand: {metadata.get("brand", "N/A")}
Product Name: {metadata.get("product_name", "N/A")}
Category: {metadata.get("sub_category", "N/A")}
Sport: {metadata.get("sport", "N/A")}
Color: {metadata.get("color", "N/A")}
Size: {metadata.get("size", "N/A")}
Price: {metadata.get("final_price", metadata.get("price", "N/A"))}
Availability: {metadata.get("availability", "N/A")}
Description: {doc.page_content}"""

    elif document_type == "faq":
        # question/answer live in page_content ("Question: ...\nAnswer: ...")
        # -- json_loader.py does not store them in metadata, so pulling them
        # from metadata.get("question"/"answer") always returns "N/A".
        context = f"""FAQ {index} (Category: {metadata.get("category", "N/A")})
{doc.page_content}"""

    else:
        # PDF / policy chunks. pdf_loader.py sets "source_file", not
        # "policy_name" -- that key doesn't exist in this project.
        source_label = metadata.get("source_file", metadata.get("source", "N/A"))
        context = f"""POLICY DOCUMENT {index}
Source: {source_label}
Content:
{doc.page_content}"""

    return context.strip()