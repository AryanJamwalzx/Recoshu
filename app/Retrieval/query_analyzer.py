import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


class QueryFilters(BaseModel):
    brand: Optional[str] = None
    sub_category: Optional[str] = None
    gender: Optional[str] = None
    availability: Optional[str] = None
    final_price_lte: Optional[float] = None
    final_price_gte: Optional[float] = None


class QueryAnalysis(BaseModel):
    semantic_query: str = Field(
        description="The core semantic search query with unnecessary filters removed."
    )
    filters: QueryFilters


QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query analyzer for a shoe recommendation and support system.
Extract structured product filters from the user's query.

Supported filters (use these exact values -- do not paraphrase or guess a
different spelling, since filtering is an exact match against stored data):
- brand: Nike or Adidas
- sub_category: Basketball, Football/Soccer, Lifestyle, Running,
  Skateboarding, or Training
- gender: Men, Women, Unisex, Boys, or Girls
- availability: In Stock, Low Stock, Out of Stock, or Pre-Order
- maximum final price (final_price_lte)
- minimum final price (final_price_gte)

Do NOT extract a color filter. Colors in this catalog are compound
descriptive names (e.g. "Royal Blue/White", "Bright Crimson"), not simple
words, so a color filter can never reliably match. If the user mentions a
color, leave it in semantic_query instead so vector search can use it.

Rules:
1. Extract only information explicitly stated or clearly implied.
2. Do not invent filters or values outside the exact lists above.
3. Remove structured constraints from semantic_query where possible.
4. Keep semantic_query useful for vector similarity search.
5. Return null when a filter is not present.
""",
        ),
        ("human", "User query: {query}"),
    ]
)

_analyzer: Runnable | None = None

# --- quota protection ---------------------------------------------------

MIN_CALL_INTERVAL = float(os.getenv("QUERY_ANALYZER_MIN_INTERVAL", "1.2"))
_last_call_time = 0.0

CACHE_PATH = Path(os.getenv("QUERY_ANALYZER_CACHE_PATH", ".cache/query_analysis_cache.json"))


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Query analysis cache unreadable (%s); starting fresh.", exc)
        return {}


def _save_to_cache(key: str, analysis: QueryAnalysis) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = _load_cache()
        cache[key] = analysis.model_dump()
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError as exc:
        # Never let a cache-write failure break the actual request.
        logger.warning("Could not persist query analysis cache: %s", exc)


def _throttle() -> None:
    global _last_call_time
    elapsed = time.monotonic() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.monotonic()


# --- analyzer -------------------------------------------------------------

def get_query_analyzer(model: str = "mistral-small-latest") -> Runnable:
    """
    Return a cached LangChain structured-output analyzer, backed by
    Mistral's small model, which is reliably included in the free tier
    (unlike mistral-large-latest, which returned 403 "not available in
    your subscription tier" when tested against a real free-tier account).
    The first call creates it; later calls reuse the same instance.
    """
    global _analyzer

    if _analyzer is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.mistral.ai/ -- "
                "check Admin Console -> Limits to see exactly which models "
                "your account's free tier includes, since this varies."
            )
        llm = ChatMistralAI(model=model, temperature=0)
        _analyzer = llm.with_structured_output(QueryAnalysis)

    return _analyzer


def analyze_query(query: str, use_cache: bool = True) -> QueryAnalysis:
    """
    Analyze a user query and extract semantic search text and structured
    metadata filters.

    On a rate-limited free tier, quota is precious during development, so
    this function:
      1. Checks a persistent on-disk cache first -- re-running the same
         test query costs zero API calls.
      2. Throttles proactively to roughly stay under the free-tier rate
         limit instead of waiting to get hit with a 429.
      3. Fails SOFT: any API error (429, network hiccup, etc.) logs a
         warning and returns a filter-less QueryAnalysis instead of
         raising, so one rate-limit collision doesn't kill the whole
         request/test run. Set use_cache=False to force a fresh call.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query = query.strip()
    key = _cache_key(query)

    if use_cache:
        cached = _load_cache().get(key)
        if cached is not None:
            logger.info("analyze_query(): cache hit for %r, no API call made.", query)
            return QueryAnalysis.model_validate(cached)

    analyzer = get_query_analyzer()
    prompt = QUERY_PROMPT.invoke({"query": query})

    _throttle()

    try:
        result = analyzer.invoke(prompt)
    except Exception as exc:
        logger.warning(
            "Query analysis failed for %r (falling back to plain semantic "
            "search, not raising -- likely free-tier rate limit): %s",
            query, exc,
        )
        return QueryAnalysis(semantic_query=query, filters=QueryFilters())

    if use_cache:
        _save_to_cache(key, result)

    return result


def build_pinecone_filter(filters: QueryFilters) -> Optional[dict]:
    """
    Convert extracted QueryFilters into a Pinecone metadata filter dict,
    ready to pass as retrieve(query, filter=build_pinecone_filter(...)).
    Returns None if no filters were set (pure semantic search).
    """
    conditions: dict = {}

    if filters.brand:
        conditions["brand"] = filters.brand
    if filters.sub_category:
        conditions["sub_category"] = filters.sub_category
    if filters.gender:
        conditions["gender"] = filters.gender
    if filters.availability:
        conditions["availability"] = filters.availability

    price_range = {}
    if filters.final_price_gte is not None:
        price_range["$gte"] = filters.final_price_gte
    if filters.final_price_lte is not None:
        price_range["$lte"] = filters.final_price_lte
    if price_range:
        conditions["final_price"] = price_range

    return conditions or None