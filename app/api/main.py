import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from app.Observability.Metric import (
    RAG_QUERY_COUNT,
    RAG_QUERY_LATENCY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from app.Pipeline.pipeline import DEFAULT_FINAL_K, run_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up pipeline dependencies...")

    try:
        from app.Embeddings.embeddings import get_embedding_model
        from app.Retrieval.query_analyzer import get_query_analyzer
        from app.Retrieval.reranker import get_reranker
        from app.Vector_Store.pinecone import get_vector_store

        get_embedding_model()
        get_reranker()
        get_query_analyzer()
        get_vector_store()

        logger.info("Warm-up complete.")

    except Exception:
        logger.exception(
            "Warm-up failed; continuing startup, "
            "will retry on first request."
        )

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="RAG Assistant API",
    version="1.0.0",
    description="API for the RAG Assistant",
    lifespan=lifespan,
)


# ---------------- Prometheus ----------------

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.monotonic()

    response = await call_next(request)

    elapsed = time.monotonic() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(elapsed)

    return response


# ---------------- Request / Response Models ----------------

class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="User's search question.",
    )

    final_k: int = Field(
        default=DEFAULT_FINAL_K,
        ge=1,
        le=20,
        description="Number of results to return.",
    )


class QueryResponse(BaseModel):
    query: str
    context: str
    has_results: bool


# ---------------- API Routes ----------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    start_time = time.monotonic()

    RAG_QUERY_COUNT.inc()

    try:
        context = run_pipeline(
            request.query,
            final_k=request.final_k,
        )

        has_results = True

    except ValueError as exc:
        message = str(exc)

        if "No relevant results" in message:
            logger.info(
                "No results for query: %r",
                request.query,
            )

            context = ""
            has_results = False

        else:
            logger.warning(
                "Invalid request for query %r: %s",
                request.query,
                message,
            )

            raise HTTPException(
                status_code=400,
                detail=message,
            )

    except RuntimeError as exc:
        logger.error(
            "Downstream dependency failure for query %r: %s",
            request.query,
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "A dependency is temporarily unavailable. "
                "Please try again."
            ),
        )

    except Exception:
        logger.exception(
            "Unexpected error in RAG pipeline for query: %r",
            request.query,
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )

    elapsed = time.monotonic() - start_time

    RAG_QUERY_LATENCY.observe(elapsed)

    logger.info(
        "Query %r completed in %.2fs (has_results=%s)",
        request.query,
        elapsed,
        has_results,
    )

    return QueryResponse(
        query=request.query,
        context=context,
        has_results=has_results,
    )


# ---------------- Frontend ----------------

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "UI"

app.mount(
    "/",
    StaticFiles(
        directory=str(FRONTEND_DIR),
        html=True,
    ),
    name="ui",
)