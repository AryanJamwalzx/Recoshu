from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total number of HTTP requests.",
    ["method", "endpoint", "status"],
)


REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
)


RAG_QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total number of RAG queries.",
)


RAG_QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "Time spent processing a RAG query.",
)