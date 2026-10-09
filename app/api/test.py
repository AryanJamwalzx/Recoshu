from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.main import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_success():
    with patch(
        "app.api.main.run_pipeline",
        return_value="PRODUCT 1\nNike Air Max...",
    ) as mock_pipeline:

        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"query": "Nike running shoes"},
            )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "Nike running shoes"
    assert body["context"] == "PRODUCT 1\nNike Air Max..."
    assert body["has_results"] is True

    mock_pipeline.assert_called_once()


def test_query_empty_string_rejected_by_validation():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": ""},
        )

    assert response.status_code == 422


def test_query_missing_field_rejected_by_validation():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={},
        )

    assert response.status_code == 422


def test_query_too_long_rejected_by_validation():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "x" * 501},
        )

    assert response.status_code == 422


def test_query_no_results_returns_200_not_error():
    with patch(
        "app.api.main.run_pipeline",
        side_effect=ValueError(
            "No relevant results found for query: 'xyz'"
        ),
    ):

        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"query": "some obscure query"},
            )

    assert response.status_code == 200

    body = response.json()

    assert body["has_results"] is False
    assert body["context"] == ""


def test_query_other_value_error_returns_400():
    with patch(
        "app.api.main.run_pipeline",
        side_effect=ValueError("Something else went wrong"),
    ):

        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"query": "some query"},
            )

    assert response.status_code == 400
    assert "Something else went wrong" in response.json()["detail"]


def test_query_runtime_error_returns_503():
    with patch(
        "app.api.main.run_pipeline",
        side_effect=RuntimeError("Mistral API rate limited"),
    ):

        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"query": "some query"},
            )

    assert response.status_code == 503

    assert "Mistral" not in response.json()["detail"]


def test_query_unexpected_exception_returns_500_and_does_not_leak_details():
    with patch(
        "app.api.main.run_pipeline",
        side_effect=KeyError("unexpected internal bug"),
    ):

        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"query": "some query"},
            )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."


def test_query_respects_custom_final_k():
    with patch(
        "app.api.main.run_pipeline",
        return_value="context",
    ) as mock_pipeline:

        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={
                    "query": "some query",
                    "final_k": 3,
                },
            )

    assert response.status_code == 200

    mock_pipeline.assert_called_once_with(
        "some query",
        final_k=3,
    )


def test_query_final_k_out_of_range_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "some query",
                "final_k": 100,
            },
        )

    assert response.status_code == 422


def test_lifespan_warms_up_dependencies():
    with patch(
        "app.Embeddings.embeddings.get_embedding_model"
    ) as mock_embed, \
         patch(
             "app.Retrieval.query_analyzer.get_query_analyzer"
         ) as mock_analyzer, \
         patch(
             "app.Retrieval.reranker.get_reranker"
         ) as mock_reranker, \
         patch(
             "app.Vector_Store.pinecone.get_vector_store"
         ) as mock_store:

        with TestClient(app):
            pass

    mock_embed.assert_called_once()
    mock_analyzer.assert_called_once()
    mock_reranker.assert_called_once()
    mock_store.assert_called_once()


def test_lifespan_warmup_failure_does_not_prevent_startup():
    with patch(
        "app.Embeddings.embeddings.get_embedding_model",
        side_effect=ConnectionError("no network"),
    ):

        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200