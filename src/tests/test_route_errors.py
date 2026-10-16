"""Route errors keep their status code and name the exception type."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import lex_db.api.routes as routes


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    @contextmanager
    def fake_connection():
        yield MagicMock()

    monkeypatch.setattr(routes.db, "get_db_connection", fake_connection)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


SEARCH_ENDPOINTS = [
    ("/api/vector-search/indexes/missing/query", {"query_text": "fregat"}),
    (
        "/api/vector-search/indexes/missing/batch",
        {"queries": [["fregat", "query"]], "top_k": 5},
    ),
    ("/api/hybrid-search/indexes/missing/query", {"query_text": "fregat"}),
    ("/api/hyde-search/indexes/missing/query", {"query_text": "fregat"}),
    ("/api/text-search/indexes/missing/batch", {"queries": ["fregat"]}),
]


@pytest.mark.parametrize("url, body", SEARCH_ENDPOINTS)
def test_missing_index_is_404_not_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, url: str, body: dict
) -> None:
    monkeypatch.setattr(routes, "get_vector_index_metadata", lambda conn, name: None)

    response = client.post(url, json=body)

    assert response.status_code == 404
    assert response.json() == {"detail": "Vector index 'missing' not found"}


def test_invalid_article_ids_is_422_not_500(client: TestClient) -> None:
    response = client.get("/api/articles", params=[("ids", "1"), ("ids", "x")])

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid ids format"}


class PoolTimeout(Exception):
    """Stand-in for psycopg_pool.PoolTimeout."""


@pytest.mark.parametrize("url, body", SEARCH_ENDPOINTS)
def test_server_error_detail_names_exception_type(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    url: str,
    body: dict,
) -> None:
    def fail(conn, name):
        raise PoolTimeout("couldn't get a connection after 30.00 sec")

    monkeypatch.setattr(routes, "get_vector_index_metadata", fail)

    with caplog.at_level("ERROR"):
        response = client.post(url, json=body)

    assert response.status_code == 500
    assert response.json()["detail"].endswith(
        "error: PoolTimeout: couldn't get a connection after 30.00 sec"
    )
    # Logged with the traceback, not just the message.
    assert any(r.exc_info and r.exc_info[0] is PoolTimeout for r in caplog.records)


def test_server_error_with_empty_message_still_names_type(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(conn, name):
        raise TimeoutError()

    monkeypatch.setattr(routes, "get_vector_index_metadata", fail)

    response = client.post(SEARCH_ENDPOINTS[-1][0], json=SEARCH_ENDPOINTS[-1][1])

    assert response.status_code == 500
    assert response.json() == {"detail": "Batch fulltext search error: TimeoutError"}


def test_validation_error_is_still_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(conn, name):
        raise ValueError("Unknown embedding model: foo")

    monkeypatch.setattr(routes, "get_vector_index_metadata", fail)

    response = client.post(SEARCH_ENDPOINTS[0][0], json=SEARCH_ENDPOINTS[0][1])

    assert response.status_code == 400
    assert response.json() == {"detail": "Unknown embedding model: foo"}
