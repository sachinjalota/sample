# Test Script.py
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from src.api.deps import validate_headers_and_api_key
from src.api.routers import collection_router
from src.db.collection_db_manager import DBManager
from src.main import app
from src.models.headers import HeaderInformation
from src.repository.document_repository import DocumentRepository


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_headers():
    return {"x-session-id": "test-session", "x-base-api-key": "test-api-key"}


@pytest.fixture
def valid_header_info():
    return HeaderInformation(x_session_id="test-session", x_base_api_key="test-api-key")


@pytest.fixture(autouse=True)
def override_dependencies(valid_header_info):
    # force the API key header dependency to return our fake HeaderInformation
    app.dependency_overrides[validate_headers_and_api_key] = lambda: valid_header_info
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def create_collection_payload():
    return {
        "collection_entries": [
            {
                "channel_id": 1,
                "usecase_id": "test-usecase",
                "collection_name": "test-collection",
                "model": "test-model",
            }
        ]
    }


@pytest.fixture
def delete_collection_payload():
    return {"collection_uid": "test-collection-uuid"}


def test_create_collection_success(client, create_collection_payload, valid_headers, mocker):
    # Mock embedding‐model lookup to return 128 dimensions
    mocker.patch(
        "src.api.routers.collection_router.check_embedding_model",
        return_value=128,
    )
    # Mock DB insert to return a fake primary key
    fake_pk = ["fake_db_pk"]
    mocker.patch.object(DBManager, "insert_one", return_value=fake_pk)
    # Mock document‐table creation
    mocker.patch(
        "src.api.routers.collection_router.create_document_model",
        return_value=None,
    )

    resp = client.post(
        "/v1/api/create_collection",
        json=create_collection_payload,
        headers=valid_headers,
    )
    assert resp.status_code == status.HTTP_200_OK, resp.text

    body = resp.json()
    assert "collections" in body
    assert isinstance(body["collections"], list) and len(body["collections"]) == 1
    entry = body["collections"][0]
    assert entry["collection_name"] == "test-collection"
    assert "uuid" in entry


def test_create_collection_model_not_found(client, create_collection_payload, valid_headers, mocker):
    # Simulate missing embedding model
    mocker.patch(
        "src.api.routers.collection_router.check_embedding_model",
        side_effect=HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embedding model 'test-model' not found.",
        ),
    )

    resp = client.post(
        "/v1/api/create_collection",
        json=create_collection_payload,
        headers=valid_headers,
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()["detail"] == "Embedding model 'test-model' not found."


def test_delete_collection_success(client, delete_collection_payload, valid_headers, mocker):
    # Bypass external authorization
    mocker.patch.object(
        collection_router,
        "validate_collection_access",
        return_value=None,
    )
    # Simulate that the table exists and deletion works
    mocker.patch.object(DocumentRepository, "check_table_exists", return_value=True)
    mocker.patch.object(DocumentRepository, "delete_collection", return_value=None)
    # Mock database delete of metadata
    mocker.patch.object(DBManager, "delete", return_value=1)

    resp = client.delete(
        "/v1/api/delete_collection",
        json=delete_collection_payload,
        headers=valid_headers,
    )
    assert resp.status_code == status.HTTP_200_OK, resp.text

    body = resp.json()
    assert body["message"] == "Collection has been deleted."
    assert body["collection"] == delete_collection_payload["collection_uid"]


def test_delete_collection_table_not_found(client, delete_collection_payload, valid_headers, mocker):
    # Bypass external authorization
    mocker.patch.object(
        collection_router,
        "validate_collection_access",
        return_value=None,
    )
    # Simulate missing table
    mocker.patch.object(DocumentRepository, "check_table_exists", return_value=False)

    resp = client.delete(
        "/v1/api/delete_collection",
        json=delete_collection_payload,
        headers=valid_headers,
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    # FastAPI raises HTTPException(detail=...) → response.json()["detail"]
    assert "Collection table" in resp.json()["detail"]


def test_delete_collection_not_in_metadata(client, delete_collection_payload, valid_headers, mocker):
    # Bypass external authorization
    mocker.patch.object(
        collection_router,
        "validate_collection_access",
        return_value=None,
    )
    # Simulate table exists
    mocker.patch.object(DocumentRepository, "check_table_exists", return_value=True)
    mocker.patch.object(DocumentRepository, "delete_collection", return_value=None)
    # Simulate metadata delete returning zero rows
    mocker.patch.object(DBManager, "delete", return_value=0)

    resp = client.delete(
        "/v1/api/delete_collection",
        json=delete_collection_payload,
        headers=valid_headers,
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in resp.json()["detail"]
