import pytest
from unittest.mock import MagicMock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.deps import validate_headers_and_api_key, validate_collection_access
from src.main import app
from src.models.headers import HeaderInformation


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: TestClient
@pytest.fixture
def client():
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: valid headers
@pytest.fixture
def valid_headers():
    return {
        "x-session-id": "test-session",
        "x-base-api-key": "test-api-key"
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: a HeaderInformation instance to return from validate_headers_and_api_key
@pytest.fixture
def valid_header_info():
    return HeaderInformation(
        x_session_id="test-session",
        x_base_api_key="test-api-key"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fixture: override the two deps so they never actually make HTTP calls
@pytest.fixture(autouse=True)
def override_dependencies(valid_header_info):
    # override validate_headers_and_api_key → returns our pre-made HeaderInformation
    app.dependency_overrides[validate_headers_and_api_key] = lambda: valid_header_info

    # override validate_collection_access → signature must match (base_api_key, collection_name)
    app.dependency_overrides[validate_collection_access] = lambda base_api_key, collection_uid: None

    yield

    app.dependency_overrides = {}


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: payloads
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


# ──────────────────────────────────────────────────────────────────────────────
# Test #1: happy‐path create_collection
def test_create_collection_success(client, create_collection_payload, valid_headers, mocker):
    # 1) Fake out the session factory to produce a context manager
    fake_session = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_session
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    # 2) Fake the embedding‐model lookup to return an integer dimension
    mocker.patch(
        "src.api.collection_router.check_embedding_model",
        return_value=128
    )

    # 3) Fake document‐model creation (no‐op)
    mocker.patch(
        "src.api.collection_router.create_document_model",
        return_value=None
    )

    # Exercise the endpoint
    resp = client.post(
        "/v1/api/create_collection",
        json=create_collection_payload,
        headers=valid_headers
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "collections" in data
    assert len(data["collections"]) == 1
    assert data["collections"][0]["collection_name"] == "test-collection"
    assert "uuid" in data["collections"][0]


# ──────────────────────────────────────────────────────────────────────────────
# Test #2: model lookup fails → 400
def test_create_collection_model_not_found(client, create_collection_payload, valid_headers, mocker):
    # patch session‐factory so the "with" doesn't blow up
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = MagicMock()
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    # now make check_embedding_model raise
    mocker.patch(
        "src.api.collection_router.check_embedding_model",
        side_effect=HTTPException(
            status_code=400,
            detail="Model 'test-model' not found in embedding_models table."
        )
    )

    resp = client.post(
        "/v1/api/create_collection",
        json=create_collection_payload,
        headers=valid_headers
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Model 'test-model' not found in embedding_models table."


# ──────────────────────────────────────────────────────────────────────────────
# Test #3: happy‐path delete_collection
def test_delete_collection_success(client, delete_collection_payload, valid_headers, mocker):
    # fake session‐factory
    fake_session = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_session
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    # patch out the SQLAlchemy select → pretend collection exists
    fake_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(uuid="test-collection-uuid")

    # patch DocumentRepository methods
    mocker.patch(
        "src.repository.document_repository.DocumentRepository.check_table_exists",
        return_value=True
    )
    mocker.patch(
        "src.repository.document_repository.DocumentRepository.delete_collection",
        return_value=None
    )

    resp = client.delete(
        "/v1/api/delete_collection",
        json=delete_collection_payload,
        headers=valid_headers
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message"] == "Collection has been deleted."
    assert body["collection"] == "test-collection-uuid"


# ──────────────────────────────────────────────────────────────────────────────
# Test #4: delete_collection when the table does not exist → 404
def test_delete_collection_table_not_found(client, delete_collection_payload, valid_headers, mocker):
    # fake session‐factory & pretend collection meta exists
    fake_session = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_session
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    # the meta‐query finds the collection
    fake_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(uuid="test-collection-uuid")

    # table‐exists returns False → 404
    mocker.patch(
        "src.repository.document_repository.DocumentRepository.check_table_exists",
        return_value=False
    )

    resp = client.delete(
        "/v1/api/delete_collection",
        json=delete_collection_payload,
        headers=valid_headers
    )
    assert resp.status_code == 404
    assert "Collection table" in resp.json()["detail"]
