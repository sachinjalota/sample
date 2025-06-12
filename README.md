from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.deps import validate_headers_and_api_key
from src.api.routers import collection_router
from src.main import app
from src.models.headers import HeaderInformation


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
    fake_session = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_session
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.routers.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    mocker.patch("src.api.routers.collection_router.check_embedding_model", return_value=128)

    mocker.patch("src.api.routers.collection_router.create_document_model", return_value=None)

    resp = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert "collections" in data
    assert len(data["collections"]) == 1
    assert data["collections"][0]["collection_name"] == "test-collection"
    assert "uuid" in data["collections"][0]


# ──────────────────────────────────────────────────────────────────────────────
def test_create_collection_model_not_found(client, create_collection_payload, valid_headers, mocker):
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = MagicMock()
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.routers.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    mocker.patch(
        "src.api.routers.collection_router.check_embedding_model",
        side_effect=HTTPException(status_code=400, detail="Model 'test-model' not found in embedding_models table."),
    )

    resp = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert resp.status_code == 400
    assert resp.json()["error"] == "Model 'test-model' not found in embedding_models table."


def test_delete_collection_success(client, delete_collection_payload, valid_headers, mocker):
    fake_session = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_session
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.routers.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    fake_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(uuid="test-collection-uuid")

    mocker.patch.object(
        collection_router,
        "validate_collection_access",
        return_value=None,
    )

    mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=True)
    mocker.patch("src.repository.document_repository.DocumentRepository.delete_collection", return_value=None)

    resp = client.request("DELETE", "/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message"] == "Collection has been deleted."
    assert body["collection"] == "test-collection-uuid"


def test_delete_collection_table_not_found(client, delete_collection_payload, valid_headers, mocker):
    fake_session = MagicMock()
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_session
    fake_cm.__exit__.return_value = None
    mocker.patch(
        "src.api.routers.collection_router.create_session_platform",
        return_value=fake_cm,
    )

    fake_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(uuid="test-collection-uuid")

    mocker.patch.object(
        collection_router,
        "validate_collection_access",
        return_value=None,
    )

    mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=False)

    resp = client.request("DELETE", "/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
    assert resp.status_code == 404
    assert "Collection table" in resp.json()["error"]
