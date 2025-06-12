import json
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.deps import validate_headers_and_api_key, validate_collection_access
from src.main import app
from src.models.headers import HeaderInformation


# Fixtures
@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_headers():
    return {"x-session-id": "test-session", "x-base-api-key": "test-api-key"}


@pytest.fixture
def valid_header_info():
    return HeaderInformation(x_session_id="test-session", x_base_api_key="test-api-key", team_id="test_team")


@pytest.fixture()
def valid_collection_access():
    return None


@pytest.fixture
def override_dependencies(valid_header_info, valid_collection_access):
    app.dependency_overrides[validate_headers_and_api_key] = lambda: valid_header_info
    app.dependency_overrides[validate_collection_access] = lambda: valid_collection_access
    yield
    app.dependency_overrides = {}


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


def test_create_collection_success(client, create_collection_payload, valid_headers, override_dependencies, mocker):
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(
        model_name="test-model", dimensions=128
    )

    mocker.patch("src.db.connection.create_session_platform", return_value=mock_session)

    mocker.patch("src.repository.document_repository.create_document_model", return_value=None)

    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)

    assert response.status_code == 200
    response_data = response.json()
    assert "collections" in response_data
    assert response_data["collections"][0]["collection_name"] == "test-collection"


def test_create_collection_model_not_found(
        client, create_collection_payload, valid_headers, override_dependencies, mocker
):
    mocker.patch(
        "src.api.deps.check_embedding_model",
        side_effect=HTTPException(status_code=400, detail="Model 'test-model' not found in embedding_models table."),
    )

    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert response.status_code == 400
    assert response.json()["error"] == "Model 'test-model' not found in embedding_models table."


def test_delete_collection_success(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
    mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=True)
    mocker.patch("src.repository.document_repository.DocumentRepository.delete_collection", return_value=1)

    response = client.request("DELETE", "/v1/api/delete_collection",
                              json=delete_collection_payload, headers=valid_headers
                              )
    assert response.status_code == 200
    assert response.json()["message"] == "Collection has been deleted."


def test_delete_collection_not_found(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
    mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=False)

    response = client.request("DELETE", "/v1/api/delete_collection",
                              json=delete_collection_payload, headers=valid_headers
                              )
    assert response.status_code == 404
    assert "Collection table" in response.json()["error"]
