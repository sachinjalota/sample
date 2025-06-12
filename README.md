Basis the following code, update the Test script

>> collection_router.py
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import validate_headers_and_api_key
from src.config import get_settings
from src.db.collection_db_manager import DBManager
from src.db.platform_meta_tables import CollectionInfo
from src.logging_config import Logger
from src.models.collection_payload import CreateCollections, DeleteCollection
from src.models.headers import HeaderInformation
from src.repository.document_repository import DocumentRepository, create_document_model
from src.utility.collection_helpers import (
    check_embedding_model,
    validate_collection_access,
)

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()


@router.post(
    settings.create_collection,
    summary="Create new collection(s) based on request.",
    status_code=status.HTTP_200_OK,
)
async def create_collection(
    request: CreateCollections,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
) -> dict:
    try:
        response = []

        for entry in request.collection_entries:
            dims = await check_embedding_model(entry.model)
            collection_uuid = str(uuid4())

            DBManager.insert_one(
                db_tbl=CollectionInfo,
                data={
                    "uuid": collection_uuid,
                    "channel_id": entry.channel_id,
                    "usecase_id": entry.usecase_id,
                    "collection_name": entry.collection_name,
                    "model": entry.model,
                },
            )
            create_document_model(collection_uuid, embedding_dimensions=dims)

            logger.info(f"Created collection {collection_uuid} with dimensions {dims}")
            response.append(
                {
                    "collection_name": entry.collection_name,
                    "uuid": collection_uuid,
                }
            )

        return {"collections": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error creating collection.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    settings.delete_collection,
    summary="Delete a collection from DB and its metadata.",
    status_code=status.HTTP_200_OK,
)
async def delete_collection(
    request: DeleteCollection,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
) -> dict:
    try:
        await validate_collection_access(header_information.x_base_api_key, request.collection_uid)

        repo = DocumentRepository(request.collection_uid, create_if_not_exists=False)
        if not repo.check_table_exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection table '{request.collection_uid}' does not exist in the database.",
            )
        repo.delete_collection()

        deleted = DBManager.delete(db_tbl=CollectionInfo, filters={"uuid": request.collection_uid})
        if deleted == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{request.collection_uid}' not found."
            )

        logger.info(f"Deleted collection with UID '{request.collection_uid}'")
        return {"message": "Collection has been deleted.", "collection": request.collection_uid}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting collection.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


>> collection_db_manager.py
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

from fastapi import HTTPException, status
from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import DeclarativeMeta

from src.db.connection import create_session_platform
from src.logging_config import Logger

T = TypeVar("T", bound=DeclarativeMeta)
logger = Logger.create_logger(__name__)


class DBManager:
    @staticmethod
    def _handle_error(exc: Exception, msg: str) -> None:
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @staticmethod
    def _build_filters(model: Type[T], filters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None) -> Sequence[Any]:
        if not filters:
            return []
        if isinstance(filters, dict):
            return [getattr(model, k) == v for k, v in filters.items()]
        return list(filters)

    @classmethod
    def select_one(  # type: ignore
        cls,
        db_tbl: Type[T],
        filters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
    ) -> dict:
        try:
            with create_session_platform() as session:
                stmt = select(db_tbl).where(*cls._build_filters(db_tbl, filters))
                result = session.execute(stmt).scalar_one_or_none()

                if result is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"{db_tbl.__name__} not found for filters {filters}",
                    )
                return {key: value for key, value in vars(result).items() if key != "_sa_instance_state"}

        except HTTPException:
            raise
        except Exception as exc:
            cls._handle_error(exc, f"Error selecting one {db_tbl.__name__}")

    @classmethod
    def select_many(  # type: ignore
        cls,
        db_tbl: Type[T],
        filters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[T]:
        try:
            with create_session_platform() as session:
                stmt = select(db_tbl).where(*cls._build_filters(db_tbl, filters))
                if limit is not None:
                    stmt = stmt.limit(limit)
                if offset is not None:
                    stmt = stmt.offset(offset)
                return session.execute(stmt).scalars().all()  # type: ignore

        except Exception as exc:
            cls._handle_error(exc, f"Error selecting many {db_tbl.__name__}")

    @classmethod
    def insert_one(cls, db_tbl: Type[T], data: Dict[str, Any]) -> Any:
        try:
            with create_session_platform() as session:
                stmt = insert(db_tbl).values(**data)
                result = session.execute(stmt)
                session.commit()
                return result.inserted_primary_key
        except Exception as exc:
            cls._handle_error(exc, f"Error inserting into {db_tbl.__name__}")

    @classmethod
    def update_many(  # type: ignore
        cls, db_tbl: Type[T], filters: Optional[Union[Dict[str, Any], Sequence[Any]]], data: Dict[str, Any]
    ) -> int:
        try:
            with create_session_platform() as session:
                stmt = update(db_tbl).where(*cls._build_filters(db_tbl, filters)).values(**data)
                result = session.execute(stmt)
                session.commit()
                return result.rowcount
        except Exception as exc:
            cls._handle_error(exc, f"Error updating {db_tbl.__name__}")

    @classmethod
    def delete(cls, db_tbl: Type[T], filters: Optional[Union[Dict[str, Any], Sequence[Any]]]) -> int:  # type: ignore
        try:
            with create_session_platform() as session:
                stmt = delete(db_tbl).where(*cls._build_filters(db_tbl, filters))
                result = session.execute(stmt)
                session.commit()
                return result.rowcount
        except Exception as exc:
            cls._handle_error(exc, f"Error deleting from {db_tbl.__name__}")

>> collection_helpers.py
import httpx
from fastapi import HTTPException, status

from src.config import get_settings
from src.db.collection_db_manager import DBManager
from src.db.platform_meta_tables import CollectionInfo, EmbeddingModels

settings = get_settings()


async def check_embedding_model(model_name: str) -> int:
    row = DBManager.select_one(db_tbl=EmbeddingModels, filters={"model_name": model_name})
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Embedding model '{model_name}' not found.")
    return int(row["dimensions"])


async def validate_collection_access(api_key: str, collection_uuid: str) -> None:
    validation_url = f"{settings.prompt_hub_endpoint}{settings.prompt_hub_get_usecase_by_apikey}"
    verify = settings.deployment_env == "PROD"
    async with httpx.AsyncClient(verify=verify) as client:
        resp = await client.get(validation_url, headers={"lite-llm-api-key": api_key})
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")

    data = resp.json()["data"]
    channel_id = data["channel_id"]
    team_id = data["team_id"]

    col = DBManager.select_one(db_tbl=CollectionInfo, filters={"uuid": collection_uuid})
    if not col:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_uuid}' not found in DB.",
        )

    if str(col["channel_id"]) != channel_id or str(col["usecase_id"]) != team_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is not authorized for this collection.")


>> Test Script.py
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
