>> src/api/routers/vector_store_router.py
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.deps import (
    get_embedding_service,
    get_vector_store_service,
    validate_headers_and_api_key,
)
from src.config import get_settings
from src.db.platform_meta_tables import VectorStoreInfo
from src.exception.document_store_exception import UnsupportedStorageBackendError
from src.exception.exceptions import DatabaseConnectionError
from src.logging_config import Logger
from src.models.headers import HeaderInformation
from src.models.vector_store_payload import (
    CreateVectorStoreRequest,
    CreateVectorStoreResponse,
    DeleteVectorStoreResponse,
    ListVectorStoresResponse,
    SearchVectorStoreRequest,
    SearchVectorStoreResponse,
    StorageBackend,
)
from src.repository.base_repository import BaseRepository
from src.repository.document_repository import DocumentRepository
from src.services.elasticsearch_vector_store import ElasticsearchVectorStore
from src.services.embedding_service import EmbeddingService
from src.services.pgvector_vector_store import PGVectorVectorStore
from src.services.service_layer.vector_store_service import VectorStoreService
from src.utility.vector_store_helpers import (
    get_store_model_info,
    get_valid_embedding_model,
    get_valid_usecase_id,
    validate_vector_store_name,
)

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()


@router.post(
    settings.vector_stores,
    response_model=CreateVectorStoreResponse,
    summary="Creates a Vector Store",
    status_code=status.HTTP_200_OK,
)
async def create_vector_store(
    request: CreateVectorStoreRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    valid_store_name: str = Depends(validate_vector_store_name),
    usecase_id: str = Depends(get_valid_usecase_id),
    embedding_model_info: dict = Depends(get_valid_embedding_model),
    vector_service: VectorStoreService = Depends(get_vector_store_service),
) -> CreateVectorStoreResponse:
    try:
        logger.info(f"Creating store '{valid_store_name}' from {header_information.x_session_id}")

        result = await vector_service.create_store(
            payload=request, usecase_id=usecase_id, embedding_dimensions=embedding_model_info["embedding_dimensions"]
        )

        return CreateVectorStoreResponse.model_validate(result)

    except UnsupportedStorageBackendError as exc:
        logger.warning(f"Unsupported storage backend: {request.storage_backend}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except DatabaseConnectionError:
        raise
    except Exception as exc:
        logger.exception(f"Unhandled exception in vector-store creation: {str(exc)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get(
    settings.vector_stores,
    response_model=ListVectorStoresResponse,
    summary="Lists all Vector Stores",
    status_code=status.HTTP_200_OK,
)
async def list_vector_stores(
    limit: int = 50,
    after: str | None = None,
    before: str | None = None,
    order: Literal["asc", "desc"] = Query("desc"),
    storage_backend: Literal["pgvector", "elasticsearch"] | None = Query(None),
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    usecase_id: str = Depends(get_valid_usecase_id),
    vector_service: VectorStoreService = Depends(get_vector_store_service),
) -> ListVectorStoresResponse:
    try:
        # Use pgvector backend as default for listing (it queries from VectorStoreInfo table)
        # This table contains all stores regardless of backend
        raw_stores = await vector_service.list_stores(
            usecase_id=usecase_id,
            limit=limit + 1,
            after=after,
            before=before,
            order=order,
            vector_db=storage_backend,  # Pass storage_backend for filtering (None = all)
        )

        stores = raw_stores[:limit]
        has_more = len(raw_stores) > limit
        first_id = stores[0].id if len(stores) >= 1 else None
        last_id = stores[-1].id if len(stores) >= 1 else None

        return ListVectorStoresResponse(
            object="list", data=stores, first_id=first_id, last_id=last_id, has_more=has_more
        )

    except ConnectionError as e:
        raise e
    except Exception as e:
        logger.exception("Error listing vector stores.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    f"{settings.vector_stores}/{{store_id}}/search",
    response_model=SearchVectorStoreResponse,
    summary="Searches a Vector Store (semantic / hybrid / full-text)",
    description=(
        "Executes a hybrid search combining semantic similarity and keyword-based full-text search over "
        "documents stored in the configured vector database. Accepts a query and optional filters (e.g., topic), "
        "and returns the most relevant documents based on embeddings and metadata fields. This endpoint supports "
        "filtering, ranking, and result explanation features depending on the backend implementation."
    ),
    status_code=status.HTTP_200_OK,
)
async def search_vector_store(
    store_id: str,
    request: SearchVectorStoreRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    store_model_info: dict = Depends(get_store_model_info),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> SearchVectorStoreResponse:
    try:
        model_name = store_model_info["model_name"]
        model_path = store_model_info["model_path"]
        embedding_dimensions = store_model_info["embedding_dimensions"]
        context_length = store_model_info["context_length"]

        logger.info(f"Search Request {request} from {header_information.x_session_id}")

        row_data = BaseRepository.select_one(  # type: ignore
            db_tbl=VectorStoreInfo, filters={"id": store_id, "vector_db": request.storage_backend}
        )
        if not row_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector store with Id '{store_id}' not found with '{request.storage_backend}'.",
            )

        store_name = row_data["name"]

        if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
            document_repository = DocumentRepository(f"{store_name}_chunks", embedding_dimensions=embedding_dimensions)
            if document_repository.check_table_exists():
                vector_store_service = PGVectorVectorStore(
                    embedding_service=embedding_service,
                    document_repository=document_repository,
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Vector Store '{store_name}' with Id '{store_id}' does not exist",
                )
        elif request.storage_backend.lower() == StorageBackend.ELASTICSEARCH.value:
            vector_store_service = ElasticsearchVectorStore(embedding_service=embedding_service)  # type: ignore
        else:
            raise UnsupportedStorageBackendError(f"Unsupported storage backend: {request.storage_backend}")

        return await vector_store_service.search_vector_store(request, store_id, model_name, context_length, model_path)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    f"{settings.vector_stores}/{{vector_store_id}}",
    summary="Deletes a Vector Store along with it's metadata and content",
    status_code=status.HTTP_200_OK,
)
async def delete_vector_store(
    vector_store_id: str,
    storage_backend: Literal["pgvector", "elasticsearch"] = Query(...),
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    usecase_id: str = Depends(get_valid_usecase_id),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> DeleteVectorStoreResponse:
    """
    Deletes a vector store from the chosen backend (pgvector or elasticsearch).
    Uses the new VectorStoreFactory + VectorStoreService architecture.
    """
    try:
        service = VectorStoreService(backend_name=storage_backend, embedding_service=embedding_service)
        result = await service.delete_store(store_id=vector_store_id, usecase_id=usecase_id)
        return DeleteVectorStoreResponse.model_validate(result)

    except UnsupportedStorageBackendError as exc:
        logger.warning(f"Unsupported storage backend: {storage_backend}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except DatabaseConnectionError as exc:
        logger.error("Database connection failed during vector store deletion.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    except Exception as e:
        logger.exception(f"Error deleting vector store {vector_store_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

>> src/api/routers/vector_store_files_router.py
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.deps import (
    get_embedding_service,
    get_vector_store_service,
    validate_headers_and_api_key,
)
from src.config import get_settings
from src.exception.document_store_exception import UnsupportedStorageBackendError
from src.exception.exceptions import DatabaseConnectionError
from src.logging_config import Logger
from src.models.headers import HeaderInformation
from src.models.vector_store_payload import (
    CreateVectorStoreFileRequest,
    CreateVectorStoreFileResponse,
    DeleteVectorStoreFileResponse,
    RetrieveFileResponse,
    StorageBackend,
)
from src.repository.document_repository import DocumentRepository
from src.services.elasticsearch_vector_store import ElasticsearchVectorStore
from src.services.embedding_service import EmbeddingService
from src.services.pgvector_vector_store import PGVectorVectorStore
from src.services.service_layer.vector_store_service import VectorStoreService
from src.utility.vector_store_helpers import get_store_model_info, get_valid_usecase_id

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()


@router.post(
    f"{settings.vector_stores}/{{store_id}}/files",
    response_model=CreateVectorStoreFileResponse,
    summary="Add file to Vector Store",
    status_code=status.HTTP_200_OK,
)
async def create_vector_store_file(
    store_id: str,
    request: CreateVectorStoreFileRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    usecase_id: str = Depends(get_valid_usecase_id),
    store_model_info: dict = Depends(get_store_model_info),
    vector_service: VectorStoreService = Depends(get_vector_store_service),
) -> CreateVectorStoreFileResponse:
    try:
        model_name = store_model_info["model_name"]
        model_path = store_model_info["model_path"]
        embedding_dimensions = store_model_info["embedding_dimensions"]
        context_length = store_model_info["context_length"]

        logger.info(f"Creating file in store '{store_id}' from {header_information.x_session_id}")
        logger.info(
            f"Indexing request for store '{store_id}' with {len(request.file_contents)} "
            f"documents using model '{model_name}'"
        )

        result = await vector_service.create_store_file(
            payload=request,
            store_id=store_id,
            usecase_id=usecase_id,
            model_name=model_name,
            context_length=context_length,
            model_path=model_path,
            embedding_dimensions=embedding_dimensions,
        )

        return CreateVectorStoreFileResponse.model_validate(result)

    except UnsupportedStorageBackendError as exc:
        logger.warning(f"Unsupported storage backend: {request.storage_backend}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Unhandled exception in file creation: {str(exc)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get(
    f"{settings.vector_stores}/{{vector_store_id}}/files/{{file_id}}/content",
    summary="Retrieves a vector store file (a single document) and its metadata",
    status_code=status.HTTP_200_OK,
)
async def retrieve_vector_store_file(
    vector_store_id: str,
    file_id: str,
    storage_backend: Literal["pgvector", "elasticsearch"] = Query(...),
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    usecase_id: str = Depends(get_valid_usecase_id),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RetrieveFileResponse:
    try:
        document_repository = DocumentRepository(vector_store_id, embedding_dimensions=0)

        if storage_backend.lower() == StorageBackend.PGVECTOR.value:
            pgvector_vector_file_storage = PGVectorVectorStore(
                embedding_service=embedding_service,
                document_repository=document_repository,
            )
            return await pgvector_vector_file_storage.retrieve_by_id(vector_store_id, file_id, usecase_id)
        elif storage_backend.lower() == StorageBackend.ELASTICSEARCH.value:
            vector_store_service = ElasticsearchVectorStore(embedding_service=embedding_service)
            return await vector_store_service.retrieve_by_id(vector_store_id, file_id, usecase_id)
        else:
            raise UnsupportedStorageBackendError(f"Unsupported storage backend: {storage_backend}")
    except HTTPException as e:
        raise e
    except ConnectionError:
        raise
    except DatabaseConnectionError:
        raise
    except Exception as e:
        logger.exception("Error retrieving vector store file.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    f"{settings.vector_stores}/{{vector_store_id}}/files/{{file_id}}",
    summary="Deletes a file (document) from the selected vector store backend (PGVector or Elasticsearch+GCP)",
    response_model=DeleteVectorStoreFileResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_vector_store_file(
    vector_store_id: str,
    file_id: str,
    storage_backend: Literal["pgvector", "elasticsearch"] = Query(...),
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    usecase_id: str = Depends(get_valid_usecase_id),
    embedding_service=Depends(get_embedding_service),
) -> DeleteVectorStoreFileResponse:
    """
    Deletes a vector store file (metadata, chunks, and GCP object)
    from the chosen backend using unified VectorStoreService.
    """
    try:
        service = VectorStoreService(
            backend_name=storage_backend,
            embedding_service=embedding_service,
        )
        result = await service.delete_file(vector_store_id, file_id, usecase_id)
        return DeleteVectorStoreFileResponse.model_validate(result)
    except UnsupportedStorageBackendError as exc:
        logger.warning(f"Unsupported backend: {storage_backend}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except DatabaseConnectionError as exc:
        logger.error("Database connection failed during file deletion.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    except Exception as e:
        logger.exception(f"Error deleting file {file_id} from vector store {vector_store_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

>> src/services/service_layer/vector_store_service.py
from typing import Any, Optional

from src.models.vector_store_payload import (
    CreateVectorStoreFileRequest,
    CreateVectorStoreRequest,
    SearchVectorStoreRequest,
)
from src.services.embedding_service import EmbeddingService
from src.services.factory.vector_store_factory import (  # type: ignore[attr-defined]
    VectorStoreConfig,
    VectorStoreFactory,
)


class VectorStoreService:
    """
    High-level service that delegates vector store operations
    to the appropriate backend via VectorStoreFactory.
    """

    def __init__(self, backend_name: str, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        config = VectorStoreConfig(backend=backend_name)
        self.vectorstore = VectorStoreFactory.create(
            name=backend_name, config=config, embedding_service=embedding_service
        )

    # ---------------------------------------------------------------------
    # CRUD + Search operations (delegated to strategy)
    # ---------------------------------------------------------------------

    async def create_store(
        self,
        payload: CreateVectorStoreRequest,
        usecase_id: str,
        embedding_dimensions: int,
    ) -> Any:
        """Creates a new vector store (database or index)."""
        return await self.vectorstore.create_store(payload, usecase_id, embedding_dimensions=embedding_dimensions)

    async def create_store_file(
        self,
        payload: CreateVectorStoreFileRequest,
        store_id: str,
        usecase_id: str,
        model_name: str,
        context_length: int,
        model_path: str,
        embedding_dimensions: int,
    ) -> Any:
        """Indexes a file/document into the vector store."""
        return await self.vectorstore.create_store_file(
            payload,
            store_id,
            usecase_id,
            model_name,
            context_length,
            model_path,
            embedding_dimensions,
        )

    async def search(
        self,
        payload: SearchVectorStoreRequest,
        store_id: str,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> Any:
        """Performs semantic/hybrid/full-text search."""
        return await self.vectorstore.search_vector_store(
            payload,
            store_id,
            model_name,
            context_length,
            model_path,
        )

    async def delete_store(self, store_id: str, usecase_id: str) -> Any:
        """Deletes the entire vector store (DB or index)."""
        return await self.vectorstore.delete(store_id, usecase_id)

    async def delete_file(self, store_id: str, file_id: str, usecase_id: str) -> Any:
        """Deletes a specific file/document from a vector store."""
        return await self.vectorstore.delete_by_id(store_id, file_id, usecase_id)

    async def retrieve_file(self, store_id: str, file_id: str, usecase_id: str) -> Any:
        """Retrieves a specific file/document and its metadata."""
        return await self.vectorstore.retrieve_by_id(store_id, file_id, usecase_id)

    async def list_stores(
        self,
        usecase_id: str,
        limit: int = 50,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: str = "desc",
        vector_db: Optional[str] = None,
    ) -> Any:
        """
        Lists available vector stores for a use case.
        If vector_db is None, lists all stores regardless of backend.
        If vector_db is specified, filters by that backend.
        """
        return await self.vectorstore.list_stores(usecase_id, limit, after, before, order, vector_db)

>> src/services/base_class/vector_store_base.py
import asyncio
import functools as _functools
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.config import get_settings
from src.db.platform_meta_tables import VectorStoreInfo
from src.exception.document_store_exception import (
    DocumentMaxTokenLimitExceededError,
    DocumentStoreSearchError,
)
from src.exception.exceptions import VectorStoreError
from src.logging_config import Logger
from src.models.vector_store_payload import (
    AttributesItem,
    ContentItem,
    CreateVectorStoreFileRequest,
    CreateVectorStoreFileResponse,
    CreateVectorStoreRequest,
    CreateVectorStoreResponse,
    DeleteVectorStoreFileResponse,
    DeleteVectorStoreResponse,
    FileCountsModel,
    FileStatus,
    RetrieveFileResponse,
    SearchResult,
    SearchType,
    SearchVectorStoreRequest,
    SearchVectorStoreResponse,
    VectorStoreErrorDetails,
    VectorStoreStatus,
)
from src.repository.base_repository import BaseRepository
from src.services.tokenizer_service import TokenizerService
from src.utility.vector_store_utils import get_deepsize, payload_to_internal_format

logger = Logger.create_logger(__name__)

# ---------------------------------------------------------------------
# Async wrapper helper
# ---------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])


def ensure_async(func: F) -> Callable[..., Awaitable[Any]]:
    @_functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return await run_in_threadpool(func, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------
# Config Model
# ---------------------------------------------------------------------
class VectorStoreConfig(BaseModel):
    backend: str
    embedding_model: Optional[str] = None
    context_length: Optional[int] = 4096
    embedding_dimensions: Optional[int] = 1536
    extra: Optional[dict] = None


# ---------------------------------------------------------------------
# BaseVectorStore (Template Method pattern)
# ---------------------------------------------------------------------
class BaseVectorStore(ABC):
    """
    Abstract base for all vector store backends (PGVector, Elasticsearch, etc.)
    Implements all CRUD orchestration, metadata handling, and error recovery.
    """

    def __init__(self, document_repository=None, embedding_service=None, settings=None) -> None:
        self.document_repository = document_repository
        self.embedding_service = embedding_service
        self.settings = settings or get_settings()
        self.tokenizer_service = TokenizerService()

    # ================================================================
    # CRUD OPERATIONS
    # ================================================================

    @ensure_async
    async def create_store(
        self, payload: CreateVectorStoreRequest, usecase_id: str, embedding_dimensions: int
    ) -> CreateVectorStoreResponse:
        try:
            # Generate common metadata
            store_id, now_dt, expires_at = self._generate_store_metadata(payload)

            # Validate uniqueness
            await self._validate_store_uniqueness(payload.name, usecase_id)

            # Delegate to backend-specific implementation
            result = await self._create_backend_store(
                payload=payload,
                usecase_id=usecase_id,
                embedding_dimensions=embedding_dimensions,
                store_id=store_id,
                now_dt=now_dt,
                expires_at=expires_at,
            )

            return CreateVectorStoreResponse.model_validate(result)

        except VectorStoreError as e:
            return self._build_store_response(
                store_id="",
                name=getattr(payload, "name", ""),
                created_at=0,
                status=VectorStoreStatus.FAILED.value,
                last_active_at=0,
                vs_metadata=getattr(payload, "metadata", None),
                error_details=VectorStoreErrorDetails(code="server_error", message=str(e)),
            )
        except Exception as e:
            logger.exception(f"Create store failed: {e}")
            return self._build_store_response(
                store_id="",
                name=getattr(payload, "name", ""),
                created_at=0,
                status=VectorStoreStatus.FAILED.value,
                last_active_at=0,
                vs_metadata=getattr(payload, "metadata", None),
                error_details=VectorStoreErrorDetails(code="server_error", message=str(e)),
            )

    @ensure_async
    async def create_store_file(
        self,
        payload: CreateVectorStoreFileRequest,
        store_id: str,
        usecase_id: str,
        model_name: str,
        context_length: int,
        model_path: str,
        embedding_dimensions: int,
    ) -> CreateVectorStoreFileResponse:
        try:
            # Fetch and validate store metadata
            store_record = await self._fetch_store_metadata(store_id, payload.storage_backend, usecase_id)
            store_name = store_record["name"]

            # Delegate to backend-specific indexing
            data_size = await self._index_backend(
                payload, store_id, store_name, model_name, context_length, model_path, embedding_dimensions
            )

            # Update store stats
            await self._update_store_stats_after_file_add(store_record, data_size)

            return self._build_storefile_response(payload, data_size, store_id, FileStatus.COMPLETED.value)

        except DocumentMaxTokenLimitExceededError as e:
            logger.warning(f"Token limit exceeded: {e}")
            return self._build_storefile_response(
                payload,
                0,
                store_id,
                FileStatus.FAILED.value,
                VectorStoreErrorDetails(code="max_token_limit", message=str(e)),
            )
        except Exception as e:
            logger.exception(f"Indexing failed: {e}")
            return self._build_storefile_response(
                payload,
                0,
                store_id,
                FileStatus.FAILED.value,
                VectorStoreErrorDetails(code="server_error", message=str(e)),
            )

    @ensure_async
    async def retrieve_by_id(self, vectorstoreid: str, vectorstorefileid: str, usecase_id: str) -> RetrieveFileResponse:
        try:
            file_info = await self._fetch_file_backend(vectorstoreid, vectorstorefileid, usecase_id)
            return self._build_retrieve_file_response(file_info)
        except Exception as e:
            logger.exception(f"Retrieve failed: {e}")
            return self._build_retrieve_file_response({}, exception=True)

    @ensure_async
    async def list_stores(
        self,
        usecase_id: str,
        limit: int = 50,
        after: Optional[str] = None,
        before: Optional[str] = None,
        order: str = "desc",
        vector_db: Optional[str] = None,
    ) -> List[CreateVectorStoreResponse]:
        raw_stores = await self._list_backend_stores(usecase_id, limit + 1, after, before, order, vector_db)

        # Build paginated response
        stores = await self._build_list_response(raw_stores[:limit], limit, after, before)

        return stores

    # ================================================================
    # DELETE STORE — Template Method
    # ================================================================
    @ensure_async
    async def delete(self, vector_id: str, usecase_id: str) -> DeleteVectorStoreResponse:
        try:
            record = await self._fetch_metadata(vector_id, usecase_id)
            await self._validate_backend_type(record)  # abstract, backend-specific

            backup_record = dict(record)
            await self._delete_metadata(vector_id)

            try:
                await self._drop_backend_tables(record["name"])
            except Exception as ddl_err:
                logger.error(f"DDL failed for {record['name']}: {ddl_err}")
                self._restore_metadata_on_failure(backup_record)
                raise VectorStoreError(f"DDL failed for '{record['name']}', metadata restored.")

            return self._build_delete_response(vector_id, is_file=False, deleted=True)

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return self._build_delete_response(vector_id, is_file=False, deleted=False)

    @ensure_async
    async def delete_by_id(
        self, vectorstoreid: str, vectorstorefileid: str, usecase_id: str
    ) -> DeleteVectorStoreFileResponse:
        try:
            # Step 1: Fetch metadata (shared)
            record = await self._fetch_metadata(vectorstoreid, usecase_id)
            await self._validate_backend_type(record)
            store_name = record["name"]

            # Step 2: Delete metadata and chunks (backend-specific)
            file_info_record, chunk_records = await self._delete_metadata_and_chunks(
                store_name, vectorstoreid, vectorstorefileid
            )

            # Step 4: Update store-level metadata stats
            deleted_file_size = file_info_record.get("usage_bytes", 0)
            update_success = await self._update_vectorstore_stats_after_delete(vectorstoreid, deleted_file_size)
            if not update_success:
                logger.warning(f":: STATS :: Failed to update store stats after delete for '{vectorstoreid}'")

            return self._build_delete_response(vectorstorefileid, is_file=True, deleted=True)

        except Exception as e:
            logger.exception(f" Delete file failed: {e}")
            return self._build_delete_response(vectorstorefileid, is_file=True, deleted=False)

    async def search_vector_store(
        self,
        payload: SearchVectorStoreRequest,
        store_id: str,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> SearchVectorStoreResponse:
        col = BaseRepository.select_one(db_tbl=VectorStoreInfo, filters={"id": store_id})  # type: ignore
        if not col:
            raise HTTPException(status_code=404, detail=f"Vector store '{store_id}' not found")

        store_name = col["name"]
        chunks_index = f"{store_name}_chunks"

        internal_req = payload_to_internal_format(api_payload=payload, collection=chunks_index)

        logger.info(f"Executing {internal_req.search_type} search on {store_id}")

        return await self._execute_search(
            internal_req,
            chunks_index,
            model_name,
            context_length,
            model_path,
        )

    async def _execute_search(
        self,
        search_request: Any,
        index_name: str,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> SearchVectorStoreResponse:
        try:
            search_results: List[SearchResult] = []

            # Validate query length for semantic searches
            if search_request.search_type in (SearchType.SEMANTIC, SearchType.HYBRID):
                if getattr(search_request, "search_text", None):
                    self._content_length_validation(
                        context_length,
                        model_name,
                        model_path,
                        search_request.search_text,
                        "search",
                    )

            if search_request.search_type == SearchType.SEMANTIC:
                search_results = await self._semantic_search(search_request, index_name, model_name)
            elif search_request.search_type == SearchType.FULL_TEXT:
                search_results = await self._fulltext_search(search_request, index_name)
            elif search_request.search_type == SearchType.HYBRID:
                search_results = await self._hybrid_search(search_request, index_name, model_name)

            logger.info(f"Search returned {len(search_results)} results")

            return SearchVectorStoreResponse(
                search_query=search_request.search_text,
                data=search_results,
            )

        except DocumentMaxTokenLimitExceededError:
            raise
        except Exception as e:
            logger.exception(f"Search execution failed: {str(e)}")
            raise DocumentStoreSearchError(f"Search operation failed: {str(e)}")

    # ================================================================
    # SHARED HELPERS
    # ================================================================

    # =========== CREATE_VECTOR_STORE - HELPER FUNCTIONS =============
    def _generate_store_metadata(self, payload: CreateVectorStoreRequest) -> Tuple[str, datetime, Optional[datetime]]:
        store_id = str(uuid4())
        now_dt = datetime.now(ZoneInfo(self.settings.timezone))
        expires_at = None

        if payload.expires_after and payload.expires_after.days:
            expires_at = now_dt + timedelta(days=payload.expires_after.days)

        return store_id, now_dt, expires_at

    async def _validate_store_uniqueness(self, store_name: str, usecase_id: str) -> None:
        existing = BaseRepository.select_one(
            db_tbl=VectorStoreInfo, filters={"name": store_name, "usecase_id": usecase_id}
        )

        if existing:
            raise VectorStoreError(
                f"Vector Store '{store_name}' already exists "
                f"(DB usecase: {existing['usecase_id']}, Request: {usecase_id})"
            )

    def _build_metadata_dict(
        self,
        store_id: str,
        payload: CreateVectorStoreRequest,
        usecase_id: str,
        now_dt: datetime,
        expires_at: Optional[datetime],
        backend_type: str,
    ) -> dict:
        metadata = {
            "id": store_id,
            "name": payload.name,
            "usecase_id": usecase_id,
            "model_name": payload.embedding_model,
            "created_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "last_active_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata_vs": payload.metadata or {},
            "expires_after": payload.expires_after.dict() if payload.expires_after else None,
            "file_counts": FileCountsModel().model_dump(),
            "vector_db": backend_type,
        }

        if expires_at:
            metadata["expires_at"] = expires_at.strftime("%Y-%m-%d %H:%M:%S")

        return metadata

    def _build_create_response_dict(
        self,
        store_id: str,
        payload: CreateVectorStoreRequest,
        now_dt: datetime,
        expires_at: Optional[datetime],
    ) -> dict:
        return {
            "id": store_id,
            "object": "vector_store",
            "created_at": int(now_dt.timestamp()),
            "name": payload.name,
            "usage_bytes": 0,
            "file_counts": FileCountsModel().model_dump(),
            "status": VectorStoreStatus.COMPLETED.value,
            "expires_after": payload.expires_after.dict() if payload.expires_after else None,
            "expires_at": int(expires_at.timestamp()) if expires_at else None,
            "last_active_at": int(now_dt.timestamp()),
            "metadata": payload.metadata,
            "last_error": None,
        }

    # ========= CREATE_VECTOR_STORE_FILE - HELPER FUNCTIONS ==========
    async def _fetch_store_metadata(self, store_id: str, storage_backend: str, usecase_id: str) -> dict:
        store_record = BaseRepository.select_one(
            db_tbl=VectorStoreInfo, filters={"id": store_id, "vector_db": storage_backend}
        )

        if not store_record:
            raise VectorStoreError(
                f"Either Vector store with Id '{store_id}' does not exist " f"or not found with '{storage_backend}'."
            )

        if store_record["usecase_id"] != usecase_id:
            raise VectorStoreError(f"Access denied for Vector Store '{store_id}'")

        return store_record

    async def prepare_index_emdeddings(
        self,
        payload: CreateVectorStoreFileRequest,
        model_name: str,
        model_path: str,
        context_length: int,
        embedding_service,
        timezone: str,
    ):
        content_list: List[str] = []
        for document in payload.file_contents:
            self._content_length_validation(context_length, model_name, model_path, document.content, "index")
            content_list.append(document.content)

        embeddings = await embedding_service.get_embeddings(model_name=model_name, batch=content_list)

        docs_with_embeddings = []
        for i, doc in enumerate(payload.file_contents):
            docs_with_embeddings.append(
                {
                    "content": doc.content,
                    "embedding": embeddings.data[i].embedding,
                    "links": doc.links,
                    "meta_data": json.dumps(doc.metadata, ensure_ascii=False) if doc.metadata else None,
                    "topics": doc.topics,
                    "author": doc.author,
                }
            )

        data_size = get_deepsize(docs_with_embeddings)

        now_dt = datetime.now(ZoneInfo(timezone))

        common_file_info = {
            "file_id": payload.file_id,
            "file_name": payload.file_name,
            "file_version": 1,
            "created_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "usage_bytes": data_size,
            "chunking_strategy": json.dumps(payload.chunking_strategy.dict())
            if getattr(payload, "chunking_strategy", None)
            else None,
            "attributes": payload.attributes or {},
            "status": FileStatus.COMPLETED.value,
        }

        return docs_with_embeddings, data_size, common_file_info

    async def _update_store_stats_after_file_add(self, store_record: dict, file_size: int) -> bool:
        try:
            # Update file counts
            current_file_counts = store_record.get("file_counts", {}) or FileCountsModel().model_dump()
            updated_file_counts = current_file_counts.copy()
            updated_file_counts["completed"] = updated_file_counts.get("completed", 0) + 1
            updated_file_counts["total"] = updated_file_counts.get("total", 0) + 1

            # Update usage and timestamp
            now_dt = datetime.now(ZoneInfo(self.settings.timezone))
            new_usage = (store_record.get("usage_bytes", 0) or 0) + file_size

            update_data = {
                "last_active_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "usage_bytes": new_usage,
                "file_counts": updated_file_counts,
            }

            BaseRepository.update_many(
                db_tbl=VectorStoreInfo,
                filters={"id": store_record["id"]},
                data=update_data,
            )

            logger.info(f"Updated store stats for '{store_record['id']}': +{file_size} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to update store stats for '{store_record['id']}': {e}", exc_info=True)
            return False

    # ================= LIST_STORES - HELPER METHODS =================
    async def _build_list_response(
        self, stores: List[dict], limit: int, after: Optional[str], before: Optional[str]
    ) -> List[CreateVectorStoreResponse]:
        response_stores = []

        for item in stores:
            created_at = int(item["created_at"].timestamp()) if item.get("created_at") else 0
            last_active_at = int(item["last_active_at"].timestamp()) if item.get("last_active_at") else 0

            response_stores.append(
                CreateVectorStoreResponse(
                    id=str(item["id"]),
                    created_at=created_at,
                    name=item["name"],
                    usage_bytes=item.get("usage_bytes", 0),
                    file_counts=item.get("file_counts", FileCountsModel()).copy()
                    if item.get("file_counts")
                    else FileCountsModel(),
                    status=VectorStoreStatus.COMPLETED.value,
                    expires_after=item.get("expires_after"),
                    last_active_at=last_active_at,
                    metadata=item.get("metadata_vs"),
                )
            )

        # Handle pagination
        start_index = 0
        end_index = len(response_stores)

        if after:
            after_indices = [i for i, store in enumerate(response_stores) if store.id == after]
            if after_indices:
                start_index = after_indices[0] + 1

        if before:
            before_indices = [i for i, store in enumerate(response_stores) if store.id == before]
            if before_indices:
                end_index = before_indices[0]
                start_index = max(0, end_index - limit)
                return response_stores[start_index:end_index]

        return response_stores[start_index : start_index + limit]

    async def _fetch_metadata(self, vector_id, usecase_id):
        from src.db.platform_meta_tables import VectorStoreInfo
        from src.repository.base_repository import BaseRepository

        col = BaseRepository.select_one(db_tbl=VectorStoreInfo, filters={"id": vector_id})
        if not col:
            raise VectorStoreError(f"Vector Store '{vector_id}' does not exist")
        if col["usecase_id"] != usecase_id:
            raise VectorStoreError(f"Access denied for Vector Store '{vector_id}'")
        return col

    async def _delete_metadata(self, vector_id):
        from src.db.platform_meta_tables import VectorStoreInfo
        from src.repository.base_repository import BaseRepository

        BaseRepository.delete(db_tbl=VectorStoreInfo, filters={"id": vector_id})
        logger.info(f"Deleted metadata for vector store '{vector_id}'")

    def _restore_metadata_on_failure(self, backup_record: dict):
        from src.db.platform_meta_tables import VectorStoreInfo
        from src.repository.base_repository import BaseRepository

        BaseRepository.insert_one(db_tbl=VectorStoreInfo, data=backup_record)
        logger.info(f"Restored metadata for failed DDL rollback: {backup_record['name']}")

    async def _update_vectorstore_stats_after_delete(self, vectorstoreid: str, deleted_file_size: int) -> bool:
        try:
            current_vs = BaseRepository.select_one(db_tbl=VectorStoreInfo, filters={"id": vectorstoreid})
            if not current_vs:
                logger.warning(f":: STATS :: No vector store found for id '{vectorstoreid}'")
                return False

            # Adjust file counts safely
            current_file_counts = current_vs.get("file_counts", {}) or FileCountsModel().model_dump()
            updated_file_counts = current_file_counts.copy()
            updated_file_counts["completed"] = max(updated_file_counts.get("completed", 0) - 1, 0)
            updated_file_counts["total"] = max(updated_file_counts.get("total", 0) - 1, 0)

            # Compute new usage and timestamps
            now_dt = datetime.now(ZoneInfo(self.settings.timezone))

            new_usage = max(current_vs.get("usage_bytes", 0) - deleted_file_size, 0)

            update_data = {
                "last_active_at": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "usage_bytes": new_usage,
                "file_counts": updated_file_counts,
            }

            BaseRepository.update_many(
                db_tbl=VectorStoreInfo,
                filters={"id": vectorstoreid},
                data=update_data,
            )

            logger.info(f":: STATS :: Updated metadata for vectorstore '{vectorstoreid}': {update_data}")
            return True

        except Exception as e:
            logger.error(f":: STATS :: Failed to update stats for '{vectorstoreid}': {e}", exc_info=True)
            return False

    def _content_length_validation(
        self, context_length: int, model_name: str, model_path: str, text: str, validation_for: str
    ) -> None:
        index_or_search = "content" if validation_for == "index" else "search text"
        token_count = self.tokenizer_service.get_token_count(model_name, text, model_path)

        if token_count > context_length:
            raise DocumentMaxTokenLimitExceededError(
                f"Document exceeds maximum {index_or_search} length: "
                f"max {context_length} tokens, received {token_count} tokens."
            )

    # ================================================================
    # ABSTRACT BACKEND METHODS
    # ================================================================
    @abstractmethod
    async def _create_backend_store(
        self,
        payload: CreateVectorStoreRequest,
        usecase_id: str,
        embedding_dimensions: int,
        store_id: str,
        now_dt: datetime,
        expires_at: Optional[datetime],
    ) -> dict: ...

    @abstractmethod
    async def _validate_backend_type(self, record: dict):
        """Ensures backend type matches the store type (e.g., pgvector / elasticsearch)."""
        pass

    # delete
    @abstractmethod
    async def _drop_backend_tables(self, store_name: str): ...

    # delete
    @abstractmethod
    async def _delete_metadata_and_chunks(self, store_name: str, vectorstoreid: str, vectorstorefileid: str):
        """Delete metadata and chunks from backend."""
        pass

    @abstractmethod
    async def _index_backend(
        self, payload, store_id, store_name, model_name, context_length, model_path, embedding_dimensions
    ):
        """Index documents to backend."""
        ...

    @abstractmethod
    async def _fetch_file_backend(self, vectorstoreid, vectorstorefileid, usecase_id): ...

    @abstractmethod
    async def _list_backend_stores(self, usecase_id, limit, after, before, order, vector_db):
        """List stores from backend."""
        ...

    @abstractmethod
    async def _semantic_search(self, search_request, index_name, model_name):
        # Implement semantic similarity search using ES dense_vector or kNN
        ...

    @abstractmethod
    async def _fulltext_search(self, search_request, index_name):
        # Implement match/multi_match or BM25
        ...

    @abstractmethod
    async def _hybrid_search(self, search_request, index_name, model_name):
        # Combine BM25 + vector similarity with rank fusion
        ...

    # ================================================================
    # RESPONSE BUILDERS
    # ================================================================
    def _build_store_response(
        self, store_id, name, created_at, status, last_active_at, vs_metadata, error_details
    ) -> CreateVectorStoreResponse:
        return CreateVectorStoreResponse(
            id=store_id,
            name=name,
            object="vector_store",
            created_at=created_at,
            last_active_at=last_active_at,
            status=status,
            metadata=vs_metadata,
            file_counts={"total": 0},
            expires_at=None,
            expires_after=None,
            last_error=error_details,
        )

    def _build_storefile_response(
        self, payload, data_size, store_id, status, error_details=None
    ) -> CreateVectorStoreFileResponse:
        return CreateVectorStoreFileResponse(
            id=getattr(payload, "file_id", None),
            object="vector_store.file",
            usage_bytes=data_size,
            created_at=0,
            vector_store_id=store_id,
            status=status,
            attributes=getattr(payload, "attributes", None),
            chunking_strategy=getattr(payload, "chunking_strategy", None),
            last_error=error_details,
        )

    def _build_delete_response(
        self, id_, is_file=False, deleted=True
    ) -> Union[DeleteVectorStoreResponse, DeleteVectorStoreFileResponse]:
        obj = "vector_store.file.deleted" if is_file else "vector_store.deleted"
        resp_cls = DeleteVectorStoreFileResponse if is_file else DeleteVectorStoreResponse
        return resp_cls(id=id_, object=obj, deleted=deleted)

    def _build_retrieve_file_response(
        self, vector_store_file: Optional[Dict[str, Any]], exception: bool = False
    ) -> RetrieveFileResponse:
        if not vector_store_file or exception:
            return RetrieveFileResponse(file_id="", filename="", attributes=[], content=[])
        attributes = [
            AttributesItem(key=k, value=str(v)) for k, v in (vector_store_file.get("attributes") or {}).items()
        ]
        contents = [ContentItem(type="text", text=text) for text in (vector_store_file.get("content") or [])]
        return RetrieveFileResponse(
            file_id=str(vector_store_file.get("file_id", "")),
            filename=str(vector_store_file.get("filename", "")),
            attributes=attributes,
            content=contents,
        )

    def _response_to_object_retrieve_file(
        self,
        vector_store_file: dict,
        exception: bool = False,
    ) -> RetrieveFileResponse:
        """Convert raw Elasticsearch records into a standardized RetrieveFileResponse."""
        if exception or not vector_store_file:
            return RetrieveFileResponse(
                file_id=str(vector_store_file.get("file_id", "")) if vector_store_file else "",
                filename="",
                attributes=[],
                content=[],
            )

        # Normalize attributes
        attributes = []
        raw_attrs = vector_store_file.get("attributes", {})
        if isinstance(raw_attrs, dict):
            for key, value in raw_attrs.items():
                attributes.append(AttributesItem(key=key, value=str(value)))
        elif isinstance(raw_attrs, list):
            for attr in raw_attrs:
                if isinstance(attr, dict):
                    attributes.append(AttributesItem(**attr))

        # Normalize text chunks
        content_items = []
        raw_content = vector_store_file.get("content", [])
        for text_segment in raw_content:
            if isinstance(text_segment, str) and text_segment.strip():
                content_items.append(ContentItem(type="text", text=text_segment.strip()))

        # Deduplicate while preserving order (optional but helpful)
        seen = set()
        unique_content = []
        for item in content_items:
            if item.text not in seen:
                unique_content.append(item)
                seen.add(item.text)

        return RetrieveFileResponse(
            file_id=str(vector_store_file.get("file_id", "")),
            filename=str(vector_store_file.get("filename", "")),
            attributes=attributes,
            content=unique_content,
        )
>> src/services/factory/vector_store_factory.py
import json
import os
from importlib import import_module
from typing import Dict, List, Optional, Type

from src.logging_config import Logger
from src.services.base_class.vector_store_base import BaseVectorStore, VectorStoreConfig

logger = Logger.create_logger(__name__)

VECTORSTORE_MODULES = os.getenv(
    "VECTORSTORE_MODULES",
    "src.services.strategies.vector_store_PG_strategy, src.services.strategies.vector_store_ES_strategy",
).split(",")


class VectorStoreNotFoundError(Exception):
    pass


class VectorStoreFactory:
    _instance = None
    _registry: Dict[str, dict] = {}
    _loaded: bool = False
    _cache: Dict[str, BaseVectorStore] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, description: str = "", tags: Optional[List[str]] = None):
        def decorator(store_cls: Type[BaseVectorStore]):
            cls._registry[name.lower()] = {
                "class": store_cls,
                "description": description or (store_cls.__doc__ or "").strip(),
                "tags": tags or [],
            }
            return store_cls

        return decorator

    @classmethod
    def _load_backends(cls):
        if not cls._loaded:
            for module in VECTORSTORE_MODULES:
                module = module.strip()
                if module:
                    try:
                        import_module(module)
                        logger.debug(f"Loaded vector store backend: {module}")
                    except Exception as e:
                        logger.warning(f"Failed to load vector store backend {module}: {e}")
            cls._loaded = True

    @classmethod
    def list_backends(cls) -> list[dict]:
        cls._load_backends()
        return [{"name": k, "description": v["description"], "tags": v["tags"]} for k, v in cls._registry.items()]

    @classmethod
    def _cache_key(cls, name: str, config: VectorStoreConfig) -> str:
        return f"{name}:{json.dumps(config.dict(), sort_keys=True, default=str)}"

    @classmethod
    def create(cls, name: str, config: Optional[VectorStoreConfig] = None, embedding_service=None) -> BaseVectorStore:
        cls._load_backends()

        meta = cls._registry.get(name.lower())
        if not meta:
            raise VectorStoreNotFoundError(
                f"Vector store backend '{name}' not found. Registered: {list(cls._registry.keys())}"
            )

        config = config or VectorStoreConfig(backend=name)
        key = cls._cache_key(name, config)

        if key in cls._cache:
            logger.debug(f"Using cached backend: {name}")
            return cls._cache[key]

        backend_cls = meta["class"]
        instance = backend_cls(config, embedding_service)
        cls._cache[key] = instance

        logger.info(f"Created new vector store backend: {name}")
        return instance
>> src/services/strategies/vector_store_ES_strategy.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from elasticsearch import Elasticsearch, helpers
from sqlalchemy import asc, desc

from src.db.elasticsearch_connection import get_elasticsearch_client
from src.db.platform_meta_tables import VectorStoreInfo
from src.exception.document_store_exception import (
    DocumentStoreIndexingError,
    DocumentStoreSearchError,
)
from src.exception.exceptions import VectorStoreError
from src.logging_config import Logger
from src.models.vector_store_payload import (
    ContentBlock,
    FileStatus,
    SearchResult,
    StorageBackend,
)
from src.repository.base_repository import BaseRepository
from src.repository.elasticsearch_ddl import ElasticsearchDDL
from src.repository.elasticsearch_dml import ElasticsearchDML
from src.services.base_class.vector_store_base import BaseVectorStore, VectorStoreConfig
from src.services.factory.vector_store_factory import VectorStoreFactory

logger = Logger.create_logger(__name__)


@VectorStoreFactory.register("elasticsearch", description="Elasticsearch + GCP backend")
class ElasticsearchGCPVectorStore(BaseVectorStore):
    """Concrete Elasticsearch + GCP implementation of the Vector Store interface."""

    def __init__(self, config: VectorStoreConfig, embedding_service=None):
        super().__init__(document_repository=None, embedding_service=embedding_service, settings=config.extra)
        self.client: Elasticsearch = get_elasticsearch_client()

    # ================================================================
    # REQUIRED ABSTRACT IMPLEMENTATIONS
    # ================================================================
    async def _create_backend_store(
        self,
        payload,
        usecase_id: str,
        embedding_dimensions: int,
        store_id: str,
        now_dt,
        expires_at,
    ):
        try:
            # Check if ES indices already exist
            existing_indices = ElasticsearchDDL.check_indices_exist(payload.name)
            if existing_indices["both"]:
                logger.warning(f"[ES] Indices for '{payload.name}' already exist")
                raise VectorStoreError(f"Elasticsearch indices already exist for '{payload.name}'")

            # Create Elasticsearch indices (_file_info and _chunks)
            ElasticsearchDDL.create_vectorstore_indices(payload.name, embedding_dimensions)
            logger.info(f"[ES] Created indices for '{payload.name}' with {embedding_dimensions} dims")

            # Build and insert metadata using base class helper
            insert_data = self._build_metadata_dict(
                store_id=store_id,
                payload=payload,
                usecase_id=usecase_id,
                now_dt=now_dt,
                expires_at=expires_at,
                backend_type=StorageBackend.ELASTICSEARCH.value,
            )

            BaseRepository.insert_one(db_tbl=VectorStoreInfo, data=insert_data)
            logger.info(f"[ES] Metadata inserted for store '{store_id}'")

            # Return standardized response using base class helper
            return self._build_create_response_dict(
                store_id=store_id,
                payload=payload,
                now_dt=now_dt,
                expires_at=expires_at,
            )

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"[ES] Store creation failed: {e}", exc_info=True)

            # Rollback: delete metadata and indices
            try:
                BaseRepository.delete(db_tbl=VectorStoreInfo, filters={"id": store_id})
                ElasticsearchDDL.drop_indices(payload.name)
                logger.info(f"[ES] Rollback successful for '{payload.name}'")
            except Exception as cleanup_err:
                logger.warning(f"[ES] Cleanup failed: {cleanup_err}")

            raise VectorStoreError(f"Elasticsearch store creation failed: {str(e)}")

    async def _validate_backend_type(self, record: dict):
        """Ensure the vector_db field indicates Elasticsearch."""
        if record["vector_db"] != StorageBackend.ELASTICSEARCH.value:
            raise VectorStoreError(f"Vector Store '{record['id']}' is not stored in Elasticsearch")

    async def _index_backend(
        self, payload, store_id, store_name, model_name, context_length, model_path, embedding_dimensions
    ):
        chunks_index = f"{store_name}_chunks"
        file_info_index = f"{store_name}_file_info"

        try:
            existing = self.client.get(index=file_info_index, id=payload.file_id, ignore=[404])

            if existing and existing.get("found"):
                raise DocumentStoreIndexingError(
                    f"Duplicate entry found for file_id={payload.file_id}, file_name={payload.file_name}"
                )
        except Exception as es_err:
            if "index_not_found_exception" not in str(es_err):
                raise

        try:
            docs_with_embeddings, data_size, file_info = await self.prepare_index_emdeddings(
                payload=payload,
                model_name=model_name,
                model_path=model_path,
                context_length=context_length,
                embedding_service=self.embedding_service,
                timezone=self.settings.timezone,
            )

            # Bulk index to Elasticsearch
            self._bulk_index_documents(
                chunks_index,
                docs_with_embeddings,
                file_id=payload.file_id,
                file_name=payload.file_name,
            )

            file_info_doc = {
                **file_info,
                "vs_id": store_id,
                "active": True,
            }

            self.client.index(index=file_info_index, id=payload.file_id, body=file_info_doc)

            logger.info(f"[ES] Indexed {len(docs_with_embeddings)} documents for file '{payload.file_id}'")
            return data_size

        except Exception as e:
            logger.exception(f"[ES] Indexing failed: {e}")

            # Cleanup on failure
            try:
                self._delete_documents_by_file_id(chunks_index, payload.file_id)
                self.client.delete(index=file_info_index, id=payload.file_id, ignore=[404])
            except Exception as cleanup_err:
                logger.error(f"[ES] Cleanup failed: {cleanup_err}")

            raise DocumentStoreIndexingError(f"Indexing failed: {str(e)}")

    def _bulk_index_documents(
        self, index_name: str, documents: List[Dict[str, Any]], file_id: str, file_name: str
    ) -> int:
        actions = []
        for doc in documents:
            doc_id = str(uuid4())
            doc_body = {
                "id": doc_id,
                "content": doc.get("content"),
                "embedding": doc.get("embedding"),
                "links": doc.get("links"),
                "topics": doc.get("topics"),
                "author": doc.get("author"),
                "meta_data": doc.get("meta_data"),
                "created_at": datetime.now().isoformat(),
                "file_id": file_id,
                "file_name": file_name,
            }
            actions.append({"_index": index_name, "_id": doc_id, "_source": doc_body})

        try:
            success, failed = helpers.bulk(
                self.client, actions, chunk_size=self.settings.elasticsearch_bulk_chunk_size, raise_on_error=False
            )
            logger.info(f"[ES] Indexed {success} documents, {len(failed)} failed")
            return success
        except Exception as e:
            logger.error(f"[ES] Bulk indexing failed: {e}")
            raise

    def _delete_documents_by_file_id(self, index_name: str, file_id: str) -> int:
        query = {"query": {"term": {"file_id": file_id}}}
        try:
            response = self.client.delete_by_query(index=index_name, body=query)
            return response.get("deleted", 0)
        except Exception as e:
            logger.error(f"[ES] Delete documents failed: {e}")
            raise

    async def _drop_backend_tables(self, store_name: str):
        """Drops Elasticsearch indices (file_info + chunks)."""
        try:
            ElasticsearchDDL.drop_indices(store_name)
            logger.info(f"[ES+GCP] Dropped indices for store '{store_name}'")
        except Exception as err:
            logger.error(f"[ES+GCP] Failed to drop indices: {err}", exc_info=True)
            raise VectorStoreError(f"Elasticsearch index deletion failed for store '{store_name}'")

    async def _search_backend(self, search_request, model_name, context_length, model_path):
        logger.info(f"[ES+GCP] Searching in store {search_request.collection}")
        return ElasticsearchDML.search(index_name=search_request.collection, query=search_request.query)

    async def _fetch_file_backend(self, vectorstoreid, vectorstorefileid, usecase_id):
        store_name = f"{vectorstoreid}_file_info"
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"file_id": vectorstorefileid}},
                        {"term": {"vs_id": vectorstoreid}},
                    ]
                }
            }
        }
        record = ElasticsearchDML.select_one(store_name, query)
        if not record:
            raise VectorStoreError(f":: ES :: File '{vectorstorefileid}' not found in store '{vectorstoreid}'")
        return record

    async def _list_backend_stores(self, usecase_id, limit, after, before, order, vector_db):
        try:
            if order.lower() not in {"asc", "desc"}:
                raise VectorStoreError(f"Invalid orderby value '{order}'")

            order_by_clause = (
                desc(VectorStoreInfo.created_at) if order.lower() == "desc" else asc(VectorStoreInfo.created_at)
            )

            # Build filters
            filters = {"usecase_id": usecase_id}
            if vector_db:
                filters["vector_db"] = vector_db

            response = BaseRepository.select_many(
                db_tbl=VectorStoreInfo,
                filters=filters,
                order_by=order_by_clause,
            )

            logger.info(f"[ES] Listed {len(response)} stores for usecase '{usecase_id}' (filter: {vector_db})")
            return response

        except Exception as e:
            logger.exception(f"[ES] List stores failed: {e}")
            raise VectorStoreError(f"Failed to list stores: {e}")

    # ================================================================
    # DELETE OPERATIONS
    # ================================================================
    async def _delete_metadata_and_chunks(self, store_name: str, vectorstoreid: str, vectorstorefileid: str):
        """
        Delete file metadata and its chunks from Elasticsearch.
        - Delete file info record first → raise if fails.
        - Delete chunks → if fails, restore file info only (no chunk restore).
        """

        vs_file_info = f"{store_name}_file_info"
        vs_chunks = f"{store_name}_chunks"

        file_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"file_id": vectorstorefileid}},
                        {"term": {"vs_id": vectorstoreid}},
                    ]
                }
            }
        }

        # Step 1 — Fetch file info record
        file_info_record = ElasticsearchDML.select_one(vs_file_info, file_query)
        if not file_info_record:
            raise VectorStoreError(
                f":: ES :: File '{vectorstorefileid}' does not exist in vector store '{vectorstoreid}'"
            )

        # Step 2 — Delete file info metadata
        try:
            deleted_meta = ElasticsearchDML.delete(index_name=vs_file_info, doc_id=vectorstorefileid)
            if not deleted_meta or deleted_meta.get("deleted", 0) == 0:
                raise VectorStoreError(f":: ES :: Failed to delete metadata for '{vectorstorefileid}'")
            logger.info(f":: ES :: Deleted file_info '{vectorstorefileid}' from '{vs_file_info}'")
        except Exception as err:
            logger.error(f":: ES :: Metadata delete failed: {err}", exc_info=True)
            raise

        # Step 3 — Fetch chunk references for logging only
        chunk_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"file_id": vectorstorefileid}},
                    ]
                }
            }
        }

        chunk_records = ElasticsearchDML.select_many(vs_chunks, chunk_query)
        logger.info(f":: ES :: Found {len(chunk_records)} chunks for deletion in '{vs_chunks}'")

        # Step 4 — Delete chunks, rollback file_info only if chunk delete fails
        try:
            deleted_chunks = ElasticsearchDML.delete(index_name=vs_chunks, query=chunk_query)
            deleted_count = deleted_chunks.get("deleted", 0)
            if deleted_count <= 0:
                raise VectorStoreError(f":: ES :: No chunks deleted for '{vectorstorefileid}'")

            logger.info(f":: ES :: Successfully deleted {deleted_count} chunks for '{vectorstorefileid}'")

        except Exception as chunk_err:
            logger.error(f":: ES :: Chunk delete failed: {chunk_err}", exc_info=True)

            # Step 5 — Rollback file_info only
            restored = self._restore_file_info_record(vs_file_info, file_info_record)
            if restored:
                logger.info(
                    f":: ROLLBACK :: Restored file_info record for '{vectorstorefileid}' after chunk delete failure"
                )
            else:
                logger.error(f":: ROLLBACK :: Failed to restore file_info record for '{vectorstorefileid}'")

            raise VectorStoreError(
                f":: ES :: Chunk deletion failed. File info rollback {'succeeded' if restored else 'failed'} for '{vectorstorefileid}'"
            )

        logger.info(f":: ES :: Deleted metadata and {len(chunk_records)} chunks for '{vectorstorefileid}' successfully")
        return file_info_record, chunk_records

    def _restore_file_info_record(self, index_name: str, file_info_record: dict) -> bool:
        try:
            if not file_info_record:
                logger.warning(f"No file_info_record provided for restore into {index_name}")
                return False

            restore_data = {
                "active": file_info_record.get("active", "true"),
                "vs_id": file_info_record["vs_id"],
                "file_id": file_info_record["file_id"],
                "file_name": file_info_record["file_name"],
                "file_version": file_info_record.get("file_version", "1"),
                "created_at": file_info_record.get("created_at"),
                "usage_bytes": file_info_record.get("usage_bytes", 0),
                "chunking_strategy": file_info_record.get("chunking_strategy"),
                "attributes": file_info_record.get("attributes") or {},
                "status": file_info_record.get("status", FileStatus.COMPLETED.value),
            }

            self.client.index(
                index=index_name,
                id=file_info_record["file_id"],
                body=restore_data,
            )

            logger.info(f"Successfully restored file_info record '{file_info_record['file_id']}' into {index_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to restore file_info record '{file_info_record.get('file_id', '')}': {e}",
                exc_info=True,
            )
            return False

    async def _semantic_search(
        self,
        search_request: Any,
        index_name: str,
        model_name: str,
    ) -> List[SearchResult]:
        embeddings = await self.embedding_service.get_embeddings(
            model_name=model_name,
            batch=[search_request.search_text],
        )
        query_vector = embeddings.data[0].embedding

        filters = self._build_filters(search_request)

        knn_query = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": search_request.limit,
            "num_candidates": search_request.limit * 10,
        }

        if filters:
            knn_query["filter"] = filters

        query_body = {
            "knn": knn_query,
            "min_score": search_request.min_score,
            "size": search_request.limit,
            "_source": {"excludes": ["embedding"]},
        }

        response = self.client.search(index=index_name, body=query_body)

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                SearchResult(
                    id=source.get("id", ""),
                    file_id=source.get("file_id", ""),
                    filename=source.get("file_name", ""),
                    score=hit["_score"],
                    content=[ContentBlock(type="text", text=source.get("content", ""))],
                    attributes=source.get("meta_data", {}),
                )
            )

        return results

    async def _fulltext_search(
        self,
        search_request: Any,
        index_name: str,
    ) -> List[SearchResult]:
        filters = self._build_filters(search_request)

        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": search_request.search_text,
                                "fields": ["content^2", "topics", "author", "links"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        }
                    ]
                }
            },
            "min_score": search_request.min_score,
            "size": search_request.limit,
            "_source": {"excludes": ["embedding"]},
        }

        if filters:
            query_body["query"]["bool"]["filter"] = filters

        response = self.client.search(index=index_name, body=query_body)

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                SearchResult(
                    id=source.get("id", ""),
                    file_id=source.get("file_id", ""),
                    filename=source.get("file_name", ""),
                    score=hit["_score"],
                    content=[ContentBlock(type="text", text=source.get("content", ""))],
                    attributes=source.get("meta_data", {}),
                )
            )

        return results

    async def _hybrid_search(
        self,
        search_request: Any,
        index_name: str,
        model_name: str,
    ) -> List[SearchResult]:
        try:
            embeddings = await self.embedding_service.get_embeddings(
                model_name=model_name,
                batch=[search_request.search_text],
            )
            query_vector = embeddings.data[0].embedding

            filters = self._build_filters(search_request)

            query_body = {
                "rank": {
                    "rrf": {
                        "window_size": getattr(self.settings, "rrf_window_size", 100),
                        "rank_constant": getattr(self.settings, "rrf_rank_constant", 60),
                    }
                },
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": search_request.search_text,
                                    "fields": ["content^2", "topics", "author", "links"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",
                                }
                            }
                        ],
                        **({"filter": filters} if filters else {}),
                    }
                },
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": search_request.limit,
                    "num_candidates": search_request.limit * 10,
                    **({"filter": filters} if filters else {}),
                },
                "_source": {"excludes": ["embedding"]},
                "size": search_request.limit,
            }

            response = self.client.search(index=index_name, body=query_body)

            results: List[SearchResult] = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append(
                    SearchResult(
                        id=source.get("id", ""),
                        file_id=source.get("file_id", ""),
                        filename=source.get("file_name", ""),
                        score=hit.get("_score", 0.0) or 0.0,
                        content=[ContentBlock(type="text", text=source.get("content", ""))],
                        attributes=source.get("meta_data", {}),
                    )
                )

            logger.info(
                f"[HYBRID] Native RRF returned {len(results)} results from index '{index_name}' "
                f"(window={query_body['rank']['rrf']['window_size']}, k={query_body['rank']['rrf']['rank_constant']})"
            )
            return results

        except Exception as e:
            logger.exception(f"[HYBRID] Native RRF search failed: {str(e)}")
            raise DocumentStoreSearchError(f"Hybrid search failed: {str(e)}")

    def _build_filters(self, search_request: Any) -> Optional[Dict[str, Any]]:
        filters = []

        if search_request.content_filter:
            filters.append({"terms": {"content": search_request.content_filter}})

        if search_request.link_filter:
            filters.append({"terms": {"links": search_request.link_filter}})

        if search_request.topic_filter:
            filters.append({"terms": {"topics": search_request.topic_filter}})

        if not filters:
            return None

        return {"bool": {"must": filters}} if len(filters) > 1 else filters[0]

>> src/services/strategies/vector_store_PG_strategy.py
from collections import defaultdict
from typing import Dict, List

from sqlalchemy import asc, desc

from src.db.connection import create_session
from src.db.platform_meta_tables import VectorStoreInfo
from src.exception.document_store_exception import DocumentStoreIndexingError
from src.exception.exceptions import VectorStoreError
from src.logging_config import Logger
from src.models.storage_payload import SearchRequest
from src.models.vector_store_payload import SearchResult, StorageBackend
from src.repository.base_repository import BaseRepository
from src.repository.document_repository import DocumentRepository
from src.repository.vectorstore_ddl import VectorStoreDDL
from src.services.base_class.vector_store_base import BaseVectorStore, VectorStoreConfig
from src.services.factory.vector_store_factory import VectorStoreFactory
from src.utility.vector_store_utils import (
    create_chunks_tbl_model,
    create_file_info_tbl_model,
)

logger = Logger.create_logger(__name__)


@VectorStoreFactory.register("pgvector", description="Postgres + pgvector backend")
class PGVectorStore(BaseVectorStore):
    """Concrete PGVector implementation of the Vector Store interface."""

    def __init__(self, config: VectorStoreConfig, embedding_service=None):
        super().__init__(document_repository=None, embedding_service=embedding_service, settings=config.extra)

    # =================================================================
    # Implement all abstract methods required by BaseVectorStore
    # =================================================================

    async def _create_backend_store(
        self,
        payload,
        usecase_id: str,
        embedding_dimensions: int,
        store_id: str,
        now_dt,
        expires_at,
    ):
        try:
            # Create PostgreSQL tables with pgvector extension
            VectorStoreDDL.create_tables_and_index(table_name=payload.name, dimensions=embedding_dimensions)
            logger.info(f"[PGVector] Created tables for '{payload.name}' with {embedding_dimensions} dims")

            # Build and insert metadata using base class helper
            insert_data = self._build_metadata_dict(
                store_id=store_id,
                payload=payload,
                usecase_id=usecase_id,
                now_dt=now_dt,
                expires_at=expires_at,
                backend_type=StorageBackend.PGVECTOR.value,
            )

            BaseRepository.insert_one(db_tbl=VectorStoreInfo, data=insert_data)
            logger.info(f"[PGVector] Metadata inserted for store '{store_id}'")

            # Return standardized response using base class helper
            return self._build_create_response_dict(
                store_id=store_id,
                payload=payload,
                now_dt=now_dt,
                expires_at=expires_at,
            )

        except VectorStoreError:
            raise
        except Exception as e:
            logger.error(f"[PGVector] Store creation failed: {e}", exc_info=True)

            # Rollback: delete metadata and tables
            try:
                BaseRepository.delete(db_tbl=VectorStoreInfo, filters={"id": store_id})
                VectorStoreDDL.drop_table_and_index(tbl_name=payload.name)
                logger.info(f"[PGVector] Rollback successful for '{payload.name}'")
            except Exception as cleanup_err:
                logger.warning(f"[PGVector] Cleanup failed: {cleanup_err}")

            raise VectorStoreError(f"PGVector store creation failed: {str(e)}")

    async def _index_backend(
        self, payload, store_id, store_name, model_name, context_length, model_path, embedding_dimensions
    ):
        try:
            vs_file_info = create_file_info_tbl_model(f"{store_name}_file_info")
            existing = BaseRepository.find_one(
                db_tbl=vs_file_info,
                filters={"file_id": payload.file_id, "vs_id": store_id},
                session_factory=create_session,
            )
            if existing:
                raise DocumentStoreIndexingError(
                    f"Duplicate entry found for file_id={payload.file_id}, file_name={payload.file_name}"
                )

            docs_with_embeddings, data_size, file_info = await self.prepare_index_emdeddings(
                payload=payload,
                model_name=model_name,
                model_path=model_path,
                context_length=context_length,
                embedding_service=self.embedding_service,
                timezone=self.settings.timezone,
            )

            document_repository = DocumentRepository(f"{store_name}_chunks", embedding_dimensions=embedding_dimensions)

            document_repository.insert_documents(
                table_name=f"{store_name}_chunks",
                documents=docs_with_embeddings,
                service_type="vectorstore",
                file_id=payload.file_id,
                file_name=payload.file_name,
            )

            file_metadata = {
                **file_info,
                "vs_id": store_id,
                "active": True,
            }

            BaseRepository.insert_one(db_tbl=vs_file_info, data=file_metadata, session_factory=create_session)

            logger.info(f"[PGVector] Indexed {len(docs_with_embeddings)} documents for file '{payload.file_id}'")
            return data_size

        except Exception as e:
            logger.exception(f"[PGVector] Indexing failed: {e}")

            # Cleanup on failure
            try:
                vs_file_info = create_file_info_tbl_model(f"{store_name}_file_info")
                vs_chunks = create_chunks_tbl_model(f"{store_name}_chunks", embedding_dimensions)

                BaseRepository.delete(
                    db_tbl=vs_file_info,
                    filters={"file_id": payload.file_id, "vs_id": store_id},
                    session_factory=create_session,
                )
                BaseRepository.delete(
                    db_tbl=vs_chunks,
                    filters={"file_id": payload.file_id},
                    session_factory=create_session,
                )
            except Exception as cleanup_err:
                logger.warning(f"[PGVector] Cleanup failed: {cleanup_err}")

            raise DocumentStoreIndexingError(f"Indexing failed: {str(e)}")

    async def _search_backend(self, payload, store_id, model_name, context_length, model_path):
        logger.info(f"[PGVector] Searching store {store_id}")
        return await self._service.search_vector_store(payload, store_id, model_name, context_length, model_path)

    async def _drop_backend_tables(self, store_name: str):
        from src.repository.vectorstore_ddl import VectorStoreDDL

        VectorStoreDDL.drop_table_and_index(tbl_name=store_name)

    async def _fetch_file_backend(self, vectorstoreid, vectorstorefileid, usecase_id):
        logger.info(f"[PGVector] Retrieving file {vectorstorefileid} from store {vectorstoreid}")
        return await self._service.retrieve_by_id(vectorstoreid, vectorstorefileid, usecase_id)

    async def _list_backend_stores(self, usecase_id, limit, after, before, order, vector_db):
        try:
            if order.lower() not in {"asc", "desc"}:
                raise VectorStoreError(f"Invalid orderby value '{order}'")

            order_by_clause = (
                desc(VectorStoreInfo.created_at) if order.lower() == "desc" else asc(VectorStoreInfo.created_at)
            )

            # Build filters
            filters = {"usecase_id": usecase_id}
            if vector_db:  # If backend specified, filter by it
                filters["vector_db"] = vector_db

            response = BaseRepository.select_many(
                db_tbl=VectorStoreInfo,
                filters=filters,
                order_by=order_by_clause,
            )

            logger.info(f"[PGVector] Listed {len(response)} stores for usecase '{usecase_id}' (filter: {vector_db})")
            return response

        except Exception as e:
            logger.exception(f"[PGVector] List stores failed: {e}")
            raise VectorStoreError(f"Failed to list stores: {e}")

    async def _validate_backend_type(self, record: dict):
        if record["vector_db"] != StorageBackend.PGVECTOR.value:
            raise VectorStoreError(f"Vector Store '{record['id']}' is not stored in PGVector")

    async def _delete_metadata_and_chunks(self, store_name: str, vectorstoreid: str, vectorstorefileid: str):
        vs_file_info_tbl = create_file_info_tbl_model(f"{store_name}_file_info")
        vs_chunks_tbl = create_chunks_tbl_model(f"{store_name}_chunks", dimensions=0)
        file_info_record = BaseRepository.select_one(
            db_tbl=vs_file_info_tbl,
            filters={"file_id": vectorstorefileid, "vs_id": vectorstoreid},
            session_factory=create_session,
        )
        if not file_info_record:
            raise VectorStoreError(f":: PG :: File '{vectorstorefileid}' not found in store '{vectorstoreid}'")
        chunk_records = BaseRepository.select_many(
            db_tbl=vs_chunks_tbl,
            filters={"file_id": vectorstorefileid},
            session_factory=create_session,
        )
        chunk_count = len(chunk_records) if chunk_records else 0
        logger.info(f":: PG :: Found {chunk_count} chunks for file '{vectorstorefileid}'")
        deleted_meta = BaseRepository.delete(
            db_tbl=vs_file_info_tbl,
            filters={"file_id": vectorstorefileid, "vs_id": vectorstoreid},
            session_factory=create_session,
        )
        if deleted_meta == 0:
            raise VectorStoreError(f":: PG :: Metadata delete failed for '{vectorstorefileid}'")
        try:
            deleted_chunks = BaseRepository.delete(
                db_tbl=vs_chunks_tbl,
                filters={"file_id": vectorstorefileid},
                session_factory=create_session,
            )
            if deleted_chunks == 0:
                raise VectorStoreError(f":: PG :: No chunks deleted for '{vectorstorefileid}'")
        except Exception as chunk_err:
            logger.error(f":: PG :: Chunk delete failed: {chunk_err}", exc_info=True)
            try:
                BaseRepository.insert_one(
                    db_tbl=vs_file_info_tbl,
                    data=file_info_record,
                    session_factory=create_session,
                )
                logger.info(f":: ROLLBACK :: Restored metadata for '{vectorstorefileid}' after chunk deletion failure.")
            except Exception as rollback_err:
                logger.error(f":: ROLLBACK :: Failed to restore metadata: {rollback_err}", exc_info=True)
            raise VectorStoreError(f"Chunk deletion failed — metadata restored for '{vectorstorefileid}'")

        logger.info(f":: PG :: Deleted metadata and {chunk_count} chunks for file '{vectorstorefileid}' successfully.")
        return file_info_record, chunk_records

    async def fulltext_search(self, search_request: SearchRequest) -> list[SearchResult]:
        _, results = self.document_repository.fulltext_search(
            query=search_request.search_text,
            search_terms=search_request.content_filter,
            include_links=search_request.link_filter,
            include_topics=search_request.topic_filter,
            top_k=search_request.limit,
            min_relevance_score=search_request.min_score,
        )
        return results

    async def semantic_search(self, search_request: SearchRequest) -> list[SearchResult]:
        embeddings = await self.embedding_service.get_embeddings(
            model_name=self.settings.default_model_embeddings,
            batch=[search_request.search_text],
        )
        query_vector = embeddings.data[0].embedding
        _, results = self.document_repository.sematic_search(
            query_vector=query_vector,
            search_terms=search_request.content_filter,
            include_links=search_request.link_filter,
            include_topics=search_request.topic_filter,
            top_k=search_request.limit,
            min_similarity_score=search_request.min_score,
        )
        return results

    async def hybrid_search(self, search_request: SearchRequest) -> list[SearchResult]:
        semantic_results: list[SearchResult] = await self.semantic_search(search_request)
        fulltext_results: list[SearchResult] = await self.fulltext_search(search_request)
        logger.info(
            f"Hybrid search -> Semantic search results: {len(semantic_results)}, "
            f"Full-text search results: {len(fulltext_results)}"
        )
        score_map = defaultdict(lambda: {"semantic": 0.0, "fulltext": 0.0})  # type: ignore
        result_map: Dict[str, SearchResult] = {}
        for res in semantic_results:
            score_map[res.file_id]["semantic"] = res.score
            result_map[res.file_id] = res
        for res in fulltext_results:
            score_map[res.file_id]["fulltext"] = res.score
            result_map.setdefault(res.file_id, res)
        reranked: List[SearchResult] = []
        for file_id_, scores in score_map.items():
            weighted_score = round(0.6 * scores["semantic"] + 0.4 * scores["fulltext"], 4)
            result = result_map[file_id_]
            result.score = weighted_score
            reranked.append(result)
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[: search_request.limit]
