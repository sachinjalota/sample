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
    extra: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------
# BaseVectorStore (Template Method pattern)
# ---------------------------------------------------------------------
class BaseVectorStore(ABC):
    """
    Abstract base for all vector store backends (PGVector, Elasticsearch, etc.)
    Implements all CRUD orchestration, metadata handling, and error recovery.
    """

    def __init__(
        self, 
        document_repository: Any = None, 
        embedding_service: Any = None, 
        settings: Any = None
    ) -> None:
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
    def _generate_store_metadata(
        self, payload: CreateVectorStoreRequest
    ) -> Tuple[str, datetime, Optional[datetime]]:
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
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
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
    ) -> Dict[str, Any]:
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
    async def _fetch_store_metadata(self, store_id: str, storage_backend: str, usecase_id: str) -> Dict[str, Any]:
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
        embedding_service: Any,
        timezone: str,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
        content_list: List[str] = []
        for document in payload.file_contents:
            self._content_length_validation(context_length, model_name, model_path, document.content, "index")
            content_list.append(document.content)

        embeddings = await embedding_service.get_embeddings(model_name=model_name, batch=content_list)

        docs_with_embeddings: List[Dict[str, Any]] = []
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

        common_file_info: Dict[str, Any] = {
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

    async def _update_store_stats_after_file_add(self, store_record: Dict[str, Any], file_size: int) -> bool:
        try:
            # Update file counts
            current_file_counts = store_record.get("file_counts", {}) or FileCountsModel().model_dump()
            updated_file_counts = current_file_counts.copy()
            updated_file_counts["completed"] = updated_file_counts.get("completed", 0) + 1
            updated_file_counts["total"] = updated_file_counts.get("total", 0) + 1

            # Update usage and timestamp
            now_dt = datetime.now(ZoneInfo(self.settings.timezone))
            new_usage = (store_record.get("usage_bytes", 0) or 0) + file_size

            update_data: Dict[str, Any] = {
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
        self, stores: List[Dict[str, Any]], limit: int, after: Optional[str], before: Optional[str]
    ) -> List[CreateVectorStoreResponse]:
        response_stores: List[CreateVectorStoreResponse] = []

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

    async def _fetch_metadata(self, vector_id: str, usecase_id: str) -> Dict[str, Any]:
        from src.db.platform_meta_tables import VectorStoreInfo
        from src.repository.base_repository import BaseRepository

        col = BaseRepository.select_one(db_tbl=VectorStoreInfo, filters={"id": vector_id})
        if not col:
            raise VectorStoreError(f"Vector Store '{vector_id}' does not exist")
        if col["usecase_id"] != usecase_id:
            raise VectorStoreError(f"Access denied for Vector Store '{vector_id}'")
        return col

    async def _delete_metadata(self, vector_id: str) -> None:
        from src.db.platform_meta_tables import VectorStoreInfo
        from src.repository.base_repository import BaseRepository

        BaseRepository.delete(db_tbl=VectorStoreInfo, filters={"id": vector_id})
        logger.info(f"Deleted metadata for vector store '{vector_id}'")

    def _restore_metadata_on_failure(self, backup_record: Dict[str, Any]) -> None:
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

            update_data: Dict[str, Any] = {
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
    ) -> Dict[str, Any]:
        """Create backend store and return response dict"""
        ...

    @abstractmethod
    async def _validate_backend_type(self, record: Dict[str, Any]) -> None:
        """Ensures backend type matches the store type (e.g., pgvector / elasticsearch)."""
        ...

    @abstractmethod
    async def _drop_backend_tables(self, store_name: str) -> None:
        """Drops backend tables/indices"""
        ...

    @abstractmethod
    async def _delete_metadata_and_chunks(
        self, store_name: str, vectorstoreid: str, vectorstorefileid: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Delete metadata and chunks from backend.
        Returns: (file_info_record, chunk_records)
        """
        ...

    @abstractmethod
    async def _index_backend(
        self,
        payload: CreateVectorStoreFileRequest,
        store_id: str,
        store_name: str,
        model_name: str,
        context_length: int,
        model_path: str,
        embedding_dimensions: int,
    ) -> int:
        """
        Index documents to backend.
        Returns: data_size in bytes
        """
        ...

    @abstractmethod
    async def _fetch_file_backend(
        self, vectorstoreid: str, vectorstorefileid: str, usecase_id: str
    ) -> Dict[str, Any]:
        """Fetch file metadata from backend"""
        ...

    @abstractmethod
    async def _list_backend_stores(
        self,
        usecase_id: str,
        limit: int,
        after: Optional[str],
        before: Optional[str],
        order: str,
        vector_db: Optional[str],
    ) -> List[Dict[str, Any]]:
        """List stores from backend"""
        ...

    @abstractmethod
    async def _semantic_search(
        self, search_request: Any, index_name: str, model_name: str
    ) -> List[SearchResult]:
        """Semantic similarity search using embeddings"""
        ...

    @abstractmethod
    async def _fulltext_search(self, search_request: Any, index_name: str) -> List[SearchResult]:
        """Full-text BM25 search"""
        ...

    @abstractmethod
    async def _hybrid_search(
        self, search_request: Any, index_name: str, model_name: str
    ) -> List[SearchResult]:
        """Hybrid search combining semantic + fulltext"""
        ...

    # ================================================================
    # RESPONSE BUILDERS
    # ================================================================
    def _build_store_response(
        self,
        store_id: str,
        name: str,
        created_at: int,
        status: str,
        last_active_at: int,
        vs_metadata: Optional[Dict[str, Any]],
        error_details: Optional[VectorStoreErrorDetails],
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
        self,
        payload: CreateVectorStoreFileRequest,
        data_size: int,
        store_id: str,
        status: str,
        error_details: Optional[VectorStoreErrorDetails] = None,
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
        self, id_: str, is_file: bool = False, deleted: bool = True
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
        vector_store_file: Dict[str, Any],
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
        attributes: List[AttributesItem] = []
        raw_attrs = vector_store_file.get("attributes", {})
        if isinstance(raw_attrs, dict):
            for key, value in raw_attrs.items():
                attributes.append(AttributesItem(key=key, value=str(value)))
        elif isinstance(raw_attrs, list):
            for attr in raw_attrs:
                if isinstance(attr, dict):
                    attributes.append(AttributesItem(**attr))

        # Normalize text chunks
        content_items: List[ContentItem] = []
        raw_content = vector_store_file.get("content", [])
        for text_segment in raw_content:
            if isinstance(text_segment, str) and text_segment.strip():
                content_items.append(ContentItem(type="text", text=text_segment.strip()))

        # Deduplicate while preserving order (optional but helpful)
        seen = set()
        unique_content: List[ContentItem] = []
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
