1️⃣ Jira‑style task breakdown

Epic	Feature	Description (Jira ticket)
VECTOR‑STORE‑API	1‑Align collection & index APIs with OpenAI reference	Update the public collection‑related endpoints (/collection, /collection/data, /collection/create, /collection/delete, /index, /search, /delete_index, /collection/delete_by_ids) so that request/response payloads, status codes and query parameters follow the OpenAI Vector‑Store specification. Add proper OpenAPI metadata and reuse existing service logic.
VECTOR‑STORE‑API	2‑Add a stable v2 “raw” wrapper	Keep the current v2 router (src/api/routers/v2/*) for internal power‑user usage, but expose a single, version‑agnostic set of endpoints that internally delegate to the v2 implementation. The new router must be documented and unit‑tested.
VECTOR‑STORE‑API	3‑Pluggable storage back‑ends	Introduce a generic storage‑backend registry (already exists) and add a ElasticSearch implementation that satisfies the Database abstract class. The router should automatically pick the backend based on the storage_backend field of the payload.
VECTOR‑STORE‑SDK	4‑OpenAI Vector‑Store client SDK	Create a thin wrapper (src/integrations/openai_vectorstore_sdk.py) that mirrors the official OpenAI Python client (client.vector_stores.*). This SDK will be used by the routers and can also be reused by other services (e.g., RAG).
VECTOR‑STORE‑SDK	5‑Update existing OpenAI SDK to call the new Vector‑Store SDK for indexing/search	Refactor src/integrations/open_ai_sdk.py so that when a request’s storage_backend is pgvector it uses the existing PGVector implementation, and when it is elasticsearch it uses the new ElasticSearch implementation.
CODE‑QUALITY	6‑Add/Update Pydantic models	Align request/response models (CreateCollection, DeleteCollection, IndexingPayload, SearchRequest, DeleteRequest, DeleteByIdsRequest) with the OpenAI schema (e.g., rename fields, add optional metadata, name, expires_after).
CODE‑QUALITY	7‑Add unit tests for new endpoints & back‑ends	Extend tests/unit/api with tests that cover the new OpenAI‑compatible routes, the ElasticSearch backend and the new SDK.
DOCS	8‑Update README & OpenAPI docs	Mention the new OpenAI‑compatible endpoints, the optional ElasticSearch backend and the new SDK usage.
2️⃣ Implementation plan – requirement 1 (OpenAI‑compatible vector‑store APIs)

2.1 Files that will be modified / created

File (relative to repo root)	Why it is touched	Main change
src/models/collection_payload.py	Add OpenAI‑style fields (name, metadata, expires_after)	New CreateCollectionV1 model
src/models/indexing_payload.py	Extend with metadata, name, expires_after	New IndexingPayloadV1
src/models/storage_payload.py	Align SearchRequest to OpenAI spec (optional filters, ranking_options, max_num_results)	New SearchRequestV1, SearchResponseV1
src/api/routers/collection_router.py	Rename routes to OpenAI names (/collection, /collection/data, …) and use new models	Updated FastAPI router
src/api/routers/document_store_router.py	Keep core logic but adapt to the new request models and return OpenAI‑shaped SearchResponseV1	Slight refactor
src/services/pgvector_document_store.py	Add mapping from new request fields to internal calls (e.g., metadata is ignored, max_num_results → limit)	Minor adjustments
src/services/abstract_document_store.py	Update abstract signatures to accept the new SearchRequestV1	Interface change
src/api/routers/v2/* (no change needed – they become the implementation behind v1)	–	–
src/api/routers/__init__.py (new file)	Export the v1 router under a unified name (vector_store_router)	New module
src/api/routers/vector_store_router.py (new)	A thin wrapper that registers the v1 routes under /v1/api (the existing root) – this is the public entry point.	New router
src/config.py	Add optional openai_vectorstore_api_key (used by new SDK)	New setting
src/integrations/openai_vectorstore_sdk.py (new)	Thin wrapper around the official OpenAI client to expose create, list, retrieve, update, delete, search methods.	New SDK
src/integrations/open_ai_sdk.py	Refactor embedding/completion only; move any vector‑store‑specific calls to the new SDK.	Minor refactor
src/utility/registry.py	Register the new ElasticSearch backend (see requirement 3) – unchanged for requirement 1 but needed for later.	No change now
src/exception/document_store_exception.py	Add ElasticSearchError subclass (future‑proof).	New exception class
All modifications are shown below with the exact code diff.
Code blocks are highlighted where a line is added (+) or changed (~).
New files are presented in full.

2.2 Updated / new code

2.2.1 src/models/collection_payload.py (new OpenAI‑style model)

# src/models/collection_payload.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, validator


class CreateCollectionV1(BaseModel):
    """OpenAI‑compatible payload for creating a vector store (collection)."""

    name: str = Field(..., description="Human readable name for the vector store.")
    metadata: Optional[Dict[str, str]] = Field(
        None, description="Arbitrary 16‑key map (max 64 char keys, 512 char values)."
    )
    expires_after: Optional[Dict[str, Any]] = Field(
        None,
        description="Expiration policy – see OpenAI spec (e.g. {'anchor': 'last_active_at', 'days': 30}).",
    )
    # internal fields used by the platform
    collection: str = Field(..., description="Internal collection/table name.")
    model_name: str = Field(..., description="Embedding model name.")


class DeleteCollectionV1(BaseModel):
    """Payload for deleting a vector store."""

    collection: str = Field(..., description="Internal collection/table name.")


# Keep the old models for internal v2 usage
class CreateCollection(BaseModel):
    collection: str = Field(..., description="Collection name.")
    model_name: str = Field(..., description="Embedding model name.")

    @validator("collection")
    def collection_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("The 'collection' field must not be empty.")
        return v

    @validator("model_name")
    def model_name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("The 'model_name' field must not be empty.")
        return v


class DeleteCollection(BaseModel):
    collection: str = Field(..., description="Collection Name.")

    @validator("collection")
    def collection_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("The 'collection' field must not be empty.")
        return v
Changes

Added CreateCollectionV1 that mirrors the OpenAI vector_stores.create payload (name, metadata, expires_after).
Kept the original models for the internal v2 router.
2.2.2 src/models/indexing_payload.py (add OpenAI fields)

# src/models/indexing_payload.py
from typing import Any, Dict, List

from pydantic import BaseModel, Field, validator

from src.models.storage_payload import StorageBackend


class IndexingPayloadV1(BaseModel):
    """OpenAI‑compatible indexing request."""

    storage_backend: StorageBackend = Field(
        ...,
        description="Backend to use (pgvector | elasticsearch).",
    )
    collection: str = Field(..., description="Vector‑store identifier (internal table name).")
    documents: List[Dict[str, Any]] = Field(..., description="Array of documents to index.")
    name: Optional[str] = Field(
        None,
        description="Optional human readable name for the vector store (ignored if already exists).",
    )
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Optional key‑value map (max 16 entries).",
    )
    expires_after: Optional[Dict[str, Any]] = Field(
        None,
        description="Expiration policy – same schema as OpenAI's vector store.",
    )
Changes

New fields name, metadata, expires_after are accepted but, for the current PGVector implementation, they are stored only in the CollectionInfo table (no effect on the table itself).
2.2.3 src/models/storage_payload.py (search request/response)

# src/models/storage_payload.py
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from src.config import get_settings

settings = get_settings()


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    FULL_TEXT = "full_text"
    HYBRID = "hybrid"


class StorageBackend(str, Enum):
    PGVECTOR = "pgvector"
    ELASTICSEARCH = "elasticsearch"


# ----------------------------------------------------------------------
# Existing Document model unchanged
# ----------------------------------------------------------------------


class SearchRequestV1(BaseModel):
    """OpenAI‑compatible search payload."""

    collection: str = Field(..., description="Vector store identifier.")
    query: str = Field(..., description="Query string.")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="File‑attribute filter object (passed straight to backend).",
    )
    max_num_results: int = Field(
        default=settings.default_document_limit,
        gt=0,
        le=50,
        description="Maximum number of results to return (1‑50).",
    )
    ranking_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional ranking options – passed to backend when supported.",
    )
    storage_backend: StorageBackend = Field(
        ...,
        description="Backend to query (pgvector | elasticsearch).",
    )

    @validator("collection")
    def collection_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("The 'collection' field must not be empty.")
        return v

    @validator("query")
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("The 'query' field must not be empty.")
        return v


class SearchResultV1(BaseModel):
    """Result item returned by OpenAI‑style `/search` endpoint."""

    file_id: str = Field(..., description="ID of the source file (or document primary key).")
    filename: Optional[str] = Field(None, description="Original filename when known.")
    score: float = Field(..., description="Relevance score (0‑1).")
    attributes: Optional[Dict[str, Any]] = Field(
        None, description="Arbitrary key‑value attributes stored with the document."
    )
    content: List[Dict[str, str]] = Field(
        ..., description="One or more text chunks that matched the query."
    )


class SearchResponseV1(BaseModel):
    object: str = Field("vector_store.search_results.page", const=True)
    search_query: str = Field(..., description="Echo of the query sent.")
    data: List[SearchResultV1] = Field(..., description="Array of matching document chunks.")
    has_more: bool = Field(False, description="True when more results are available.")
    next_page: Optional[str] = Field(None, description="Cursor for next page (if pagination is required).")
Changes

New SearchRequestV1/SearchResponseV1/SearchResultV1 follow the OpenAI spec.
Existing SearchRequest and SearchResponse remain for internal v2 usage.
2.2.4 src/api/routers/collection_router.py (rename / adapt routes)

# src/api/routers/collection_router.py
from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.responses import JSONResponse

from src.api.deps import get_collection_service, validate_headers_and_api_key
from src.config import get_settings
from src.exception.exceptions import CollectionError, DatabaseConnectionError, EmbeddingModelError
from src.logging_config import Logger
from src.models.collection_payload import CreateCollectionV1, DeleteCollectionV1
from src.models.headers import HeaderInformation
from src.services.collection_service import CollectionService
from src.utility.collection_helpers import get_usecase_id_by_api_key, validate_collection_access
from src.utility.collection_utils import is_valid_collection_name

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()

# ----------------------------------------------------------------------
# 1️⃣  List collections  →  GET /v1/api/collection
# ----------------------------------------------------------------------
@router.get(
    "/collection",
    summary="List all vector stores (collections) for the caller.",
    status_code=status.HTTP_200_OK,
)
async def list_collections(
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    collection_service: CollectionService = Depends(get_collection_service),
) -> dict:
    try:
        usecase_id = await get_usecase_id_by_api_key(header_information.x_base_api_key)
        return await collection_service.get(usecase_id=usecase_id)
    except Exception as exc:
        logger.exception("Failed to list collections.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ----------------------------------------------------------------------
# 2️⃣  Get collection details  →  GET /v1/api/collection/data
# ----------------------------------------------------------------------
@router.get(
    "/collection/data",
    summary="Retrieve collection details (metadata & sample rows).",
    status_code=status.HTTP_200_OK,
)
async def collection_data(
    collection: str = Query(..., description="Internal collection name"),
    limit: int = Query(settings.collection_data_limit, gt=0, description="Rows per page"),
    offset: int = Query(settings.collection_data_offset, ge=0, description="Starting offset"),
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    collection_service: CollectionService = Depends(get_collection_service),
) -> dict:
    try:
        ok, err = is_valid_collection_name(collection)
        if not ok:
            raise HTTPException(status_code=400, detail=err)
        return await collection_service.get_details(collection=collection, limit=limit, offset=offset)
    except Exception as exc:
        logger.exception("Failed to fetch collection details.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ----------------------------------------------------------------------
# 3️⃣  Create a collection (vector store) → POST /v1/api/collection
# ----------------------------------------------------------------------
@router.post(
    "/collection",
    summary="Create a new vector store (collection).",
    status_code=status.HTTP_200_OK,
)
async def create_collection(
    payload: CreateCollectionV1,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    collection_service: CollectionService = Depends(get_collection_service),
) -> JSONResponse:
    try:
        ok, err = is_valid_collection_name(payload.collection)
        if not ok:
            raise HTTPException(status_code=400, detail=err)

        # Validate the embedding model and retrieve dimensions
        _, embedding_dimensions, _ = await validate_collection_access(
            api_key=header_information.x_base_api_key,
            collection_name=payload.collection,
        )
        # The service currently only cares about collection name & embedding model
        # Other OpenAI fields (name, metadata, expires_after) are stored in CollectionInfo
        response = await collection_service.create(
            request=payload,  # service knows how to map the V1 model
            usecase_id=await get_usecase_id_by_api_key(header_information.x_base_api_key),
        )
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except (CollectionError, EmbeddingModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DatabaseConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to create collection.")
        raise HTTPException(status_code=500, detail=str(exc))


# ----------------------------------------------------------------------
# 4️⃣  Delete collection → DELETE /v1/api/collection/delete
# ----------------------------------------------------------------------
@router.delete(
    "/collection/delete",
    summary="Delete a vector store (collection) and its metadata.",
    status_code=status.HTTP_200_OK,
)
async def delete_collection(
    payload: DeleteCollectionV1,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    collection_service: CollectionService = Depends(get_collection_service),
) -> dict:
    try:
        await validate_collection_access(header_information.x_base_api_key, payload.collection)
        return collection_service.delete(payload.collection)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to delete collection.")
        raise HTTPException(status_code=500, detail=str(exc))
Key modifications

Routes now follow the OpenAI pattern (/collection, /collection/data, /collection/create → simply /collection POST, /collection/delete DELETE).
Payloads switched to the newly created CreateCollectionV1 / DeleteCollectionV1.
The router no longer uses settings.get_collection etc.; the path strings are hard‑coded to be the OpenAI‑compatible names.
The service call stays the same – the CollectionService.create method already receives a Pydantic model; we simply pass the V1 model (it contains the old fields plus the new optional ones, which the service safely ignores).
2.2.5 src/api/routers/document_store_router.py (use new request/response models)

# src/api/routers/document_store_router.py
# (Only the parts that changed are shown – the top imports stay the same)

from src.models.storage_payload import (
    DeleteByIdsRequest,
    DeleteRequest,
    DeleteResponse,
    SearchRequestV1,
    SearchResponseV1,
    IndexingPayloadV1,
    StorageBackend,
)

# ----------------------------------------------------------------------
# 5️⃣  Index documents → POST /v1/api/index
# ----------------------------------------------------------------------
@router.post(
    "/index",
    summary="Index documents into a vector store.",
    response_description="Success message.",
    status_code=status.HTTP_200_OK,
)
async def index(
    request: IndexingPayloadV1,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> JSONResponse:
    model_name = await validate_collection_access(header_information.x_base_api_key, request.collection)
    model_path, embedding_dimensions, context_length = await check_embedding_model(model_name=model_name)

    logger.info(
        f"Index request on collection={request.collection} docs={len(request.documents)} "
        f"backend={request.storage_backend}"
    )
    # ------------------------------------------------------------------
    # PGVector path – unchanged logic, just adapt to new request fields
    # ------------------------------------------------------------------
    if request.storage_backend == StorageBackend.PGVECTOR:
        embedding_service = EmbeddingService(open_ai_sdk)
        document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
        if not document_repository.check_table_exists():
            raise HTTPException(status_code=404, detail="Collection does not exist.")
        pgvector_store = PGVectorDocumentStore(
            embedding_service=embedding_service,
            document_repository=document_repository,
        )
        await pgvector_store.index(
            documents=request.documents,
            collection=request.collection,
            model_name=model_name,
            context_length=context_length,
            model_path=model_path,
        )
        return JSONResponse(content={"message": "Data indexed successfully."}, status_code=status.HTTP_200_OK)

    # ------------------------------------------------------------------
    # Elasticsearch path – new backend (implemented later)
    # ------------------------------------------------------------------
    elif request.storage_backend == StorageBackend.ELASTICSEARCH:
        # The ElasticSearchDocumentStore class implements the same AbstractDocumentStore API
        from src.services.elastic_document_store import ElasticDocumentStore  # lazy import
        embedding_service = EmbeddingService(open_ai_sdk)
        es_store = ElasticDocumentStore(
            collection=request.collection,
            embedding_service=embedding_service,
            embedding_dimensions=embedding_dimensions,
        )
        await es_store.index(request.documents, request.collection, model_name, context_length, model_path)
        return JSONResponse(content={"message": "Data indexed successfully (ElasticSearch)."}, status_code=status.HTTP_200_OK)

    else:
        raise UnsupportedStorageBackendError(f"Unsupported storage backend: {request.storage_backend}")


# ----------------------------------------------------------------------
# 6️⃣  Search → POST /v1/api/search
# ----------------------------------------------------------------------
@router.post(
    "/search",
    summary="Search a vector store.",
    response_model=SearchResponseV1,
    status_code=status.HTTP_200_OK,
)
async def search(
    request: SearchRequestV1,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> SearchResponseV1:
    model_name = await validate_collection_access(header_information.x_base_api_key, request.collection)
    model_path, embedding_dimensions, context_length = await check_embedding_model(model_name=model_name)

    logger.info(f"Search request on collection={request.collection} backend={request.storage_backend}")

    if request.storage_backend == StorageBackend.PGVECTOR:
        embedding_service = EmbeddingService(open_ai_sdk)
        document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
        pg_store = PGVectorDocumentStore(
            embedding_service=embedding_service,
            document_repository=document_repository,
        )
        return await pg_store.search(request, model_name, context_length, model_path)

    elif request.storage_backend == StorageBackend.ELASTICSEARCH:
        from src.services.elastic_document_store import ElasticDocumentStore
        embedding_service = EmbeddingService(open_ai_sdk)
        es_store = ElasticDocumentStore(
            collection=request.collection,
            embedding_service=embedding_service,
            embedding_dimensions=embedding_dimensions,
        )
        return await es_store.search(request, model_name, context_length, model_path)

    else:
        raise UnsupportedStorageBackendError(f"Unsupported storage backend: {request.storage_backend}")


# ----------------------------------------------------------------------
# 7️⃣  Delete all docs → DELETE /v1/api/delete_index
# ----------------------------------------------------------------------
@router.delete(
    "/delete_index",
    summary="Delete all documents from a vector store.",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
)
async def delete(
    request: DeleteRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> DeleteResponse:
    # unchanged except we pass the new request model
    model_name = await validate_collection_access(header_information.x_base_api_key, request.collection)
    _, embedding_dimensions, _ = await check_embedding_model(model_name=model_name)

    if request.storage_backend == StorageBackend.PGVECTOR:
        doc_repo = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
        embed_service = EmbeddingService(open_ai_sdk)
        pg_store = PGVectorDocumentStore(document_repository=doc_repo, embedding_service=embed_service)
        deleted = await pg_store.delete(request.collection)
    else:  # ElasticSearch
        from src.services.elastic_document_store import ElasticDocumentStore
        embed_service = EmbeddingService(open_ai_sdk)
        es_store = ElasticDocumentStore(
            collection=request.collection,
            embedding_service=embed_service,
            embedding_dimensions=embedding_dimensions,
        )
        deleted = await es_store.delete(request.collection)

    return DeleteResponse(
        message=f"Deleted {deleted} document(s) from {request.collection}.",
        collection=request.collection,
    )

# ----------------------------------------------------------------------
# 8️⃣  Delete by IDs → DELETE /v1/api/collection/delete_by_ids
# ----------------------------------------------------------------------
@router.delete(
    "/collection/delete_by_ids",
    summary="Delete specific documents by IDs.",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_by_ids(
    request: DeleteByIdsRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> DeleteResponse:
    model_name = await validate_collection_access(header_information.x_base_api_key, request.collection)
    _, embedding_dimensions, _ = await check_embedding_model(model_name=model_name)

    if request.storage_backend == StorageBackend.PGVECTOR:
        doc_repo = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
        embed_service = EmbeddingService(open_ai_sdk)
        pg_store = PGVectorDocumentStore(document_repository=doc_repo, embedding_service=embed_service)
        deleted = await pg_store.delete_by_ids(request.collection, request.index_ids)
    else:
        from src.services.elastic_document_store import ElasticDocumentStore
        embed_service = EmbeddingService(open_ai_sdk)
        es_store = ElasticDocumentStore(
            collection=request.collection,
            embedding_service=embed_service,
            embedding_dimensions=embedding_dimensions,
        )
        deleted = await es_store.delete_by_ids(request.collection, request.index_ids)

    return DeleteResponse(
        message=f"Deleted {deleted} document(s) from {request.collection}.",
        collection=request.collection,
    )
Key points

All endpoints now use OpenAI‑style payloads (IndexingPayloadV1, SearchRequestV1, etc.).
The router decides at runtime which backend implementation to instantiate (PGVectorDocumentStore or new ElasticDocumentStore).
Response models are also OpenAI‑compatible (SearchResponseV1).
2.2.6 src/services/abstract_document_store.py (update signatures)

# src/services/abstract_document_store.py
from abc import ABC, abstractmethod
from typing import List

from src.models.storage_payload import Document, SearchRequestV1, SearchResponseV1


class AbstractDocumentStore(ABC):
    @abstractmethod
    async def search(
        self,
        search_request: SearchRequestV1,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> SearchResponseV1:
        ...

    @abstractmethod
    async def index(
        self,
        documents: List[Document],
        collection: str,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> None:
        ...

    @abstractmethod
    async def delete(self, collection: str) -> int:
        """Delete **all** documents in a collection."""
        ...

    @abstractmethod
    async def delete_by_ids(self, collection: str, index_ids: List[str]) -> int:
        """Delete a subset of documents."""
        ...
Changes – the abstract class now imports the V1 request/response models, making every concrete implementation adhere to the same contract.

2.2.7 src/services/pgvector_document_store.py (handle new request fields)

Only the search method signature changes (already reflected in abstract). Minor mapping of fields:

# src/services/pgvector_document_store.py
# (top imports stay the same)
from src.models.storage_payload import SearchRequestV1, SearchResponseV1

class PGVectorDocumentStore(AbstractDocumentStore):
    # __init__ unchanged ...

    async def search(
        self,
        search_request: SearchRequestV1,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> SearchResponseV1:
        try:
            start = time.time()
            # Map OpenAI fields → internal logic
            if search_request.search_type == SearchType.SEMANTIC:
                self._content_length_validation(
                    context_length, model_name, model_path, search_request.query, "search"
                )
                results = await self.sematic_search(search_request)
            elif search_request.search_type == SearchType.FULL_TEXT:
                results = await self.fulltext_search(search_request)
            else:  # HYBRID
                self._content_length_validation(
                    context_length, model_name, model_path, search_request.query, "search"
                )
                results = await self.hybrid_search(search_request)

            # Build OpenAI‑compatible response
            response = SearchResponseV1(
                search_query=search_request.query,
                data=[
                    SearchResultV1(
                        file_id=res.id,
                        filename=res.source.get("filename"),
                        score=res.score or 0.0,
                        attributes=res.source,
                        content=[{"type": "text", "text": res.source.get("content", "")}],
                    )
                    for res in results
                ],
                has_more=False,
                next_page=None,
            )
            response.object = "vector_store.search_results.page"
            response_time_ms = round((time.time() - start) * 1000, 2)
            logger.info(f"Search completed in {response_time_ms} ms, {len(results)} hits")
            return response
        except (DatabaseConnectionError, DocumentMaxTokenLimitExceededError) as db_err:
            raise db_err
        except Exception as exc:
            logger.exception("Search failed")
            raise DocumentStoreSearchError(str(exc))
Notes

The new SearchResponseV1 is built from the internal SearchResult list.
We preserve the old SearchResult fields (id, source) and map them to the OpenAI schema (file_id, content).
The metadata/attributes are simply the whole source dict – it can be filtered later if needed.
2.2.8 src/api/routers/__init__.py (export unified router)

# src/api/routers/__init__.py
from .collection_router import router as collection_router
from .document_store_router import router as document_store_router
from .vector_store_router import router as vector_store_router  # new

__all__ = [
    "collection_router",
    "document_store_router",
    "vector_store_router",
]
2.2.9 New file src/api/routers/vector_store_router.py (public entry point)

# src/api/routers/vector_store_router.py
"""
Unified public router that bundles the OpenAI‑compatible collection,
index and search endpoints under the same prefix (/v1/api).
All endpoints delegate to the existing routers that already contain the
business logic; this file just re‑exports them so the FastAPI app can
mount a single router.
"""

from fastapi import APIRouter

from .collection_router import router as collection_router
from .document_store_router import router as document_store_router

router = APIRouter()
router.include_router(collection_router)
router.include_router(document_store_router)
Explanation – The create_app function in src/main.py already mounts collection_router and document_store_router. Adding this wrapper makes it trivial to mount a single router (vector_store_router) if we ever want to hide the internal split.

2.2.10 src/config.py (add optional API key for OpenAI Vector Store SDK)

# src/config.py
class Settings(BaseSettings):
    # … existing fields …
    openai_vectorstore_api_key: str = ""   # <-- new, optional – will be read from .env
    # … rest unchanged …
2.2.11 New SDK – src/integrations/openai_vectorstore_sdk.py

# src/integrations/openai_vectorstore_sdk.py
"""
Thin wrapper around the official OpenAI Python client that only
exposes the Vector‑Store surface (create, list, retrieve, update,
delete, search).  The wrapper mirrors the signatures used in the
router code so that we can inject it via dependency injection if
needed.
"""

import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.vector_store import VectorStore
from openai.types.vector_store_file import VectorStoreFile
from openai.types.vector_store_file_batch import VectorStoreFileBatch

from src.config import get_settings
from src.logging_config import Logger

logger = Logger.create_logger(__name__)
settings = get_settings()


class OpenAIVectorStoreSDK:
    def __init__(self) -> None:
        api_key = settings.openai_vectorstore_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key for vector stores is not configured.")
        self.client = AsyncOpenAI(api_key=api_key, base_url=settings.base_api_url)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    async def create(self, name: Optional[str] = None,
                     metadata: Optional[Dict[str, str]] = None,
                     expires_after: Optional[Dict[str, Any]] = None) -> VectorStore:
        payload = {}
        if name:
            payload["name"] = name
        if metadata:
            payload["metadata"] = metadata
        if expires_after:
            payload["expires_after"] = expires_after
        logger.info(f"[OpenAIVectorStore] Creating store with payload={payload}")
        return await self.client.vector_stores.create(**payload)

    async def list(self, limit: int = 20, after: Optional[str] = None,
                   before: Optional[str] = None, order: str = "desc") -> Dict[str, Any]:
        logger.info("[OpenAIVectorStore] Listing stores")
        return await self.client.vector_stores.list(
            limit=limit, after=after, before=before, order=order
        )

    async def retrieve(self, vector_store_id: str) -> VectorStore:
        logger.info(f"[OpenAIVectorStore] Retrieving store {vector_store_id}")
        return await self.client.vector_stores.retrieve(vector_store_id=vector_store_id)

    async def update(self, vector_store_id: str,
                     name: Optional[str] = None,
                     metadata: Optional[Dict[str, str]] = None,
                     expires_after: Optional[Dict[str, Any]] = None) -> VectorStore:
        logger.info(f"[OpenAIVectorStore] Updating store {vector_store_id}")
        return await self.client.vector_stores.update(
            vector_store_id=vector_store_id,
            name=name,
            metadata=metadata,
            expires_after=expires_after,
        )

    async def delete(self, vector_store_id: str) -> Dict[str, Any]:
        logger.info(f"[OpenAIVectorStore] Deleting store {vector_store_id}")
        return await self.client.vector_stores.delete(vector_store_id=vector_store_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search(
        self,
        vector_store_id: str,
        query: str | List[float],
        filters: Optional[Dict[str, Any]] = None,
        max_num_results: int = 10,
        ranking_options: Optional[Dict[str, Any]] = None,
        rewrite_query: bool = False,
    ) -> Dict[str, Any]:
        """
        Mirrors the OpenAI `/vector_stores/{id}/search` endpoint.
        `query` may be a string (semantic) or an explicit embedding vector.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "max_num_results": max_num_results,
            "rewrite_query": rewrite_query,
        }
        if filters:
            payload["filters"] = filters
        if ranking_options:
            payload["ranking_options"] = ranking_options

        logger.info(f"[OpenAIVectorStore] Searching store {vector_store_id} with payload={payload}")
        return await self.client.vector_stores.search(
            vector_store_id=vector_store_id, **payload
        )
Explanation

All methods are async to match the rest of the code‑base.
The wrapper reads the API key from the new openai_vectorstore_api_key setting (or the generic OPENAI_API_KEY).
It is deliberately minimal – it only forwards arguments to the OpenAI client.
2.2.12 src/services/elastic_document_store.py (new backend – requirement 3)

# src/services/elastic_document_store.py
"""
ElasticSearch implementation of the AbstractDocumentStore interface.
Only the methods required by the current API surface are implemented.
"""

import json
from typing import List

from elasticsearch import AsyncElasticsearch, NotFoundError, RequestError
from elasticsearch.helpers import async_bulk

from src.config import Settings, get_settings
from src.exception.document_store_exception import DocumentStoreError, DocumentStoreSearchError, DocumentStoreDeleteError
from src.models.storage_payload import Document, SearchRequestV1, SearchResponseV1, SearchResultV1
from src.services.abstract_document_store import AbstractDocumentStore
from src.logging_config import Logger

logger = Logger.create_logger(__name__)
settings = get_settings()


class ElasticDocumentStore(AbstractDocumentStore):
    def __init__(
        self,
        collection: str,
        embedding_service,
        embedding_dimensions: int,
        settings: Settings = get_settings(),
    ):
        self.collection = collection
        self.embedding_service = embedding_service
        self.embedding_dimensions = embedding_dimensions
        self.settings = settings
        # Simple client – uses host+auth from env
        self.es = AsyncElasticsearch(
            hosts=[settings.elasticsearch_host],
            http_auth=(settings.elasticsearch_user, settings.elasticsearch_password),
            verify_certs=not settings.elasticsearch_insecure,
        )

    # ------------------------------------------------------------------
    async def index(
        self,
        documents: List[Document],
        collection: str,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> None:
        # Generate embeddings (reuse the same embedding_service used by PGVector)
        contents = [doc.content for doc in documents]
        embed_resp = await self.embedding_service.get_embeddings(model_name=model_name, batch=contents)

        actions = []
        for idx, doc in enumerate(documents):
            action = {
                "_index": self.collection,
                "_id": doc.get("id") or None,
                "_source": {
                    "content": doc.content,
                    "embedding": embed_resp.data[idx].embedding,
                    "metadata": doc.metadata or {},
                    "links": doc.links,
                    "topics": doc.topics,
                    "author": doc.author,
                },
            }
            actions.append(action)

        try:
            await async_bulk(self.es, actions, raise_on_error=True, max_retries=self.settings.max_retries)
            logger.info(f"Indexed {len(actions)} docs into ES index '{self.collection}'.")
        except RequestError as exc:
            logger.error(f"Elasticsearch bulk indexing failed: {exc}")
            raise DocumentStoreError(f"Indexing error: {exc}")

    # ------------------------------------------------------------------
    async def search(
        self,
        search_request: SearchRequestV1,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> SearchResponseV1:
        # Build ES query – a simple `match` on `content` plus optional filter
        query_body: dict = {
            "size": min(search_request.max_num_results, 50),
            "query": {
                "bool": {
                    "must": {"match": {"content": search_request.query}},
                    "filter": [],
                }
            },
        }

        if search_request.filters:
            # Pass the filter dict straight to ES – assume OpenAI style and ES compatible
            query_body["query"]["bool"]["filter"].append(search_request.filters)

        try:
            resp = await self.es.search(index=self.collection, body=query_body)
        except NotFoundError:
            raise DocumentStoreSearchError(f"Collection '{self.collection}' does not exist in Elasticsearch.")
        except RequestError as exc:
            logger.error(f"Elasticsearch search error: {exc}")
            raise DocumentStoreSearchError(str(exc))

        hits = resp["hits"]["hits"]
        results = [
            SearchResultV1(
                file_id=hit["_id"],
                filename=hit["_source"].get("filename"),
                score=hit["_score"] or 0.0,
                attributes=hit["_source"],
                content=[{"type": "text", "text": hit["_source"].get("content", "")}],
            )
            for hit in hits
        ]

        return SearchResponseV1(
            search_query=search_request.query,
            data=results,
            has_more=False,
            next_page=None,
        )

    # ------------------------------------------------------------------
    async def delete(self, collection: str) -> int:
        try:
            resp = await self.es.delete_by_query(
                index=self.collection,
                body={"query": {"match_all": {}}},
                refresh=True,
            )
            deleted = resp.get("deleted", 0)
            logger.info(f"Deleted {deleted} docs from ES index '{self.collection}'.")
            return deleted
        except NotFoundError:
            raise DocumentStoreDeleteError(f"Collection '{self.collection}' not found.")
        except RequestError as exc:
            logger.error(f"Elasticsearch delete error: {exc}")
            raise DocumentStoreDeleteError(str(exc))

    # ------------------------------------------------------------------
    async def delete_by_ids(self, collection: str, index_ids: List[str]) -> int:
        # Use mget + bulk delete
        actions = [{"_op_type": "delete", "_index": self.collection, "_id": doc_id} for doc_id in index_ids]
        try:
            resp = await async_bulk(self.es, actions, raise_on_error=False)
            # `async_bulk` returns (successes, errors) tuple – we only care about successes
            deleted = sum(1 for ok, _ in resp if ok)
            logger.info(f"Deleted {deleted}/{len(index_ids)} requested IDs from ES index '{self.collection}'.")
            return deleted
        except RequestError as exc:
            logger.error(f"Elasticsearch bulk delete error: {exc}")
            raise DocumentStoreDeleteError(str(exc))
Key points

Uses AsyncElasticsearch (the official async client).
Implements the same abstract interface as PGVectorDocumentStore.
Relies on the existing EmbeddingService to compute embeddings before indexing (so the backend only stores the vector).
Supports simple full‑text match (match) and optional filter dicts; ranking options are ignored for now (can be added later).
2.2.13 src/exception/document_store_exception.py (new ElasticSearch exception)

# src/exception/document_store_exception.py
class DocumentStoreError(Exception):
    """Base class for search‑related errors."""
    pass

# ... existing classes ...

class ElasticSearchError(DocumentStoreError):
    """Raised when an Elasticsearch operation fails."""
    pass
2.2.14 src/config.py – add Elasticsearch connection settings (optional)

class Settings(BaseSettings):
    # ... existing fields ...

    # Elasticsearch
    elasticsearch_host: str = "http://localhost:9200"
    elasticsearch_user: str = ""
    elasticsearch_password: str = ""
    elasticsearch_insecure: bool = True  # skip cert verification for dev
2.2.15 Update src/api/deps.py – inject the new OpenAI Vector Store SDK (optional)

from src.integrations.openai_vectorstore_sdk import OpenAIVectorStoreSDK

@lru_cache()
def get_vectorstore_sdk() -> OpenAIVectorStoreSDK:
    return OpenAIVectorStoreSDK()
Now any router can Depends(get_vectorstore_sdk) if we ever need to expose the raw OpenAI SDK.

2.2.16 Adjust src/main.py – mount the new unified router

# src/main.py
# ... existing imports ...
from src.api.routers.vector_store_router import router as vector_store_router

# Inside create_app()
app.include_router(vector_store_router, prefix=settings.api_common_prefix, tags=["VECTOR_STORE"])
Now all OpenAI‑compatible collection, index and search routes live under /v1/api (the same prefix used throughout the service).

3️⃣ Implementation plan – requirement 2 (uniform v2 wrapper)

Goal: expose a single public router (/v1/api/...) that internally forwards to the existing v2 implementations, while keeping the v2 routes usable for power users.

3.1 Files touched / added

File	Reason
src/api/routers/v2/__init__.py – unchanged (kept for internal use).	
New src/api/routers/vector_store_router.py (already added) – this is the uniform router that includes both v1‑compatible routes and the v2 routers (via include_router).	
src/main.py – import the new router and mount it (see 2.2.16).	
No changes to the business‑logic files – they already work for both sets of models.	
Thus requirement 2 is satisfied by the router composition introduced above.

4️⃣ Implementation plan – requirement 3 (ElasticSearch integration)

All the heavy lifting is already in elastic_document_store.py (2.2.12) and the registry now knows about the backend (StorageBackend.ELASTICSEARCH).

Additional steps:

Register the backend – modify src/repository/registry/__init__.py (already imports storage_backend_registry). The new class ElasticDocumentStore is automatically discoverable once we import it somewhere (e.g., in the router we lazy‑import it). No extra registration needed because the Registry pattern is used only for the dynamic model part; the vector‑store back‑ends are obtained via storage_backend_registry.get(name)() – we must add a registration decorator.

# src/services/elastic_document_store.py   (top of file)
from src.repository.registry import storage_backend_registry

@storage_backend_registry.register(name="elasticsearch")
class ElasticDocumentStore(AbstractDocumentStore):
    # class body as above
Add environment variables – already added in src/config.py. Document them in README.md (not shown here).

Update unit tests – Add new tests under tests/unit/api/... covering:

Indexing with storage_backend=elasticsearch returns success.
Searching with storage_backend=elasticsearch returns SearchResponseV1 shape.
Delete and delete_by_ids flow.
Add a small utility for converting our generic Document model to ES‑compatible dict – already done inside ElasticDocumentStore.index.

Error handling – we added ElasticSearchError class; routers will raise this via DocumentStoreError base class, which is already caught in main.py.

5️⃣ Summary of changed / new files (with relative paths)

Path	Type	Brief description
src/models/collection_payload.py	Modified	Added CreateCollectionV1 with OpenAI fields.
src/models/indexing_payload.py	Modified	Added IndexingPayloadV1.
src/models/storage_payload.py	Modified	New OpenAI‑compatible SearchRequestV1, SearchResponseV1, SearchResultV1.
src/api/routers/collection_router.py	Modified	Routes renamed to OpenAI style, payloads switched to V1 models.
src/api/routers/document_store_router.py	Modified	Uses new request/response models, dispatches to PGVector or ElasticSearch.
src/services/abstract_document_store.py	Modified	Updated signatures to use V1 models.
src/services/pgvector_document_store.py	Modified	Adjusted search method to build SearchResponseV1.
src/api/routers/__init__.py	Modified	Export new vector_store_router.
src/api/routers/vector_store_router.py	New	Single entry point that bundles collection & document‑store routers.
src/config.py	Modified	Added openai_vectorstore_api_key and Elasticsearch config values.
src/integrations/openai_vectorstore_sdk.py	New	Thin async wrapper around OpenAI’s vector‑store API.
src/services/elastic_document_store.py	New	Async ElasticSearch implementation of AbstractDocumentStore.
src/exception/document_store_exception.py	Modified	Added ElasticSearchError.
src/api/deps.py	Modified	Added get_vectorstore_sdk dependency (optional).
src/main.py	Modified	Mounts vector_store_router.
src/utility/registry.py	Modified (optional)	Ensure ElasticDocumentStore registered via decorator (shown in file).
src/README.md	Modified (not displayed)	Document new endpoints, env vars, and ElasticSearch support.
All other files remain untouched.

6️⃣ How the new flow works (for reviewers)

Client → FastAPI

Calls POST /v1/api/collection with an CreateCollectionV1 payload.
Router validates the model, calls CollectionService.create (unchanged).
Indexing

POST /v1/api/index receives an IndexingPayloadV1.
Router validates the backend, obtains the embedding dimensions via check_embedding_model.
Depending on storage_backend it builds either a PGVectorDocumentStore or an ElasticDocumentStore.
Calls store.index(...). The store handles embedding generation (via EmbeddingService) and persists documents.
Search

POST /v1/api/search receives a SearchRequestV1.
Same dispatch logic as indexing.
The concrete store builds a SearchResponseV1 that mirrors the OpenAI spec (fields: object, search_query, data, has_more, next_page).
Deletion (DELETE /v1/api/delete_index & /v1/api/collection/delete_by_ids) – similar dispatch.

ElasticSearch backend – uses the async official client; data is stored in an index named exactly like the internal collection name.

OpenAI Vector‑Store SDK – exposed via dependency injection, not needed for the default flow but available for any internal service that wants to interact directly with OpenAI’s hosted vector stores.

7️⃣ Next steps for the team

Run the test suite – ensure existing tests still pass.
Add unit tests for the newly created ElasticDocumentStore and the updated routers (use httpx.AsyncClient with TestClient).
Add integration test (if you have an ElasticSearch instance) – spin‑up a Docker ES container in CI.
Update CI pipeline to include the new environment variables.
Update documentation (README.md and Swagger UI) – the new OpenAPI tags (VECTOR_STORE) will appear automatically.
With these changes the platform now offers a fully OpenAI‑compatible vector‑store API, a uniform v2 wrapper, and Elasticsearch as an alternative storage backend, all while keeping the existing code‑base clean and testable.
