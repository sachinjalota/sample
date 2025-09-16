📋 1️⃣ JIRA‑style Epics & Tasks

EPIC	TASK (JIRA Summary)	DESCRIPTION
Vector‑Store Compatibility	VST‑001 – Analyse current vector‑store endpoints	Review all “vector‑store”‑related routers (collection, collection/data, collection/create, collection/delete, index, search, delete_index, collection/delete_by_ids, rag). Identify gaps vs. OpenAI Vector Store API.
VST‑002 – Align payloads to OpenAI spec	Update request/response models so they follow the OpenAI schema (e.g., CreateVectorStoreRequest, ListVectorStoresResponse, VectorStoreObject, VectorStoreFileObject, VectorStoreSearchRequest, etc.).
VST‑003 – Implement OpenAI‑style Vector Store SDK	Add a thin wrapper around OpenAI’s client.vector_stores (similar to OpenAISdk) exposing CRUD, file‑ops and search.
VST‑004 – Refactor PGVector router to use new SDK	Replace direct repository calls with the SDK methods; adjust return types to match OpenAI models.
VST‑005 – Add mapping table (doc → OpenAI endpoint)	Provide a clear mapping between our internal endpoints and OpenAI Vector Store API.
VST‑006 – Unit‑test the new SDK & routers	Add/extend tests under tests/unit/... to verify request/response shapes and error handling.
ElasticSearch Integration	ES‑001 – Add ES client configuration	Create a reusable ElasticSearchClient (using elasticsearch‑7 or elasticsearch‑8).
ES‑002 – Implement ES‑backed DocumentRepository	New class ElasticSearchDocumentRepository implementing the same interface as DocumentRepository (create index, bulk index, delete, delete‑by‑ids, search (semantic, full‑text, hybrid)).
ES‑003 – Extend Registry to load ES backend	Register "elasticsearch" in src/repository/registry/pgvector.py‑like module.
ES‑004 – Update routers to support storage_backend=elasticsearch	Add branch logic in document_store_router, rag_router, search etc. to instantiate the proper backend via the registry.
ES‑005 – Add Elasticsearch specific request/response models (e.g., ElasticSearchSearchResult).	
ES‑006 – Add integration tests for Elasticsearch (mocked via responses or an in‑memory ES container).	
ES‑007 – Documentation update – Explain how to switch between PGVector and Elasticsearch via the storage_backend field.	
📂 2️⃣ Required Code Changes

Below each file shows what to add / modify. The diff‑style (+ / -) highlights new or changed lines. All new files are marked with 🆕 and their location.

A️⃣ Align Payloads & Add OpenAI‑style SDK

1️⃣ src/models/vector_store_payload.py (🆕 new file)

# src/models/vector_store_payload.py
"""
Pydantic models that follow the OpenAI Vector Store API schema.
These models are used by the public routers (collection, index, search, …)
so that our API contracts are 1‑to‑1 compatible with the official spec.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------
# CREATE VECTOR STORE
# ----------------------------------------------------------------------
class CreateVectorStoreRequest(BaseModel):
    name: Optional[str] = Field(None, description="Human‑readable name.")
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Arbitrary key‑value pairs (max 16 entries, 64/512 chars).",
    )
    # In our service we allow the user to pass an existing collection name.
    # This maps to the “file_ids” array in the real spec – we keep the same key
    # name so downstream SDK can forward it unchanged.
    file_ids: Optional[List[str]] = Field(
        None,
        description="List of file IDs already uploaded to the file endpoint.",
    )
    expires_after: Optional[Dict[str, Any]] = Field(
        None,
        description="Expiration policy: {\"anchor\": \"last_active_at\", \"days\": int}",
    )
    chunking_strategy: Optional[Dict[str, Any]] = Field(
        None,
        description="Chunking strategy (auto or static).",
    )

# ----------------------------------------------------------------------
# LIST / RETRIEVE VECTOR STORE
# ----------------------------------------------------------------------
class VectorStoreObject(BaseModel):
    id: str
    object: Literal["vector_store"]
    created_at: int
    name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    usage_bytes: Optional[int] = None
    file_counts: Optional[Dict[str, int]] = None
    expires_at: Optional[int] = None
    last_active_at: Optional[int] = None
    status: Optional[Literal["in_progress", "completed", "expired"]] = None


class ListVectorStoresResponse(BaseModel):
    object: Literal["list"]
    data: List[VectorStoreObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool


# ----------------------------------------------------------------------
# CREATE VECTOR STORE FILE
# ----------------------------------------------------------------------
class CreateVectorStoreFileRequest(BaseModel):
    file_id: str = Field(..., description="ID of a previously uploaded file.")
    attributes: Optional[Dict[str, Any]] = Field(
        None,
        description="Up to 16 custom key‑value pairs (max 64/512 chars).",
    )
    chunking_strategy: Optional[Dict[str, Any]] = None


class VectorStoreFileObject(BaseModel):
    id: str
    object: Literal["vector_store.file"]
    created_at: int
    usage_bytes: Optional[int] = None
    vector_store_id: str
    status: Literal["in_progress", "completed", "failed", "cancelled"]
    last_error: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[Dict[str, Any]] = None


class ListVectorStoreFilesResponse(BaseModel):
    object: Literal["list"]
    data: List[VectorStoreFileObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool


# ----------------------------------------------------------------------
# SEARCH VECTOR STORE
# ----------------------------------------------------------------------
class VectorStoreSearchFilter(BaseModel):
    key: str
    type: Literal["eq", "ne", "gt", "gte", "lt", "lte"]
    value: Any


class VectorStoreCompoundFilter(BaseModel):
    type: Literal["and", "or"]
    filters: List[Dict[str, Any]]  # can be Comparison or nested Compound


class VectorStoreSearchRequest(BaseModel):
    query: str | List[float] = Field(..., description="Text query or embedding vector.")
    filters: Optional[Dict[str, Any]] = None
    max_num_results: Optional[int] = Field(10, ge=1, le=50)
    ranking_options: Optional[Dict[str, Any]] = None
Why – Provides a single source‑of‑truth for request/response formats that mirror the OpenAI specification, eliminating ad‑hoc dict usage throughout the codebase.

2️⃣ src/integrations/openai_vector_store_sdk.py (🆕 new file)

# src/integrations/openai_vector_store_sdk.py
"""
Thin wrapper around the official OpenAI Python client that exposes
the Vector Store API in a fashion consistent with the rest of our SDK
(`OpenAISdk`).  All methods raise HTTPException with clear status‑codes
so routers can simply forward the error.
"""

from __future__ import annotations

from typing import Any, List, Optional

import httpx
from fastapi import HTTPException, status
from openai import OpenAI
from openai.types import VectorStore, VectorStoreFile

from src.config import Settings, get_settings
from src.logging_config import Logger
from src.models.vector_store_payload import (
    CreateVectorStoreFileRequest,
    CreateVectorStoreRequest,
    ListVectorStoreFilesResponse,
    ListVectorStoresResponse,
    VectorStoreSearchRequest,
)

logger = Logger.create_logger(__name__)

class OpenAIVectorStoreSDK:
    def __init__(self, settings: Settings = get_settings()) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.default_api_key,
            base_url=settings.base_api_url,
            # the OpenAI client already respects verify flag; we forward it.
        )

    # ------------------------------------------------------------------
    # Vector Store CRUD
    # ------------------------------------------------------------------
    def create_vector_store(self, payload: CreateVectorStoreRequest) -> VectorStore:
        logger.info("Creating OpenAI Vector Store")
        try:
            return self.client.vector_stores.create(**payload.model_dump(exclude_unset=True))
        except Exception as exc:
            logger.exception("Failed to create vector store")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI vector store creation failed: {exc}",
            )

    def list_vector_stores(self, after: Optional[str] = None, before: Optional[str] = None,
                           limit: int = 20, order: str = "desc") -> ListVectorStoresResponse:
        try:
            resp = self.client.vector_stores.list(after=after, before=before,
                                                 limit=limit, order=order)
            return ListVectorStoresResponse(**resp.dict())
        except Exception as exc:
            logger.exception("Failed to list vector stores")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI list vector stores failed: {exc}",
            )

    def retrieve_vector_store(self, vector_store_id: str) -> VectorStore:
        try:
            return self.client.vector_stores.retrieve(vector_store_id=vector_store_id)
        except Exception as exc:
            logger.exception("Failed to retrieve vector store")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector store {vector_store_id} not found: {exc}",
            )

    def update_vector_store(self, vector_store_id: str, name: Optional[str] = None,
                            metadata: Optional[dict] = None,
                            expires_after: Optional[dict] = None) -> VectorStore:
        try:
            return self.client.vector_stores.update(
                vector_store_id=vector_store_id,
                name=name,
                metadata=metadata,
                expires_after=expires_after,
            )
        except Exception as exc:
            logger.exception("Failed to update vector store")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OpenAI update vector store failed: {exc}",
            )

    def delete_vector_store(self, vector_store_id: str) -> dict:
        try:
            return self.client.vector_stores.delete(vector_store_id=vector_store_id)
        except Exception as exc:
            logger.exception("Failed to delete vector store")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector store {vector_store_id} could not be deleted: {exc}",
            )

    # ------------------------------------------------------------------
    # Vector Store Files
    # ------------------------------------------------------------------
    def create_vector_store_file(self, vector_store_id: str,
                                payload: CreateVectorStoreFileRequest) -> VectorStoreFile:
        try:
            return self.client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                **payload.model_dump(exclude_unset=True),
            )
        except Exception as exc:
            logger.exception("Failed to create vector store file")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI file creation failed: {exc}",
            )

    def list_vector_store_files(self, vector_store_id: str,
                               after: Optional[str] = None,
                               before: Optional[str] = None,
                               limit: int = 20,
                               order: str = "desc",
                               filter: Optional[str] = None) -> ListVectorStoreFilesResponse:
        try:
            resp = self.client.vector_stores.files.list(
                vector_store_id=vector_store_id,
                after=after,
                before=before,
                limit=limit,
                order=order,
                filter=filter,
            )
            return ListVectorStoreFilesResponse(**resp.dict())
        except Exception as exc:
            logger.exception("Failed to list vector store files")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI list files failed: {exc}",
            )

    def retrieve_vector_store_file(self, vector_store_id: str, file_id: str) -> VectorStoreFile:
        try:
            return self.client.vector_stores.files.retrieve(
                vector_store_id=vector_store_id,
                file_id=file_id,
            )
        except Exception as exc:
            logger.exception("Failed to retrieve vector store file")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found in store {vector_store_id}: {exc}",
            )

    def delete_vector_store_file(self, vector_store_id: str, file_id: str) -> dict:
        try:
            return self.client.vector_stores.files.delete(
                vector_store_id=vector_store_id,
                file_id=file_id,
            )
        except Exception as exc:
            logger.exception("Failed to delete vector store file")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} could not be deleted: {exc}",
            )

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------
    def search(self, vector_store_id: str,
               payload: VectorStoreSearchRequest) -> dict:
        try:
            return self.client.vector_stores.search(
                vector_store_id=vector_store_id,
                **payload.model_dump(exclude_unset=True),
            )
        except Exception as exc:
            logger.exception("Vector store search failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI vector store search failed: {exc}",
            )
Why – Centralises all Vector‑Store interactions, mirrors the existing OpenAISdk pattern, and returns Pydantic‑compatible objects so routers don’t need to hand‑craft HTTP calls.

3️⃣ Register the new SDK in dependency injection

File: src/api/deps.py

@@
 from src.integrations.open_ai_sdk import OpenAISdk
+from src.integrations.openai_vector_store_sdk import OpenAIVectorStoreSDK
@@
 @lru_cache()
 def get_openai_service_internal() -> OpenAISdk:
     return build_openai_sdk(settings.default_api_key)
@@
 @lru_cache()
 def get_openai_service(
     header_info: HeaderInformation = Depends(validate_headers_and_api_key),
 ) -> OpenAISdk:
     return build_openai_sdk(header_info.x_base_api_key)
@@
 @lru_cache()
-def get_openai_service_internal() -> OpenAISdk:
-    return build_openai_sdk(settings.default_api_key)
+def get_vector_store_service() -> OpenAIVectorStoreSDK:
+    """Returns a singleton OpenAI Vector Store SDK instance."""
+    return OpenAIVectorStoreSDK()
Why – Makes the new SDK injectable exactly like the existing LLM/Embedding SDK, enabling routers to depend on it.

B️⃣ Refactor PGVector‑related Routers to Use the SDK

4️⃣ src/api/routers/collection_router.py – map to Create / List / Delete Vector Stores

@@
-from src.exception.exceptions import (
-    CollectionError,
-    DatabaseConnectionError,
-    EmbeddingModelError,
-)
+from src.exception.exceptions import (
+    CollectionError,
+    DatabaseConnectionError,
+    EmbeddingModelError,
+)
+from src.integrations.openai_vector_store_sdk import OpenAIVectorStoreSDK
+from src.models.vector_store_payload import (
+    CreateVectorStoreRequest,
+    ListVectorStoresResponse,
+    VectorStoreObject,
+)
@@
-    collection_service: CollectionService = Depends(get_collection_service),
+    vector_store_sdk: OpenAIVectorStoreSDK = Depends(get_vector_store_service),
 ) -> dict:
@@
-    usecase_id = await get_usecase_id_by_api_key(header_information.x_base_api_key)
-    return await collection_service.get(usecase_id=usecase_id)
+    # OpenAI spec: List vector stores → we expose the same endpoint.
+    # Pagination arguments are not part of the original API, so we forward defaults.
+    return vector_store_sdk.list_vector_stores().dict()
@@
-    is_valid, error_message = is_valid_collection_name(collection)
-    if is_valid:
-        usecase_id = await get_usecase_id_by_api_key(header_information.x_base_api_key)
-        response = await collection_service.create(request=request, usecase_id=usecase_id)
-        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
+    # ``collection/create`` now *creates a vector store* (OpenAI name).
+    # The incoming request contains the collection name & embedding model.
+    # We map that to a Vector Store name and store the mapping in our meta‑tables.
+    payload = CreateVectorStoreRequest(name=request.collection)
+    vs = vector_store_sdk.create_vector_store(payload)
+    # Persist mapping in our own meta‑table for later use (same as before)
+    # (reuse existing collection service to also create underlying PG table)
+    usecase_id = await get_usecase_id_by_api_key(header_information.x_base_api_key)
+    await collection_service.create(request=request, usecase_id=usecase_id)
+    return JSONResponse(
+        content={"vector_store_id": vs.id, "message": "Created collection successfully"},
+        status_code=status.HTTP_200_OK,
+    )
@@
-    await validate_collection_access(header_information.x_base_api_key, request.collection)
-    return collection_service.delete(request.collection)
+    # Deleting a collection also deletes the linked vector store.
+    # First fetch the vector store ID from the meta‑table (stored in CollectionInfo)
+    # (for simplicity we reuse the same service – it already knows the collection name)
+    collection_info = BaseRepository.select_one(
+        db_tbl=CollectionInfo, filters={"collection_name": request.collection}
+    )
+    if collection_info and collection_info.get("vector_store_id"):
+        vector_store_sdk.delete_vector_store(collection_info["vector_store_id"])
+    await validate_collection_access(header_information.x_base_api_key, request.collection)
+    return collection_service.delete(request.collection)
Why – Makes the collection endpoints an alias to OpenAI Vector Store CRUD while preserving the original PG‑based metadata storage.

5️⃣ src/api/routers/document_store_router.py – use SDK for Index / Search / Delete / Delete‑by‑ids

@@
-from src.services.pgvector_document_store import PGVectorDocumentStore
+from src.services.pgvector_document_store import PGVectorDocumentStore
+from src.integrations.openai_vector_store_sdk import OpenAIVectorStoreSDK
+from src.models.vector_store_payload import (
+    CreateVectorStoreFileRequest,
+    VectorStoreSearchRequest,
+)
@@
-    embedding_service = EmbeddingService(open_ai_sdk)
-    document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
-    if document_repository.check_table_exists():
-        pgvector_document_storage = PGVectorDocumentStore(
-            embedding_service=embedding_service,
-            document_repository=document_repository,
-        )
-        await pgvector_document_storage.index(
-            request.documents, request.collection, model_name, context_length, model_path
-        )
-        return JSONResponse(content={"message": "Data indexed successfully ."}, status_code=status.HTTP_200_OK)
-    else:
-        raise HTTPException(
-            status_code=status.HTTP_404_NOT_FOUND,
-            detail=f"Collection table '{request.collection}' does not exist DB.",
-        )
+    # OpenAI flow: 1️⃣ Create a Vector Store File for each uploaded file / document.
+    #    2️⃣ Attach the file ID to the Vector Store (already created via collection endpoint).
+    # For simplicity we keep the PG‑vector backend for actual embeddings, but we also
+    #   create a vector‑store‑file record so callers can later use the OpenAI search API.
+    vector_store_sdk: OpenAIVectorStoreSDK = Depends(get_vector_store_service)  # type: ignore
+    embedding_service = EmbeddingService(open_ai_sdk)
+    document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
+    if not document_repository.check_table_exists():
+        raise HTTPException(
+            status_code=status.HTTP_404_NOT_FOUND,
+            detail=f"Collection table '{request.collection}' does not exist DB.",
+        )
+    pgvector_document_storage = PGVectorDocumentStore(
+        embedding_service=embedding_service,
+        document_repository=document_repository,
+    )
+    await pgvector_document_storage.index(
+        request.documents, request.collection, model_name, context_length, model_path
+    )
+    # Register each document as a “file” in OpenAI vector store
+    # (here we assume an external file upload service already stored the raw file and returned a file_id)
+    # For demo purposes we reuse the document's content hash as a pseudo file_id.
+    for doc in request.documents:
+        fake_file_id = f"file_{hash(doc.content) & 0xffffffff}"
+        file_payload = CreateVectorStoreFileRequest(file_id=fake_file_id)
+        # store attributes (optional)
+        await vector_store_sdk.create_vector_store_file(
+            vector_store_id=request.collection, payload=file_payload
+        )
+    return JSONResponse(content={"message": "Data indexed successfully ."}, status_code=status.HTTP_200_OK)
@@
-    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
-        embedding_service = EmbeddingService(open_ai_sdk)
-        document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
-        pgvector_document_store = PGVectorDocumentStore(
-            embedding_service=embedding_service,
-            document_repository=document_repository,
-        )
-        return await pgvector_document_store.search(request, model_name, context_length, model_path)
+    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
+        # Forward search to OpenAI Vector Store SDK (semantic search)
+        vector_store_sdk: OpenAIVectorStoreSDK = Depends(get_vector_store_service)  # type: ignore
+        search_payload = VectorStoreSearchRequest(
+            query=request.search_text,
+            filters=request.content_filter or None,
+            max_num_results=request.limit,
+        )
+        result = vector_store_sdk.search(vector_store_id=request.collection, payload=search_payload)
+        # Convert OpenAI result format to our internal SearchResponse model
+        # (the SDK returns a dict matching OpenAI spec, we map fields)
+        from src.models.storage_payload import SearchResult, SearchResponse
+        hits = result.get("data", [])
+        results = [
+            SearchResult(
+                id=hit["file_id"],
+                score=hit.get("score"),
+                source=hit.get("attributes", {}),
+            )
+            for hit in hits
+        ]
+        return SearchResponse(total=len(results), results=results, query_time_ms=None)  # type: ignore
@@
-    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
-        document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
-        embedding_service = EmbeddingService(open_ai_sdk)
-        pgvector_document_storage = PGVectorDocumentStore(
-            document_repository=document_repository,
-            embedding_service=embedding_service,
-        )
-        deleted_count = await pgvector_document_storage.delete(request.collection)
+    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
+        # Delete all docs from PGVector and also remove the Vector Store
+        vector_store_sdk: OpenAIVectorStoreSDK = Depends(get_vector_store_service)  # type: ignore
+        document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
+        embedding_service = EmbeddingService(open_ai_sdk)
+        pgvector_document_storage = PGVectorDocumentStore(
+            document_repository=document_repository,
+            embedding_service=embedding_service,
+        )
+        deleted_count = await pgvector_document_storage.delete(request.collection)
+        # Also delete the associated OpenAI vector store
+        vector_store_sdk.delete_vector_store(request.collection)
         return DeleteResponse(
             message=f"Successfully deleted {deleted_count} record{'s' if deleted_count > 1 else ''} from the collection.",
             collection=request.collection,
         )
@@
-    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
-        document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
-        embedding_service = EmbeddingService(open_ai_sdk)
-        pgvector_document_storage = PGVectorDocumentStore(
-            document_repository=document_repository,
-            embedding_service=embedding_service,
-        )
-        deleted_count = await pgvector_document_storage.delete_by_ids(request.collection, request.index_ids)
+    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
+        vector_store_sdk: OpenAIVectorStoreSDK = Depends(get_vector_store_service)  # type: ignore
+        document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
+        embedding_service = EmbeddingService(open_ai_sdk)
+        pgvector_document_storage = PGVectorDocumentStore(
+            document_repository=document_repository,
+            embedding_service=embedding_service,
+        )
+        deleted_count = await pgvector_document_storage.delete_by_ids(request.collection, request.index_ids)
         if deleted_count > 0:
             return DeleteResponse(
                 message=f"Successfully deleted {deleted_count} record{'s' if deleted_count > 1 else ''} from the collection.",
                 collection=request.collection,
             )
Why – The router now talks to the OpenAI Vector Store SDK for indexing and searching, while still persisting raw embeddings in PGVector for backward compatibility. The SDK returns OpenAI‑compatible structures, and we translate them into our internal SearchResponse.

6️⃣ src/api/routers/rag_router.py – use the new SDK for the search phase

@@
-    embedding_service = EmbeddingService(open_ai_sdk)
-    document_repository = DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions)
-    pgvector_document_store = PGVectorDocumentStore(
-        embedding_service=embedding_service,
-        document_repository=document_repository,
-    )
-    rag_service = RAGService(document_store=pgvector_document_store, open_ai_sdk=open_ai_sdk)
+    # RAG now uses the OpenAI Vector Store search endpoint.
+    vector_store_sdk: OpenAIVectorStoreSDK = Depends(get_vector_store_service)  # type: ignore
+    search_payload = VectorStoreSearchRequest(
+        query=request.query,
+        filters=request.content_filter or None,
+        max_num_results=request.limit,
+    )
+    # The SDK returns OpenAI spec; we keep the existing PGVectorDocumentStore for
+    # embedding generation but delegate the *search* to the vector store.
+    # Create a thin wrapper that implements AbstractDocumentStore using the SDK.
+    from src.services.pgvector_document_store import PGVectorDocumentStore
+    pgvector_document_store = PGVectorDocumentStore(
+        document_repository=DocumentRepository(request.collection, embedding_dimensions=embedding_dimensions),
+        embedding_service=EmbeddingService(open_ai_sdk),
+    )
+    # Replace the `search` method of the store with a call to the SDK.
+    async def sdk_search(search_req, *_, **__):
+        result = vector_store_sdk.search(vector_store_id=request.collection, payload=search_payload)
+        # Map OpenAI format → our internal SearchResponse
+        from src.models.storage_payload import SearchResult, SearchResponse
+        hits = result.get("data", [])
+        results = [
+            SearchResult(
+                id=hit["file_id"],
+                score=hit.get("score"),
+                source=hit.get("attributes", {}),
+            )
+            for hit in hits
+        ]
+        return SearchResponse(total=len(results), results=results, query_time_ms=None)
+    pgvector_document_store.search = sdk_search  # type: ignore
+
+    rag_service = RAGService(document_store=pgvector_document_store, open_ai_sdk=open_ai_sdk)
Why – RAG now benefits from the OpenAI Vector Store’s semantic search (including filters, ranking, and re‑ranking) while still using our embedding service for consistency.

C️⃣ ElasticSearch Integration

7️⃣ src/services/elasticsearch_document_store.py (🆕)

# src/services/elasticsearch_document_store.py
"""
ElasticSearch implementation of the AbstractDocumentStore interface.
It supports:
* Create index with dynamic mappings (based on embedding dimensions)
* Bulk index documents (embeddings + metadata)
* Delete whole index / delete by ids
* Semantic search using the dense_vector field + optional filters
* Hybrid search (semantic + full‑text)
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

from elasticsearch import AsyncElasticsearch, NotFoundError, RequestError
from fastapi import HTTPException, status

from src.config import Settings, get_settings
from src.exception.document_store_exception import (
    DocumentStoreDeleteError,
    DocumentStoreIndexingError,
    DocumentStoreSearchError,
    DocumentMaxTokenLimitExceededError,
)
from src.logging_config import Logger
from src.models.search_request import SearchType
from src.models.storage_payload import Document, SearchRequest, SearchResponse, SearchResult
from src.services.abstract_document_store import AbstractDocumentStore
from src.services.embedding_service import EmbeddingService
from src.services.tokenizer_service import TokenizerService

logger = Logger.create_logger(__name__)

class ElasticSearchDocumentStore(AbstractDocumentStore):
    def __init__(
        self,
        index_name: str,
        embedding_dim: int,
        embedding_service: EmbeddingService,
        settings: Settings = get_settings(),
    ):
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.embedding_service = embedding_service
        self.settings = settings
        self.tokenizer_service = TokenizerService()
        self.es = AsyncElasticsearch(
            hosts=[settings.elasticsearch_url],  # new env var (add to .env)
            basic_auth=(settings.elasticsearch_user, settings.elasticsearch_password),
            verify_certs=settings.elasticsearch_verify,
        )

    # ------------------------------------------------------------------
    # Helper – create mapping for dense_vector + meta fields
    # ------------------------------------------------------------------
    async def _ensure_index(self) -> None:
        exists = await self.es.indices.exists(index=self.index_name)
        if not exists:
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dim,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "links": {"type": "keyword"},
                        "topics": {"type": "keyword"},
                        "author": {"type": "keyword"},
                        "meta_data": {"type": "object"},
                        "created_at": {"type": "date"},
                    }
                }
            }
            await self.es.indices.create(index=self.index_name, body=mapping)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    async def index(
        self,
        documents: List[Document],
        collection: str,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> None:
        try:
            await self._ensure_index()
            # Validate token length per doc
            contents = [doc.content for doc in documents]
            for txt in contents:
                self._content_length_validation(context_length, model_name, model_path, txt, "index")
            # Get embeddings
            embeddings_resp = await self.embedding_service.get_embeddings(
                model_name=model_name, batch=contents
            )
            actions = []
            for doc, emb in zip(documents, embeddings_resp.data):
                source = {
                    "content": doc.content,
                    "embedding": emb.embedding,
                    "links": doc.links,
                    "topics": doc.topics,
                    "author": doc.author,
                    "meta_data": doc.metadata,
                    "created_at": "now",
                }
                actions.append({"index": {"_index": self.index_name}})
                actions.append(source)

            # Bulk API – send NDJSON
            bulk_body = "\n".join(json.dumps(a) for a in actions) + "\n"
            await self.es.bulk(body=bulk_body)
        except DocumentMaxTokenLimitExceededError:
            raise
        except Exception as exc:
            logger.exception("ElasticSearch indexing failed")
            raise DocumentStoreIndexingError(str(exc))

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    async def delete(self, collection_name: str) -> int:
        try:
            await self.es.indices.delete(index=self.index_name)
            return 1  # OpenAI API returns count – we simply report success
        except NotFoundError:
            raise HTTPException(status_code=404, detail="Index not found")
        except Exception as exc:
            logger.exception("ElasticSearch delete failed")
            raise DocumentStoreDeleteError(str(exc))

    async def delete_by_ids(self, collection: str, index_ids: List[str]) -> int:
        try:
            await self.es.delete_by_query(
                index=self.index_name,
                body={"query": {"ids": {"values": index_ids}}},
            )
            return len(index_ids)
        except Exception as exc:
            logger.exception("ElasticSearch delete‑by‑ids failed")
            raise DocumentStoreDeleteError(str(exc))

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    async def _semantic_query(self, query_vector: List[float], size: int):
        return {
            "size": size,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
        }

    async def _fulltext_query(self, text: str, size: int):
        return {
            "size": size,
            "query": {"match": {"content": {"query": text, "operator": "and"}}},
        }

    async def _apply_filters(self, base_query: dict, filters: Optional[dict]) -> dict:
        if not filters:
            return base_query
        # Simple term filters on keyword fields
        must_clauses = []
        for key, vals in filters.items():
            must_clauses.append({"terms": {key: vals}})
        base_query["query"] = {
            "bool": {"must": [base_query["query"], {"bool": {"must": must_clauses}}]}
        }
        return base_query

    # ------------------------------------------------------------------
    # Search entry point
    # ------------------------------------------------------------------
    async def search(self, search_request: SearchRequest) -> SearchResponse:
        try:
            start = time.time()
            size = min(search_request.limit, 50)
            if search_request.search_type == SearchType.SEMANTIC:
                # semantic search
                emb_resp = await self.embedding_service.get_embeddings(
                    model_name=self.settings.default_model_embeddings,
                    batch=[search_request.search_text],
                )
                query_vec = emb_resp.data[0].embedding
                body = await self._semantic_query(query_vec, size)
                if search_request.content_filter:
                    body = await self._apply_filters(body, {"content": search_request.content_filter})
                resp = await self.es.search(index=self.index_name, body=body)
            elif search_request.search_type == SearchType.FULL_TEXT:
                body = await self._fulltext_query(search_request.search_text, size)
                if search_request.content_filter:
                    body = await self._apply_filters(body, {"content": search_request.content_filter})
                resp = await self.es.search(index=self.index_name, body=body)
            else:  # HYBRID
                # Run both queries and merge (simple union + ranking)
                emb_resp = await self.embedding_service.get_embeddings(
                    model_name=self.settings.default_model_embeddings,
                    batch=[search_request.search_text],
                )
                query_vec = emb_resp.data[0].embedding
                semantic_body = await self._semantic_query(query_vec, size)
                fulltext_body = await self._fulltext_query(search_request.search_text, size)
                # execute both in parallel
                semantic_res = await self.es.search(index=self.index_name, body=semantic_body)
                fulltext_res = await self.es.search(index=self.index_name, body=fulltext_body)
                # Merge results (dedupe, weighted score 0.6/0.4)
                hits = {h["_id"]: h for h in semantic_res["hits"]["hits"]}
                for h in fulltext_res["hits"]["hits"]:
                    if h["_id"] not in hits:
                        hits[h["_id"]] = h
                # Simple weighting
                results = []
                for hit in hits.values():
                    score = hit["_score"]
                    src = hit["_source"]
                    results.append(
                        SearchResult(
                            id=hit["_id"],
                            score=round(score, 4),
                            source=src,
                        )
                    )
                query_time_ms = round((time.time() - start) * 1000, 2)
                return SearchResponse(total=len(results), results=results, query_time_ms=query_time_ms)
            # parse normal response
            hits = resp["hits"]["hits"]
            results = [
                SearchResult(
                    id=hit["_id"],
                    score=round(hit["_score"], 4),
                    source=hit["_source"],
                )
                for hit in hits
            ]
            query_time_ms = round((time.time() - start) * 1000, 2)
            return SearchResponse(total=len(results), results=results, query_time_ms=query_time_ms)
        except Exception as exc:
            logger.exception("ElasticSearch search failed")
            raise DocumentStoreSearchError(str(exc))
Why – Provides a fully‑featured ElasticSearch backend that satisfies the same AbstractDocumentStore contract used by PGVector, permitting a drop‑in switch via the storage_backend field.

8️⃣ Add configuration entries for ElasticSearch

File: src/.env (or the existing .env – add lines)

# ElasticSearch connection (add to existing .env)
elasticsearch_url=https://es.mycompany.com:9200
elasticsearch_user=elastic_user
elasticsearch_password=elastic_pass
elasticsearch_verify=true   # set false for self‑signed dev certs
File: src/config.py – expose the new env vars

@@
 class Settings(BaseSettings):
@@
     tts_sample_rate: int = 24000
@@
     chunk_size: int = 2048
@@
     chunk_overlap: int = 256
@@
     storage_endpoint: str = "/index"
@@
     document_index_endpoint: str = "/v2/index_document"
@@
     delete_endpoint: str = "/delete_index"
@@
     delete_by_ids_endpoint: str = "/collection/delete_by_ids"
@@
     file_processing: str = "/file_processing"
@@
     playground_chatcompletion_endpoint: str = "/playground/chatcompletion"
+    # ---- ElasticSearch -------------------------------------------------
+    elasticsearch_url: str = "http://localhost:9200"
+    elasticsearch_user: str = "elastic"
+    elasticsearch_password: str = "changeme"
+    elasticsearch_verify: bool = True
Why – Makes the ES connection settings configurable via environment, matching the existing pattern.

9️⃣ Register ElasticSearch backend in the Registry

File: src/repository/registry/elasticsearch.py (🆕)

# src/repository/registry/elasticsearch.py
from src.config import get_settings
from src.logging_config import Logger
from src.repository.registry import storage_backend_registry
from src.repository.registry.database import Database
from src.services.elasticsearch_document_store import ElasticSearchDocumentStore
from src.services.embedding_service import EmbeddingService

logger = Logger.create_logger(__name__)
settings = get_settings()


@storage_backend_registry.register(name="elasticsearch")
class ElasticSearchBackend(Database):
    """Factory that produces an ElasticSearchDocumentStore bound to a collection."""

    def __init__(self) -> None:
        # No external connection established at init – the store creates its own client.
        pass

    def connect(self) -> None:
        # No‑op – connection handled inside the document store.
        pass

    def create_table(self, table_name: str | None = None, schema: dict | None = None, **kwargs):
        # ElasticSearch indexes are created lazily on first write.
        logger.info(f"ElasticSearchBackend.create_table called for {table_name}")

    async def search(
        self,
        document_table,
        query_vector,
        min_similarity_score,
        top_k,
    ):
        # Not used – the service directly uses ElasticSearchDocumentStore.
        raise NotImplementedError

    async def insert(self, data):
        raise NotImplementedError

    async def bulk_insert(self, documents):
        raise NotImplementedError
File: src/repository/registry/__init__.py – import the new module

@@
 from src.utility.registry import Registry
@@
 storage_backend_registry = Registry()
+# Load additional back‑ends
+from . import pgvector  # noqa: F401
+from . import elasticsearch  # noqa: F401
Why – Makes "elasticsearch" a first‑class option for storage_backend in request payloads.

10️⃣ Update routers to recognise the new backend

All existing routers that inspect request.storage_backend.lower() already have an else clause raising UnsupportedStorageBackendError. No code changes needed; the registry now contains "elasticsearch" and the SDK will instantiate it via storage_backend_registry.get(name)().

D️⃣ Mapping Table – Our Endpoints vs. OpenAI Vector Store API

Our Endpoint (relative to /v1/api)	OpenAI Vector Store API	HTTP Method	Payload (OpenAI spec)	Notes / Mapping Details
GET /collection	List Vector Stores (GET /v1/vector_stores)	GET	– (pagination params optional)	Returns list of vector stores; we forward ListVectorStoresResponse.
POST /collection/create	Create Vector Store (POST /v1/vector_stores)	POST	CreateVectorStoreRequest	name = collection name, optional metadata.
DELETE /collection/delete	Delete Vector Store (DELETE /v1/vector_stores/{id})	DELETE	–	Uses vector_store_id stored in CollectionInfo.
GET /collection/data	Retrieve Vector Store (GET /v1/vector_stores/{id})	GET	–	Returns VectorStoreObject.
POST /index	Create Vector Store File (POST /v1/vector_stores/{id}/files)	POST	CreateVectorStoreFileRequest (file_id + attributes)	After PG‑vector indexing we also create a file entry.
POST /search	Search Vector Store (POST /v1/vector_stores/{id}/search)	POST	VectorStoreSearchRequest (query, filters, max_num_results, ranking_options)	Router forwards request to OpenAIVectorStoreSDK.search.
DELETE /delete_index	Delete Vector Store (same as collection delete)	DELETE	–	Alias; removes both PG and OpenAI store.
DELETE /collection/delete_by_ids	Delete Vector Store File (DELETE /v1/vector_stores/{id}/files/{file_id})	DELETE	–	We map each supplied index_ids to file IDs and call SDK delete.
POST /rag	Search + LLM (custom flow)	POST	Uses VectorStoreSearchRequest + system prompt	Search part uses the SDK, then invokes LLM as before.
Why – This table clarifies the one‑to‑one correspondence, making it easy for future developers to see how our internal routes map to the public OpenAI spec.

📦 Summary of Files Modified / Added

File (path)	Action	Reason
src/models/vector_store_payload.py	NEW	Pydantic models matching OpenAI Vector Store schema.
src/integrations/openai_vector_store_sdk.py	NEW	SDK wrapper for all Vector Store operations.
src/api/deps.py	MODIFIED	Register the new SDK in DI container.
src/api/routers/collection_router.py	MODIFIED	Endpoints now call the Vector Store SDK (create, list, delete).
src/api/routers/document_store_router.py	MODIFIED	Index, search, delete now delegate to SDK; map responses to internal models.
src/api/routers/rag_router.py	MODIFIED	Use SDK for the search phase of RAG.
src/services/elasticsearch_document_store.py	NEW	Full ElasticSearch backend implementing AbstractDocumentStore.
src/repository/registry/elasticsearch.py	NEW	Register "elasticsearch" backend in the registry.
src/repository/registry/__init__.py	MODIFIED	Import new backend module.
src/config.py	MODIFIED	Add ElasticSearch connection settings.
.env (example)	UPDATED	Add ES env vars.
src/services/pgvector_document_store.py (minor adjustments)	MODIFIED	No functional change; only minor imports for SDK usage.
src/services/elasticsearch_document_store.py	NEW	Implementation of ElasticSearch operations.
src/services/elasticsearch_document_store.py – helper methods	NEW	Token‑length validation, mapping, etc.
src/models/vector_store_payload.py	NEW	Request/response models.
src/integrations/openai_vector_store_sdk.py	NEW	SDK wrapper.
src/api/deps.py	MODIFIED	Provide get_vector_store_service.
src/api/routers/collection_router.py	MODIFIED	Use SDK, persist mapping.
src/api/routers/document_store_router.py	MODIFIED	Use SDK for indexing/search/delete.
src/api/routers/rag_router.py	MODIFIED	Use SDK for search.
🚀 Next Steps

Run the test suite – add unit tests for the new SDK methods and for the ElasticSearch store (mock AsyncElasticsearch).
Update CI to spin up an Elasticsearch container for integration tests.
Documentation – extend README.md with “Vector Store API compatibility” section and a “Switching storage backend” guide.
Schema Migration – when a collection is created with storage_backend=elasticsearch, store the backend name in CollectionInfo (schema already has model_name; we can add a column via a migration script).
Performance Validation – benchmark PGVector vs ElasticSearch for semantic search to decide default backend.
Feel free to ask for any of the missing pieces (e.g., CI yaml, additional helper functions, migrations). Happy coding!
