## src/services/pgvector_document_store.py

# --------------------------------------------------------------
#  Updated file: src/services/pgvector_vector_store.py
# --------------------------------------------------------------
import json
import time
from collections import defaultdict
from typing import List, Dict, Any

from fastapi import HTTPException

from src.config import Settings, get_settings
from src.exception.document_store_exception import (
    DocumentMaxTokenLimitExceededError,
    DocumentStoreDeleteError,
    DocumentStoreIndexingError,
    DocumentStoreSearchError,
)
from src.exception.exceptions import DatabaseConnectionError
from src.logging_config import Logger
from src.models.vector_store_payload import (
    CreateVectorStoreRequest,
    VectorStoreObject,
    CreateVectorStoreFileRequest,
    VectorStoreFileObject,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)
from src.repository.document_repository import DocumentRepository
from src.services.embedding_service import EmbeddingService
from src.services.tokenizer_service import TokenizerService
from src.services.vector_store_interface import VectorStoreInterface

logger = Logger.create_logger(__name__)

class PGVectorVectorStore(VectorStoreInterface):
    """PGVector implementation that satisfies the OpenAI‑compatible VectorStoreInterface."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService,
        settings: Settings = get_settings(),
    ):
        self.document_repository = document_repository
        self.embedding_service = embedding_service
        self.settings = settings
        self.tokenizer_service = TokenizerService()

    # ------------------------------------------------------------------ #
    #  Helper – translate PGVector's internal representation to the API model
    # ------------------------------------------------------------------ #
    def _store_to_object(self, table_name: str) -> VectorStoreObject:
        # PGVector does not store meta‑information like bytes, so we fill what we have.
        return VectorStoreObject(
            id=table_name,
            created_at=int(time.time()),
            name=table_name,
            object="vector_store",
            file_counts={"completed": 0, "in_progress": 0, "failed": 0, "cancelled": 0, "total": 0},
        )

    # ------------------------------------------------------------------ #
    #  Vector‑Store CRUD (maps to the existing Collection APIs)
    # ------------------------------------------------------------------ #
    async def create_store(self, payload: CreateVectorStoreRequest) -> VectorStoreObject:
        # In the current system a “store” is just a collection (table) + embedding dim
        # We reuse the collection‑creation flow.
        from src.services.collection_service import CollectionService
        collection_svc = CollectionService(base_repository=BaseRepository(),
                                         base_model_ops=BaseModelOps())
        # Create collection with the requested name (or generate one)
        collection_name = payload.name or f"vs_{uuid.uuid4().hex[:8]}"
        # The embedding model is taken from the request metadata or default
        model_name = payload.metadata.get("embedding_model") if payload.metadata else None
        model_name = model_name or self.settings.default_model_embeddings
        # this will raise if model not found
        from src.utility.collection_helpers import check_embedding_model
        _, embedding_dims, _ = await check_embedding_model(model_name)

        await collection_svc.create(
            request=CreateCollection(collection=collection_name, model_name=model_name),  # type: ignore
            usecase_id="system",  # no auth check needed – internal call
        )
        return self._store_to_object(collection_name)

    async def list_stores(self, limit: int = 20, after: str | None = None,
                          before: str | None = None, order: str = "desc") -> List[VectorStoreObject]:
        # The collection table list is stored in CollectionInfo
        from src.services.collection_service import CollectionService
        collection_svc = CollectionService(base_repository=BaseRepository(),
                                         base_model_ops=BaseModelOps())
        collections = await collection_svc.get(usecase_id="system")   # returns dict
        stores = [
            self._store_to_object(name) for name in collections.get("collections", [])
        ]
        # Simple slice based pagination (real backend would use cursors)
        start = 0
        if after:
            start = next((i + 1 for i, s in enumerate(stores) if s.id == after), 0)
        elif before:
            end = next((i for i, s in enumerate(stores) if s.id == before), len(stores))
            start = max(0, end - limit)
        return stores[start:start + limit]

    async def retrieve_store(self, store_id: str) -> VectorStoreObject:
        # just check existence
        if not DocumentRepository(store_id).check_table_exists():
            raise HTTPException(status_code=404, detail="Vector store not found")
        return self._store_to_object(store_id)

    async def update_store(self, store_id: str, payload: CreateVectorStoreRequest) -> VectorStoreObject:
        # Only metadata/name updates are supported – we store them in a simple table “vector_store_meta”
        # For brevity we just return the same object (no persistence needed for now)
        return self._store_to_object(store_id)

    async def delete_store(self, store_id: str) -> Dict[str, Any]:
        # Re‑use collection deletion logic
        from src.services.collection_service import CollectionService
        collection_svc = CollectionService(base_repository=BaseRepository(),
                                         base_model_ops=BaseModelOps())
        collection_svc.delete(store_id)
        return {"id": store_id, "object": "vector_store.deleted", "deleted": True}

    # ------------------------------------------------------------------ #
    #  File operations – a “file” is a collection of chunks stored in the same
    #  table.  We treat each uploaded document as a file.
    # ------------------------------------------------------------------ #
    async def add_file(self, store_id: str, payload: CreateVectorStoreFileRequest) -> VectorStoreFileObject:
        # The file_id is actually the source file identifier (e.g. GCS object name)
        # We simply index the provided file content (if any) as a new document.
        # For this repo the actual file handling is done earlier (cloud storage upload).
        # Here we just record a placeholder entry in the table.
        # We reuse the indexing logic that already expects a list of Documents.
        # The caller must have already uploaded the file to cloud storage and passed
        # the `file_id` (cloud object path).  We'll fetch the bytes via CloudStorage.
        from src.integrations.cloud_storage import CloudStorage
        storage = CloudStorage()
        file_bytes = storage.download_object(payload.file_id)

        # Convert to plain text (PDF → markdown) – reuse PDF helper
        from src.utility.pdf_helpers import get_markdown_from_pdf
        tmp_path = "/tmp/tmp_upload_file.pdf"
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        content = get_markdown_from_pdf(tmp_path)

        # Build a Document payload and index it
        document = {
            "content": content,
            "links": [payload.file_id],
            "topics": [],
            "author": None,
            "meta_data": {},
        }
        # Index using existing DocumentRepository
        repo = DocumentRepository(store_id, embedding_dimensions=0)  # dims resolved later
        embedding_dims = (await check_embedding_model(self.settings.default_model_embeddings))[1]
        await PGVectorVectorStore(repo, self.embedding_service).index(
            documents=[document], collection=store_id,
            model_name=self.settings.default_model_embeddings,
            context_length=self.settings.min_similarity_score,  # dummy, not used here
            model_path=self.settings.default_model_embeddings,
        )
        # Return minimal file object
        return VectorStoreFileObject(
            id=payload.file_id,
            created_at=int(time.time()),
            vector_store_id=store_id,
            object="vector_store.file",
            status="completed",
            attributes=payload.attributes,
            chunking_strategy=payload.chunking_strategy.dict() if payload.chunking_strategy else None,
        )

    async def list_files(self, store_id: str, limit: int = 20,
                         after: str | None = None, before: str | None = None,
                         order: str = "desc", filter: str | None = None) -> List[VectorStoreFileObject]:
        # List the “documents” stored in the collection – we map each row to a file.
        repo = DocumentRepository(store_id, embedding_dimensions=0)
        rows = repo.document_table.__table__.select()
        # Use BaseRepository.select_table_details for simplicity
        from src.repository.base_repository import BaseRepository
        details = BaseRepository.select_table_details(table_name=store_id, limit=limit, offset=0)
        files = [
            VectorStoreFileObject(
                id=str(row["id"]),
                created_at=int(row["created_at"].timestamp()),
                vector_store_id=store_id,
                object="vector_store.file",
                status="completed",
                attributes={"links": row.get("links"), "topics": row.get("topics")},
                usage_bytes=len(row["content"].encode()),
            )
            for row in details
        ]
        return files

    async def retrieve_file(self, store_id: str, file_id: str) -> VectorStoreFileObject:
        repo = DocumentRepository(store_id, embedding_dimensions=0)
        row = repo.document_table.__table__.select().where(repo.document_table.id == file_id)
        # use BaseRepository.select_one shortcut
        from src.repository.base_repository import BaseRepository
        rec = BaseRepository.select_one(db_tbl=repo.document_table, filters={"id": file_id})
        if not rec:
            raise HTTPException(status_code=404, detail="File not found")
        return VectorStoreFileObject(
            id=file_id,
            created_at=int(rec["created_at"].timestamp()),
            vector_store_id=store_id,
            object="vector_store.file",
            status="completed",
            attributes={"links": rec.get("links"), "topics": rec.get("topics")},
            usage_bytes=len(rec["content"].encode()),
        )

    async def update_file(self, store_id: str, file_id: str,
                          attributes: Dict[str, Any]) -> VectorStoreFileObject:
        # Update the metadata columns `links`, `topics`, `author`, `meta_data`
        repo = DocumentRepository(store_id, embedding_dimensions=0)
        # Simple update – we only touch the JSON columns.
        repo.update_many(
            db_tbl=repo.document_table,
            filters={"id": file_id},
            data={"meta_data": attributes.get("meta_data", {}), "links": attributes.get("links", []),
                  "topics": attributes.get("topics", []), "author": attributes.get("author")}
        )
        return await self.retrieve_file(store_id, file_id)

    async def delete_file(self, store_id: str, file_id: str) -> Dict[str, Any]:
        repo = DocumentRepository(store_id, embedding_dimensions=0)
        deleted = repo.delete_by_ids([file_id])
        return {"id": file_id, "object": "vector_store.file.deleted", "deleted": deleted > 0}

    # ------------------------------------------------------------------ #
    #  Search (semantic / full‑text / hybrid) – reuse PGVectorDocumentStore
    # ------------------------------------------------------------------ #
    async def search(self, store_id: str, request: VectorStoreSearchRequest) -> VectorStoreSearchResponse:
        repo = DocumentRepository(store_id, embedding_dimensions=0)
        pg_store = PGVectorDocumentStore(document_repository=repo,
                                         embedding_service=self.embedding_service,
                                         settings=self.settings)
        # Translate OpenAI request to our internal SearchRequest
        from src.models.storage_payload import SearchRequest, SearchType
        internal_req = SearchRequest(
            collection=store_id,
            search_type=SearchType.SEMANTIC if isinstance(request.query, str) else SearchType.FULL_TEXT,
            storage_backend=StorageBackend.PGVECTOR,
            search_text=request.query if isinstance(request.query, str) else "",
            limit=request.max_num_results or self.settings.default_document_limit,
            min_score=request.ranking_options.get("score_threshold") if request.ranking_options else self.settings.min_similarity_score,
        )
        results = await pg_store.search(internal_req, self.settings.default_model_embeddings,
                                        self.settings.min_similarity_score,
                                        self.settings.default_model_embeddings)
        # Build the OpenAI‑compatible response
        search_results = [
            VectorStoreSearchResult(
                id=res.id,
                score=res.score or 0.0,
                attributes=res.source,
                content=[{"type": "text", "text": res.source.get("content", "")}],
            )
            for res in results
        ]
        return VectorStoreSearchResponse(
            search_query=request.query if isinstance(request.query, str) else "",
            data=search_results,
        )
