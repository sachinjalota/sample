from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import asc, desc

from src.db.connection import create_session
from src.db.platform_meta_tables import VectorStoreInfo
from src.exception.document_store_exception import DocumentStoreIndexingError
from src.exception.exceptions import VectorStoreError
from src.logging_config import Logger
from src.models.storage_payload import SearchRequest
from src.models.vector_store_payload import SearchResult, StorageBackend, CreateVectorStoreRequest
from src.repository.base_repository import BaseRepository
from src.repository.document_repository import DocumentRepository
from src.repository.vectorstore_ddl import VectorStoreDDL
from src.services.base_class.vector_store_base import BaseVectorStore, VectorStoreConfig
from src.services.embedding_service import EmbeddingService
from src.services.factory.vector_store_factory import VectorStoreFactory
from src.utility.vector_store_utils import (
    create_chunks_tbl_model,
    create_file_info_tbl_model,
)

logger = Logger.create_logger(__name__)


@VectorStoreFactory.register("pgvector", description="Postgres + pgvector backend")
class PGVectorStore(BaseVectorStore):
    """Concrete PGVector implementation of the Vector Store interface."""

    def __init__(self, config: VectorStoreConfig, embedding_service: Optional[EmbeddingService] = None) -> None:
        super().__init__(document_repository=None, embedding_service=embedding_service, settings=config.extra)

    # =================================================================
    # Implement all abstract methods required by BaseVectorStore
    # =================================================================

    async def _create_backend_store(
        self,
        payload: CreateVectorStoreRequest,
        usecase_id: str,
        embedding_dimensions: int,
        store_id: str,
        now_dt: datetime,
        expires_at: Optional[datetime],
    ) -> Dict[str, Any]:
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
        self,
        payload: Any,
        store_id: str,
        store_name: str,
        model_name: str,
        context_length: int,
        model_path: str,
        embedding_dimensions: int,
    ) -> int:
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

            file_metadata: Dict[str, Any] = {
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

    async def _drop_backend_tables(self, store_name: str) -> None:
        """Drop PostgreSQL tables for the vector store."""
        from src.repository.vectorstore_ddl import VectorStoreDDL

        VectorStoreDDL.drop_table_and_index(tbl_name=store_name)

    async def _fetch_file_backend(
        self, vectorstoreid: str, vectorstorefileid: str, usecase_id: str
    ) -> Dict[str, Any]:
        logger.info(f"[PGVector] Retrieving file {vectorstorefileid} from store {vectorstoreid}")
        # This needs to be implemented to fetch from PostgreSQL
        vs_file_info_tbl = create_file_info_tbl_model(f"{vectorstoreid}_file_info")
        
        file_record = BaseRepository.select_one(
            db_tbl=vs_file_info_tbl,
            filters={"file_id": vectorstorefileid, "vs_id": vectorstoreid},
            session_factory=create_session,
        )
        
        if not file_record:
            raise VectorStoreError(f":: PG :: File '{vectorstorefileid}' not found in store '{vectorstoreid}'")
        
        return file_record

    async def _list_backend_stores(
        self,
        usecase_id: str,
        limit: int,
        after: Optional[str],
        before: Optional[str],
        order: str,
        vector_db: Optional[str],
    ) -> List[Dict[str, Any]]:
        try:
            if order.lower() not in {"asc", "desc"}:
                raise VectorStoreError(f"Invalid orderby value '{order}'")

            order_by_clause = (
                desc(VectorStoreInfo.created_at) if order.lower() == "desc" else asc(VectorStoreInfo.created_at)
            )

            # Build filters
            filters: Dict[str, Any] = {"usecase_id": usecase_id}
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

    async def _validate_backend_type(self, record: Dict[str, Any]) -> None:
        """Ensure the vector_db field indicates PGVector."""
        if record["vector_db"] != StorageBackend.PGVECTOR.value:
            raise VectorStoreError(f"Vector Store '{record['id']}' is not stored in PGVector")

    async def _delete_metadata_and_chunks(
        self, store_name: str, vectorstoreid: str, vectorstorefileid: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Delete file metadata and chunks from PostgreSQL."""
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

    # ================================================================
    # SEARCH OPERATIONS
    # ================================================================
    async def _semantic_search(
        self, search_request: Any, index_name: str, model_name: str
    ) -> List[SearchResult]:
        """Semantic similarity search using pgvector."""
        embeddings = await self.embedding_service.get_embeddings(
            model_name=model_name,
            batch=[search_request.search_text],
        )
        query_vector = embeddings.data[0].embedding
        
        document_repository = DocumentRepository(index_name, embedding_dimensions=len(query_vector))
        _, results = document_repository.sematic_search(
            query_vector=query_vector,
            search_terms=search_request.content_filter,
            include_links=search_request.link_filter,
            include_topics=search_request.topic_filter,
            top_k=search_request.limit,
            min_similarity_score=search_request.min_score,
        )
        return results

    async def _fulltext_search(
        self, search_request: Any, index_name: str
    ) -> List[SearchResult]:
        """Full-text search using PostgreSQL full-text search."""
        document_repository = DocumentRepository(index_name, embedding_dimensions=0)
        _, results = document_repository.fulltext_search(
            query=search_request.search_text,
            search_terms=search_request.content_filter,
            include_links=search_request.link_filter,
            include_topics=search_request.topic_filter,
            top_k=search_request.limit,
            min_relevance_score=search_request.min_score,
        )
        return results

    async def _hybrid_search(
        self, search_request: Any, index_name: str, model_name: str
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and full-text results."""
        semantic_results: List[SearchResult] = await self._semantic_search(search_request, index_name, model_name)
        fulltext_results: List[SearchResult] = await self._fulltext_search(search_request, index_name)
        
        logger.info(
            f"Hybrid search -> Semantic search results: {len(semantic_results)}, "
            f"Full-text search results: {len(fulltext_results)}"
        )
        
        score_map: Dict[str, Dict[str, float]] = defaultdict(lambda: {"semantic": 0.0, "fulltext": 0.0})
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
