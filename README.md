from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
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
    CreateVectorStoreFileRequest,
    CreateVectorStoreRequest,
    FileStatus,
    SearchResult,
    StorageBackend,
)
from src.repository.base_repository import BaseRepository
from src.repository.elasticsearch_ddl import ElasticsearchDDL
from src.repository.elasticsearch_dml import ElasticsearchDML
from src.services.base_class.vector_store_base import BaseVectorStore, VectorStoreConfig
from src.services.embedding_service import EmbeddingService
from src.services.factory.vector_store_factory import VectorStoreFactory

logger = Logger.create_logger(__name__)


@VectorStoreFactory.register("elasticsearch", description="Elasticsearch + GCP backend")
class ElasticsearchGCPVectorStore(BaseVectorStore):
    """Concrete Elasticsearch + GCP implementation of the Vector Store interface."""

    def __init__(self, config: VectorStoreConfig, embedding_service: Optional[EmbeddingService] = None) -> None:
        super().__init__(document_repository=None, embedding_service=embedding_service, settings=config.extra)
        self.client: Elasticsearch = get_elasticsearch_client()

    # ================================================================
    # REQUIRED ABSTRACT IMPLEMENTATIONS
    # ================================================================
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

    async def _validate_backend_type(self, record: Dict[str, Any]) -> None:
        """Ensure the vector_db field indicates Elasticsearch."""
        if record["vector_db"] != StorageBackend.ELASTICSEARCH.value:
            raise VectorStoreError(f"Vector Store '{record['id']}' is not stored in Elasticsearch")

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

            file_info_doc: Dict[str, Any] = {
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
        actions: List[Dict[str, Any]] = []
        for doc in documents:
            doc_id = str(uuid4())
            doc_body: Dict[str, Any] = {
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
        query: Dict[str, Any] = {"query": {"term": {"file_id": file_id}}}
        try:
            response = self.client.delete_by_query(index=index_name, body=query)
            return response.get("deleted", 0)
        except Exception as e:
            logger.error(f"[ES] Delete documents failed: {e}")
            raise

    async def _drop_backend_tables(self, store_name: str) -> None:
        """Drops Elasticsearch indices (file_info + chunks)."""
        try:
            ElasticsearchDDL.drop_indices(store_name)
            logger.info(f"[ES+GCP] Dropped indices for store '{store_name}'")
        except Exception as err:
            logger.error(f"[ES+GCP] Failed to drop indices: {err}", exc_info=True)
            raise VectorStoreError(f"Elasticsearch index deletion failed for store '{store_name}'")

    async def _fetch_file_backend(
        self, vectorstoreid: str, vectorstorefileid: str, usecase_id: str
    ) -> Dict[str, Any]:
        store_name = f"{vectorstoreid}_file_info"
        query: Dict[str, Any] = {
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
    async def _delete_metadata_and_chunks(
        self, store_name: str, vectorstoreid: str, vectorstorefileid: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Delete file metadata and its chunks from Elasticsearch.
        - Delete file info record first → raise if fails.
        - Delete chunks → if fails, restore file info only (no chunk restore).
        """

        vs_file_info = f"{store_name}_file_info"
        vs_chunks = f"{store_name}_chunks"

        file_query: Dict[str, Any] = {
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
        chunk_query: Dict[str, Any] = {
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

    def _restore_file_info_record(self, index_name: str, file_info_record: Dict[str, Any]) -> bool:
        try:
            if not file_info_record:
                logger.warning(f"No file_info_record provided for restore into {index_name}")
                return False

            restore_data: Dict[str, Any] = {
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

    # ================================================================
    # SEARCH OPERATIONS
    # ================================================================
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

        knn_query: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": search_request.limit,
            "num_candidates": search_request.limit * 10,
        }

        if filters:
            knn_query["filter"] = filters

        query_body: Dict[str, Any] = {
            "knn": knn_query,
            "min_score": search_request.min_score,
            "size": search_request.limit,
            "_source": {"excludes": ["embedding"]},
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

        query_body: Dict[str, Any] = {
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

        results: List[SearchResult] = []
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

            filter_dict: Dict[str, Any] = {"filter": filters} if filters else {}

            query_body: Dict[str, Any] = {
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
                        **filter_dict,
                    }
                },
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": search_request.limit,
                    "num_candidates": search_request.limit * 10,
                    **filter_dict,
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
        """Build Elasticsearch query filters from search request."""
        filters: List[Dict[str, Any]] = []

        if search_request.content_filter:
            filters.append({"terms": {"content": search_request.content_filter}})

        if search_request.link_filter:
            filters.append({"terms": {"links": search_request.link_filter}})

        if search_request.topic_filter:
            filters.append({"terms": {"topics": search_request.topic_filter}})

        if not filters:
            return None

        return {"bool": {"must": filters}} if len(filters) > 1 else filters[0]
