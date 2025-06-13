
I have following few files that belongs to fastapi based project.

>> document_store_router.py
from fastapi import APIRouter, Depends, Request, status
from starlette.responses import JSONResponse

from src.api.deps import get_openai_service, validate_headers_and_api_key
from src.config import get_settings
from src.exception.document_store_exception import UnsupportedStorageBackendError
from src.integrations.open_ai_sdk import OpenAISdk
from src.logging_config import Logger
from src.models.headers import HeaderInformation
from src.models.indexing_payload import IndexingPayload
from src.models.storage_payload import (
    DeleteRequest,
    DeleteResponse,
    SearchRequest,
    SearchResponse,
    StorageBackend,
)
from src.repository.document_repository import DocumentRepository
from src.services.embedding_service import EmbeddingService
from src.services.pgvector_document_store import PGVectorDocumentStore
from src.utility.collection_helpers import validate_collection_access

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()


@router.post(
    f"{settings.storage_endpoint}",
    summary="Index documents into the specified vector storage backend.",
    description=(
        "Accepts a batch of documents and processes them for embedding generation and storage. "
        "Currently supports PGVector as the storage backend. Embeddings are generated using the provided "
        "OpenAI-compatible model "
        "and stored in the specified collection. Requires valid API key headers for authentication."
    ),
    response_description="Confirmation message upon successful indexing.",
    status_code=status.HTTP_200_OK,
)
# @opik.track
async def index(
    fastapi_request: Request,
    request: IndexingPayload,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> JSONResponse:
    await validate_collection_access(header_information.x_base_api_key, request.collection)
    logger.info(
        f"Indexing request from {header_information.x_session_id} for collection: {request.collection} and document count: {len(request.documents)}"
    )
    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
        embedding_service = EmbeddingService(open_ai_sdk)
        document_repository = DocumentRepository(request.collection, create_if_not_exists=False)
        pgvector_document_storage = PGVectorDocumentStore(
            embedding_service=embedding_service,
            document_repository=document_repository,
        )
        await pgvector_document_storage.index(request.documents, request.collection)
        return JSONResponse(content="Successfully indexed.", status_code=status.HTTP_200_OK)
    else:
        raise UnsupportedStorageBackendError(f"Unsupported storage backend: {request.storage_backend}")


@router.post(
    f"{settings.search_endpoint}",
    summary="Perform semantic and full-text search over indexed documents.",
    description=(
        "Executes a hybrid search combining semantic similarity and keyword-based full-text search over "
        "documents stored in the configured vector database. Accepts a query and optional filters (e.g., topic), "
        "and returns the most relevant documents based on embeddings and metadata fields. This endpoint supports "
        "filtering, ranking, and result explanation features depending on the backend implementation."
    ),
    response_description="List of documents matching the search criteria, ranked by relevance.",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
)
# @opik.track
async def search(
    request: SearchRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> SearchResponse:
    await validate_collection_access(header_information.x_base_api_key, request.collection)
    logger.info(f"Search Request {request} from {header_information.x_session_id}")
    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
        embedding_service = EmbeddingService(open_ai_sdk)
        document_repository = DocumentRepository(request.collection, create_if_not_exists=False)
        pgvector_document_store = PGVectorDocumentStore(
            embedding_service=embedding_service,
            document_repository=document_repository,
        )
        return await pgvector_document_store.search(request)
    else:
        raise UnsupportedStorageBackendError(f"Unsupported storage backend: {request.storage_backend}")


@router.delete(
    f"{settings.delete_endpoint}",
    summary="Delete all documents from the specified vector storage backend.",
    description=(
        "Deletes all documents stored in the specified collection within the configured vector database. "
        "Requires valid API key headers for authentication."
    ),
    response_description="Confirmation message upon successful deletion.",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
)
# @opik.track
async def delete(
    request: DeleteRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> DeleteResponse:
    await validate_collection_access(header_information.x_base_api_key, request.collection)
    logger.info(f"Delete request for collection: {request.collection}")
    if request.storage_backend.lower() == StorageBackend.PGVECTOR.value:
        document_repository = DocumentRepository(request.collection, create_if_not_exists=False)
        embedding_service = EmbeddingService(open_ai_sdk)
        pgvector_document_storage = PGVectorDocumentStore(
            document_repository=document_repository,
            embedding_service=embedding_service,
        )
        await pgvector_document_storage.delete(request.collection)
        return DeleteResponse(
            message="Successfully deleted the indexes of collection.",
            collection=request.collection,
        )
    else:
        raise UnsupportedStorageBackendError(f"Unsupported storage backend: {request.storage_backend}")

>> pgvector_document_store.py
import time
from typing import List

from fastapi import HTTPException

from src.config import Settings, get_settings
from src.exception.document_store_exception import (
    DocumentStoreDeleteError,
    DocumentStoreIndexingError,
    DocumentStoreSearchError,
)
from src.exception.exceptions import DatabaseConnectionError
from src.logging_config import Logger
from src.models.storage_payload import Document, SearchRequest, SearchResponse
from src.repository.document_repository import DocumentRepository
from src.services.abstract_document_store import AbstractDocumentStore
from src.services.embedding_service import EmbeddingService

logger = Logger.create_logger(__name__)


class PGVectorDocumentStore(AbstractDocumentStore):
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService,
        settings: Settings = get_settings(),
    ):
        self.document_repository = document_repository
        self.embedding_service = embedding_service
        self.settings = settings

    async def delete(self, collection: str) -> None:
        try:
            logger.info(f"Deleting all records from collection: {collection}")
            deleted_count = self.document_repository.delete()
            if deleted_count > 0:
                logger.info(f"Successfully deleted {deleted_count} records from collection: {collection}")
            else:
                logger.info(f"No records were deleted from collection: {collection}. Table may have been dropped.")
        except DatabaseConnectionError as e:
            logger.error(f"Collection '{collection}' does not exist: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Collection '{collection}' does not exist in the database.")
        except Exception as e:
            logger.exception(f"Failed to  delete table {collection}")
            raise DocumentStoreDeleteError(f"Unexpected error during delete: {str(e)}")

    async def index(self, documents: List[Document], collection: str) -> None:
        try:
            document_model = self.document_repository.get_document_model()
            content_list = [document.content for document in documents]
            embeddings = await self.embedding_service.get_embeddings(
                model_name=self.settings.default_model_embeddings, batch=content_list
            )
            docs_with_embeddings = []
            for i, doc in enumerate(documents):
                doc_row = document_model(
                    content=doc.content,
                    embedding=embeddings.data[i].embedding,
                    links=doc.links,
                    meta_data=doc.metadata,
                    topics=doc.topics,
                    author=doc.author,
                )

                docs_with_embeddings.append(doc_row)
            self.document_repository.insert(documents=docs_with_embeddings)
        except Exception as e:
            logger.exception(f"Indexing failed for table {collection} document count{len(documents)}")
            raise DocumentStoreIndexingError(f"Unexpected error during indexing: {str(e)}")

    async def search(self, search_request: SearchRequest) -> SearchResponse:
        try:
            start_time = time.time()
            embeddings = await self.embedding_service.get_embeddings(
                model_name=self.settings.default_model_embeddings,
                batch=[search_request.search_text],
            )
            query_vector = embeddings.data[0].embedding
            results = self.document_repository.search(
                query_vector=query_vector,
                search_terms=search_request.content_filter,
                include_links=search_request.link_filter,
                include_topics=search_request.topic_filter,
                top_k=search_request.limit,
                min_similarity_score=search_request.min_score,
            )
            query_time_ms = round((time.time() - start_time) * 1000, 2)

            logger.info(f"Total searched document count {len(results)}")
            return SearchResponse(total=len(results), results=results, query_time_ms=query_time_ms)
        except DatabaseConnectionError as db_error:
            raise db_error
        except Exception as e:
            logger.exception(f"Search failed for request {search_request}")
            raise DocumentStoreSearchError(f"Unexpected error during search: {str(e)}")


>> document_repository.py
import uuid
from datetime import datetime
from typing import Dict, Type, TypeAlias

from pgvector.sqlalchemy import VECTOR  # type: ignore
from sqlalchemy import ARRAY, JSON, DateTime, Index, String, Text, cast, delete, inspect
from sqlalchemy.dialects.postgresql import UUID, array
from sqlalchemy.dialects.postgresql.ext import plainto_tsquery, to_tsvector
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import (
    DeclarativeBase,
    InstrumentedAttribute,
    Mapped,
    defer,
    mapped_column,
)
from sqlalchemy.sql.expression import BinaryExpression

from src.config import get_settings
from src.db.connection import create_session, engine
from src.exception.exceptions import DatabaseConnectionError
from src.logging_config import Logger
from src.models.storage_payload import SearchResult

logger = Logger.create_logger(__name__)
DocumentModelType: TypeAlias = type["DocumentBase"]

# Cache to avoid redefinition
_document_model_cache: Dict[str, DocumentModelType] = {}
settings = get_settings()


class DocumentBase(DeclarativeBase):
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    links: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    topics: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    author: Mapped[str | None] = mapped_column(String, nullable=True)
    meta_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    @declared_attr  # type: ignore
    def __tablename__(cls) -> str:
        return cls.__name__.lower()


def create_document_model(
    table_name: str, embedding_dimensions: int, create_if_not_exists: bool = True
) -> DocumentModelType:
    if table_name in _document_model_cache:
        logger.info(f"Document table already exists, returning existing document: {table_name}")
        return _document_model_cache[table_name]
    try:
        unique_table_name = f"{table_name.capitalize()}_Document"
        index_name = f"{unique_table_name.lower()}_index"
        model: type[DocumentBase] = type(
            unique_table_name,
            (DocumentBase,),
            {
                "__tablename__": table_name,
                "__table_args__": {"extend_existing": True},
                "embedding": mapped_column(VECTOR(embedding_dimensions), nullable=False),
            },
        )
        inspector = inspect(engine)
        logger.info(f"all tables in DB: {inspector.get_table_names()} flag: {create_if_not_exists}")
        if table_name not in inspector.get_table_names():
            if create_if_not_exists:
                logger.info(f"creating table now {table_name}")
                model.metadata.create_all(engine)
                _document_model_cache[table_name] = model
                existing_indexes = {row["name"] for row in inspector.get_indexes(table_name)}
                if index_name not in existing_indexes:
                    index = Index(index_name, model.__table__.c.content)
                    index.create(bind=engine)
        else:
            _document_model_cache[table_name] = model
        return model
    except Exception as e:
        raise DatabaseConnectionError(f"{str(e)}")


class DocumentRepository:
    def __init__(self, table_name: str, create_if_not_exists: bool = False, embedding_dimensions: int = 1024) -> None:
        self.table_name = table_name
        self.document_table: DocumentModelType = create_document_model(
            table_name, embedding_dimensions, create_if_not_exists
        )
        self.fields_to_include = [c.name for c in self.document_table.__table__.columns if c.name != "embedding"]

    def get_document_model(self) -> Type[DocumentBase]:
        return self.document_table

    def check_table_exists(self) -> bool:
        inspector = inspect(engine)
        return self.table_name in inspector.get_table_names()

    def delete(self) -> int:
        """Deletes related indexes, and clears model cache."""
        _document_model_cache.pop(self.table_name, None)
        if not self.check_table_exists():
            raise DatabaseConnectionError(f"Collection '{self.table_name}' does not exist in the database.")
        try:
            with create_session() as session:
                query = delete(self.document_table)
                result = session.execute(query)
            return result.rowcount
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to delete index of '{self.table_name}': {e}")

    def delete_collection(self) -> int:
        """Deletes the table, removes related indexes, and clears model cache."""
        _document_model_cache.pop(self.table_name, None)
        if not self.check_table_exists():
            raise DatabaseConnectionError(f"Collection '{self.table_name}' does not exist in the database.")
        try:
            with create_session() as session:
                self.document_table.__table__.drop(bind=session.bind)  # type: ignore
                return 1
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to delete Collection '{self.table_name}' and its indexes: {e}")

    @staticmethod
    def insert(documents: list[DocumentBase]) -> None:
        with create_session() as session:
            session.bulk_save_objects(documents)

    def search(
        self,
        query_vector: list[float],
        search_terms: list | None = None,
        include_links: list | None = None,
        include_topics: list | None = None,
        top_k: int = settings.default_document_limit,
        min_similarity_score: float = settings.min_similarity_score,
    ) -> list[SearchResult]:
        if not self.check_table_exists():
            raise DatabaseConnectionError(f"Collection '{self.table_name}' does not exist in the database.")
        max_distance = 1 - min_similarity_score
        cosine_distance = self.document_table.embedding.cosine_distance(query_vector)  # type: ignore
        with create_session() as session:
            query = session.query(
                self.document_table,
                cosine_distance.label("similarity_score"),
            ).options(defer(self.document_table.embedding))  # type: ignore
            # Apply exact match on link array
            if include_links:
                query = query.filter(
                    self.document_table.links.op("&&")(cast(array(include_links), self.document_table.links.type))
                )
            # Apply exact match on topic array
            if include_topics:
                query = query.filter(
                    self.document_table.topics.op("&&")(cast(array(include_topics), self.document_table.topics.type))
                )
            # Apply full text search filter
            if search_terms:
                content_filter = self._get_full_text_search_filter(search_terms, self.document_table.content)
                query = query.filter(content_filter)
            # Apply cosine distance filter
            query = query.filter(cosine_distance <= max_distance)

            results = query.order_by(cosine_distance).limit(top_k).all()
            output = [
                SearchResult(
                    id=str(record[0].id),
                    score=round(1 - record[1], 4),
                    source=self._to_search_result(record[0]).source,
                )
                for record in results
            ]
        return output

    @staticmethod
    def _get_full_text_search_filter(search_terms: list[str], table_column: InstrumentedAttribute) -> BinaryExpression:
        search_query_string = " OR ".join(search_terms)
        ts_query = plainto_tsquery("english", search_query_string)  # type: ignore[no-untyped-call]
        search_filter = to_tsvector("english", table_column).op("@@")(ts_query)  # type: ignore[no-untyped-call]
        return search_filter

    def _to_search_result(self, record: DocumentBase) -> SearchResult:
        return SearchResult(
            id=str(record.id),
            source={field: getattr(record, field) for field in self.fields_to_include},
        )



If you observe closely, all the functions from document_repository are being called in document_store_router via pgvectore_document_store whic if i understand correctly act as a 
middlelayer. Very similarly, I want you to update the following code that also calls db_tbl_ops.py functions in collection_router.py via a middle layer and also same with 
document_repository functions.

Below I am sharing code for your task
>> collection_router.py
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import validate_headers_and_api_key, validate_collection_access
from src.config import get_settings
from src.repository.db_tbl_ops import TableOps
from src.db.platform_meta_tables import CollectionInfo
from src.logging_config import Logger
from src.models.collection_payload import CreateCollections, DeleteCollection
from src.models.headers import HeaderInformation
from src.repository.document_repository import DocumentRepository
from src.utility.collection_helpers import (
    check_embedding_model,
    validate_collection_access
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

            TableOps.insert_one(
                db_tbl=CollectionInfo,
                data={
                    "uuid": collection_uuid,
                    "channel_id": entry.channel_id,
                    "usecase_id": entry.usecase_id,
                    "collection_name": entry.collection_name,
                    "model": entry.model,
                },
            )
            DocumentRepository(table_name=collection_uuid, create_if_not_exists=True, embedding_dimensions=dims)

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
    collection_access: None = Depends(validate_collection_access)
) -> dict:
    try:
        repo = DocumentRepository(request.collection_uid, create_if_not_exists=False)
        if not repo.check_table_exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection table '{request.collection_uid}' does not exist in the database.",
            )
        repo.delete_collection()

        deleted = TableOps.delete(db_tbl=CollectionInfo, filters={"uuid": request.collection_uid})
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


>> db_tble_ops.py
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

from fastapi import HTTPException, status
from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import DeclarativeMeta

from src.db.connection import create_session_platform
from src.logging_config import Logger

T = TypeVar("T", bound=DeclarativeMeta)
logger = Logger.create_logger(__name__)


class TableOps:
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
