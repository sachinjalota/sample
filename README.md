from typing import Dict, Literal, Optional

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
    embedding_model_info: Dict[str, int] = Depends(get_valid_embedding_model),
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
    after: Optional[str] = None,
    before: Optional[str] = None,
    order: Literal["asc", "desc"] = Query("desc"),
    storage_backend: Optional[Literal["pgvector", "elasticsearch"]] = Query(None),
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
    store_model_info: Dict[str, str] = Depends(get_store_model_info),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> SearchVectorStoreResponse:
    try:
        model_name: str = store_model_info["model_name"]
        model_path: str = store_model_info["model_path"]
        embedding_dimensions: int = int(store_model_info["embedding_dimensions"])
        context_length: int = int(store_model_info["context_length"])

        logger.info(f"Search Request {request} from {header_information.x_session_id}")

        row_data = BaseRepository.select_one(  # type: ignore
            db_tbl=VectorStoreInfo, filters={"id": store_id, "vector_db": request.storage_backend}
        )
        if not row_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector store with Id '{store_id}' not found with '{request.storage_backend}'.",
            )

        store_name: str = row_data["name"]

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
