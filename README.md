from typing import Dict, Literal

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
    store_model_info: Dict[str, str] = Depends(get_store_model_info),
    vector_service: VectorStoreService = Depends(get_vector_store_service),
) -> CreateVectorStoreFileResponse:
    try:
        model_name: str = store_model_info["model_name"]
        model_path: str = store_model_info["model_path"]
        embedding_dimensions: int = int(store_model_info["embedding_dimensions"])
        context_length: int = int(store_model_info["context_length"])

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
    embedding_service: EmbeddingService = Depends(get_embedding_service),
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
