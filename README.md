from typing import List, Optional

from src.models.vector_store_payload import (
    CreateVectorStoreFileRequest,
    CreateVectorStoreFileResponse,
    CreateVectorStoreRequest,
    CreateVectorStoreResponse,
    DeleteVectorStoreFileResponse,
    DeleteVectorStoreResponse,
    RetrieveFileResponse,
    SearchVectorStoreRequest,
    SearchVectorStoreResponse,
)
from src.services.base_class.vector_store_base import BaseVectorStore
from src.services.embedding_service import EmbeddingService
from src.services.factory.vector_store_factory import (
    VectorStoreConfig,
    VectorStoreFactory,
)


class VectorStoreService:
    """
    High-level service that delegates vector store operations
    to the appropriate backend via VectorStoreFactory.
    """

    def __init__(self, backend_name: str, embedding_service: EmbeddingService) -> None:
        self.embedding_service: EmbeddingService = embedding_service
        config = VectorStoreConfig(backend=backend_name)
        self.vectorstore: BaseVectorStore = VectorStoreFactory.create(
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
    ) -> CreateVectorStoreResponse:
        """Creates a new vector store (database or index)."""
        result = await self.vectorstore.create_store(
            payload, usecase_id, embedding_dimensions=embedding_dimensions
        )
        return CreateVectorStoreResponse.model_validate(result)

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
        """Indexes a file/document into the vector store."""
        result = await self.vectorstore.create_store_file(
            payload,
            store_id,
            usecase_id,
            model_name,
            context_length,
            model_path,
            embedding_dimensions,
        )
        return CreateVectorStoreFileResponse.model_validate(result)

    async def search(
        self,
        payload: SearchVectorStoreRequest,
        store_id: str,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> SearchVectorStoreResponse:
        """Performs semantic/hybrid/full-text search."""
        return await self.vectorstore.search_vector_store(
            payload,
            store_id,
            model_name,
            context_length,
            model_path,
        )

    async def delete_store(self, store_id: str, usecase_id: str) -> DeleteVectorStoreResponse:
        """Deletes the entire vector store (DB or index)."""
        result = await self.vectorstore.delete(store_id, usecase_id)
        return DeleteVectorStoreResponse.model_validate(result)

    async def delete_file(
        self, store_id: str, file_id: str, usecase_id: str
    ) -> DeleteVectorStoreFileResponse:
        """Deletes a specific file/document from a vector store."""
        result = await self.vectorstore.delete_by_id(store_id, file_id, usecase_id)
        return DeleteVectorStoreFileResponse.model_validate(result)

    async def retrieve_file(
        self, store_id: str, file_id: str, usecase_id: str
    ) -> RetrieveFileResponse:
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
    ) -> List[CreateVectorStoreResponse]:
        """
        Lists available vector stores for a use case.
        If vector_db is None, lists all stores regardless of backend.
        If vector_db is specified, filters by that backend.
        """
        return await self.vectorstore.list_stores(
            usecase_id, limit, after, before, order, vector_db
        )
