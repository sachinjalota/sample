## src/services/vector_store_interface.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from src.models.vector_store_payload import (
    CreateVectorStoreRequest,
    VectorStoreObject,
    CreateVectorStoreFileRequest,
    VectorStoreFileObject,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
)

class VectorStoreInterface(ABC):
    """Common contract for any vector‑store implementation (PGVector, ElasticSearch, …)."""

    @abstractmethod
    async def create_store(self, payload: CreateVectorStoreRequest) -> VectorStoreObject: ...

    @abstractmethod
    async def list_stores(self, limit: int = 20, after: str | None = None,
                         before: str | None = None, order: str = "desc") -> List[VectorStoreObject]: ...

    @abstractmethod
    async def retrieve_store(self, store_id: str) -> VectorStoreObject: ...

    @abstractmethod
    async def update_store(self, store_id: str, payload: CreateVectorStoreRequest) -> VectorStoreObject: ...

    @abstractmethod
    async def delete_store(self, store_id: str) -> Dict[str, Any]: ...

    # -------- file operations -------------------------------------------------
    @abstractmethod
    async def add_file(self, store_id: str, payload: CreateVectorStoreFileRequest) -> VectorStoreFileObject: ...

    @abstractmethod
    async def list_files(self, store_id: str, limit: int = 20,
                         after: str | None = None, before: str | None = None,
                         order: str = "desc", filter: str | None = None) -> List[VectorStoreFileObject]: ...

    @abstractmethod
    async def retrieve_file(self, store_id: str, file_id: str) -> VectorStoreFileObject: ...

    @abstractmethod
    async def update_file(self, store_id: str, file_id: str,
                         attributes: Dict[str, Any]) -> VectorStoreFileObject: ...

    @abstractmethod
    async def delete_file(self, store_id: str, file_id: str) -> Dict[str, Any]: ...

    # -------- search ----------------------------------------------------------
    @abstractmethod
    async def search(self, store_id: str, request: VectorStoreSearchRequest) -> VectorStoreSearchResponse: ...

