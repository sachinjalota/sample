## src/models/vector_store_payload.py

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator

# --- Vector Store (collection) -------------------------------------------------
class VectorStoreChunkingStrategyStatic(BaseModel):
    type: Literal["static"] = "static"
    max_chunk_size_tokens: int = Field(..., ge=100, le=4096)
    chunk_overlap_tokens: int = Field(..., ge=0, le=2048)

class VectorStoreChunkingStrategyAuto(BaseModel):
    type: Literal["auto"] = "auto"

ChunkingStrategy = Union[VectorStoreChunkingStrategyStatic, VectorStoreChunkingStrategyAuto]

class CreateVectorStoreRequest(BaseModel):
    name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    file_ids: Optional[List[str]] = None
    chunking_strategy: Optional[ChunkingStrategy] = None
    expires_after: Optional[Dict[str, Any]] = None   # { "anchor": "...", "days": int }

class VectorStoreObject(BaseModel):
    id: str
    object: Literal["vector_store"] = "vector_store"
    created_at: int
    name: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    bytes: Optional[int] = None
    file_counts: Optional[Dict[str, int]] = None
    expires_at: Optional[int] = None
    status: Optional[str] = None
    usage_bytes: Optional[int] = None
    last_active_at: Optional[int] = None

class ListVectorStoresResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[VectorStoreObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

# --- Vector Store Files -------------------------------------------------------
class CreateVectorStoreFileRequest(BaseModel):
    file_id: str
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[ChunkingStrategy] = None

class VectorStoreFileObject(BaseModel):
    id: str
    object: Literal["vector_store.file"] = "vector_store.file"
    created_at: int
    vector_store_id: str
    status: str
    usage_bytes: Optional[int] = None
    last_error: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[Dict[str, Any]] = None

class ListVectorStoreFilesResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[VectorStoreFileObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

# --- Search ---------------------------------------------------------------
class VectorStoreSearchRequest(BaseModel):
    query: Union[str, List[float]]
    filters: Optional[Dict[str, Any]] = None
    max_num_results: Optional[int] = Field(default=10, ge=1, le=50)
    ranking_options: Optional[Dict[str, Any]] = None
    # The OpenAI spec also allows a `vector`‑style query, but we keep it simple.

class VectorStoreSearchResult(BaseModel):
    id: str
    object: Literal["vector_store.file"] = "vector_store.file"
    score: float
    attributes: Optional[Dict[str, Any]] = None
    content: List[Dict[str, str]]          # [{ "type": "text", "text": "..." }, …]

class VectorStoreSearchResponse(BaseModel):
    object: Literal["vector_store.search_results.page"] = "vector_store.search_results.page"
    search_query: str
    data: List[VectorStoreSearchResult]
    has_more: bool = False
    next_page: Optional[str] = None
