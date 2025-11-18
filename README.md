# src/models/rag_payload.py
from typing import Optional
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field, field_validator

from src.config import get_settings
from src.models.vector_store_payload import SearchType, StorageBackend
from src.prompts.default_prompts import DEFAULT_RAG_SYSTEM_PROMPT

settings = get_settings()


class RAGRequest(BaseModel):
    guardrail_id: Optional[str] = Field(
        None,
        description="Optional guardrail ID to validate the prompt or response.",
    )
    system_prompt: str = Field(
        default=DEFAULT_RAG_SYSTEM_PROMPT,
        description="System prompt for LLM",
    )
    vector_store_id: str = Field(..., description="ID of the vector store to search.")
    search_type: SearchType = Field(..., description="Type of search: semantic, full_text, or hybrid.")
    storage_backend: StorageBackend = Field(
        ...,
        description="Storage backend (pgvector or elasticsearch).",
    )
    query: str = Field(..., description="The query text for search.")
    
    content_filter: Optional[list[str]] = Field(
        default=None, 
        description="Content keywords filter"
    )
    link_filter: Optional[list[str]] = Field(
        default=None, 
        description="Links filter"
    )
    topic_filter: Optional[list[str]] = Field(
        default=None, 
        description="Topics filter"
    )
    
    limit: int = Field(
        default=settings.default_document_limit,
        ge=1,
        description="Maximum number of documents to retrieve",
    )
    min_score: float = Field(
        default=settings.min_similarity_score,
        ge=0,
        description="Minimum similarity score threshold.",
    )
    model_name: str = Field(
        default=settings.default_model,
        description="LLM model name",
    )

    @field_validator("vector_store_id")
    def vector_store_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vector_store_id cannot be empty.")
        return v

    @field_validator("query")
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty.")
        return v


class RAGResponse(BaseModel):
    llm_response: ChatCompletion = Field(
        ..., 
        description="LLM-generated response with metadata."
    )
    context: list[dict] = Field(
        ..., 
        description="Retrieved source documents used as context."
    )


# src/services/rag_service.py
import time
import uuid
from typing import List, Dict, Any

from openai.types.chat import ChatCompletion

from src.exception.document_store_exception import DocumentMaxTokenLimitExceededError
from src.exception.rag_exception import RAGError, RAGResponseGenerationError
from src.integrations.open_ai_sdk import OpenAISdk
from src.logging_config import Logger
from src.models.completion_payload import ChatCompletionRequest
from src.models.rag_payload import RAGRequest, RAGResponse
from src.models.vector_store_payload import SearchVectorStoreRequest, SearchVectorStoreResponse
from src.services.service_layer.vector_store_service import VectorStoreService

logger = Logger.create_logger(__name__)


class RAGService:
    def __init__(
        self, 
        vector_store_service: VectorStoreService, 
        open_ai_sdk: OpenAISdk
    ) -> None:
        self.vector_store_service = vector_store_service
        self.open_ai_sdk = open_ai_sdk

    async def process(
        self,
        session_id: str,
        api_key: str,
        rag_request: RAGRequest,
        model_name: str,
        context_length: int,
        model_path: str,
    ) -> RAGResponse:
        try:
            # Build search request using new vector store API
            search_request = SearchVectorStoreRequest(
                query=rag_request.query,
                search_type=rag_request.search_type,
                storage_backend=rag_request.storage_backend,
                max_num_results=rag_request.limit,
                ranking_options={
                    "ranker": "auto",
                    "score_threshold": rag_request.min_score
                }
            )

            # Perform search using VectorStoreService
            search_response: SearchVectorStoreResponse = await self.vector_store_service.search(
                search_request=search_request,
                store_id=rag_request.vector_store_id,
                model_name=model_name,
                context_length=context_length,
                model_path=model_path,
            )

            logger.info(f"Search returned {len(search_response.data)} results")

            # Handle empty results
            if not search_response.data:
                logger.warning(
                    f"No documents found in store {rag_request.vector_store_id} "
                    f"for query: {rag_request.query}"
                )
                return self._build_empty_response(rag_request)

            # Extract context for LLM
            retrieved_data = self._extract_context(search_response.data)
            user_context = self._format_user_context(search_response.data)

            # Generate LLM response
            chat_request = ChatCompletionRequest(
                user_prompt=rag_request.query,
                model_name=rag_request.model_name,
                guardrail_id=rag_request.guardrail_id,
            )

            llm_response = await self._llm_invoke(
                session_id=session_id,
                system_prompt=rag_request.system_prompt,
                api_key=api_key,
                context=retrieved_data,
                request=chat_request,
            )

            logger.info("RAG processing completed successfully")
            return RAGResponse(llm_response=llm_response, context=user_context)

        except DocumentMaxTokenLimitExceededError:
            raise
        except RAGError:
            raise
        except Exception as e:
            logger.exception(
                f"RAG failed for store {rag_request.vector_store_id} "
                f"and query {rag_request.query}"
            )
            raise RAGError(f"Unexpected error: {str(e)}")

    def _extract_context(self, results: List[Any]) -> List[str]:
        """Extract text content from search results."""
        context = []
        for result in results:
            for content_block in result.content:
                if content_block.type == "text":
                    context.append(content_block.text)
        return context

    def _format_user_context(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Format results for user-facing context."""
        formatted = []
        for result in results:
            content_texts = [
                block.text for block in result.content 
                if block.type == "text"
            ]
            formatted.append({
                "file_id": result.file_id,
                "filename": result.filename,
                "score": result.score,
                "content": " ".join(content_texts),
                "attributes": result.attributes or {},
            })
        return formatted

    def _build_empty_response(self, rag_request: RAGRequest) -> RAGResponse:
        """Build response when no documents are found."""
        placeholder = ChatCompletion(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            object="chat.completion",
            model=rag_request.model_name,
            choices=[{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": (
                        "No documents were found matching your query. "
                        "Please revise your query or verify the input."
                    ),
                },
            }],
        )
        return RAGResponse(llm_response=placeholder, context=[])

    async def _llm_invoke(
        self,
        session_id: str,
        system_prompt: str,
        api_key: str,
        context: List[str],
        request: ChatCompletionRequest,
    ) -> ChatCompletion:
        """Invoke LLM with retrieved context."""
        try:
            context_data = "\n\n".join(context)
            enhanced_prompt = f"{system_prompt}\n\nContext:\n{context_data}"

            logger.debug(f"Enhanced system prompt length: {len(enhanced_prompt)}")

            response = await self.open_ai_sdk.complete(
                request=request,
                system_prompt=enhanced_prompt,
                session_id=session_id,
                api_key=api_key,
            )
            return response

        except Exception as e:
            logger.exception("LLM invocation failed")
            raise RAGResponseGenerationError(f"LLM call error: {str(e)}")


# src/api/routers/rag_router.py
from fastapi import APIRouter, Depends, status

from src.api.deps import (
    get_openai_service,
    get_vector_store_service,
    validate_headers_and_api_key,
)
from src.config import get_settings
from src.integrations.open_ai_sdk import OpenAISdk
from src.logging_config import Logger
from src.models.headers import HeaderInformation
from src.models.rag_payload import RAGRequest, RAGResponse
from src.services.rag_service import RAGService
from src.services.service_layer.vector_store_service import VectorStoreService
from src.utility.guardrails import scan_prompt
from src.utility.vector_store_helpers import get_store_model_info

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()


@router.post(
    f"{settings.rag_endpoint}",
    summary="Query RAG system for contextual response",
    description=(
        "Performs Retrieval-Augmented Generation (RAG) by searching the vector store "
        "and using retrieved documents as context for LLM generation."
    ),
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
)
async def rag(
    request: RAGRequest,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    store_model_info: dict = Depends(get_store_model_info),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service),
    open_ai_sdk: OpenAISdk = Depends(get_openai_service),
) -> RAGResponse:
    logger.info(f"RAG request from {header_information.x_session_id}")

    model_name = store_model_info["model_name"]
    model_path = store_model_info["model_path"]
    context_length = store_model_info["context_length"]

    # Guardrails validation
    if settings.guardrail_enabled:
        logger.info("Validating query with guardrails")
        scan_args = {
            "prompt": request.query,
            "session_id": header_information.x_session_id,
            "api_key": header_information.x_base_api_key,
        }
        if request.guardrail_id:
            scan_args["guardrail_id"] = request.guardrail_id

        guardrail_result = await scan_prompt(**scan_args)
        if not guardrail_result.get("is_valid", False):
            logger.warning(f"Guardrails failed: {guardrail_result}")
            raise RAGError(
                "Query violates safety guidelines. Please revise and try again."
            )

    # Process RAG
    rag_service = RAGService(
        vector_store_service=vector_store_service,
        open_ai_sdk=open_ai_sdk,
    )

    return await rag_service.process(
        session_id=header_information.x_session_id,
        api_key=header_information.x_base_api_key,
        rag_request=request,
        model_name=model_name,
        context_length=context_length,
        model_path=model_path,
    )
