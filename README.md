## Chat Log

<h2 style='color:green;'>##System##</h2>
You are an AI assistant that helps people find information.

<h2 style='color:cyan;'>##User##</h2>


I have the following code files of a service from a project based on fastapi structure.

>> main.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable, List

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse

from src.api.deps import get_rate_limiter
from src.api.routers import (
    chatcompletion_router,
    collection_router,
    document_store_router,
    embeddings_router,
    file_processing_router,
    list_embedding_model_router,
    rag_router,
    speech_to_text_router,
    text_to_speech_router,
)
from src.api.routers.internal import (
    chatcompletion_router as internal_chatcompletion_router,
    document_qna_router,
    export_traces_router,
    file_processing_router as internal_file_processing_router,
    generate_qna_router as internal_generate_qna_router,
    playground_router as internal_playground_router,
    upload_file_router,
)
from src.api.routers.v2 import document_store_router_v2
from src.config import get_settings
from src.db.connection import initialize_platform_db
from src.exception.document_store_exception import DocumentStoreError
from src.exception.exceptions import DatabaseConnectionError, PdfChunkingError
from src.exception.rag_exception import RAGError
from src.exception.scanner_exceptions import ScanFailedException
from src.logging_config import Logger
from src.utility.registry_initializer import Storage
from src.websockets.stt_websocket_router import router as stt_ws_router
from src.websockets.tts_websocket_router import router as tts_ws_router

settings = get_settings()

allowed_origins: List[str] = settings.allowed_origins.split(",")
allowed_methods: List[str] = settings.allow_methods.split(",")
allowed_headers: List[str] = settings.allow_headers.split(",")

logger = Logger.create_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan handler for startup and shutdown events."""
    logger.info("Starting up application...")
    try:
        initialize_platform_db()
    except Exception as e:
        logger.error(f"Error during database initialization: {str(e)}")
    app.state.registry_storage = Storage()
    yield
    logger.info("Shutting down application...")


def create_app(version: str = "0.1.0") -> FastAPI:
    env = settings.deployment_env
    service_slug = settings.service_slug

    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=version,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        lifespan=lifespan,
        root_path=f"/{env}/{service_slug}",
    )

    app.state.limiter = get_rate_limiter()
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=settings.pg_api_limit_exceed_status_code,
            content={"message": settings.pg_api_limit_exceed_message},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=settings.allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
    )

    @app.middleware("http")
    async def add_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        origin = request.headers.get("origin")

        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
        if not (
            request.url.path.endswith("/docs")
            or request.url.path.endswith("/redoc")
            or request.url.path.endswith("/openapi.json")
        ):
            response.headers["Content-Security-Policy"] = "default-src 'none'"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        headers_to_remove = ["access-control-allow-headers", "access-control-allow-methods"]

        for header in headers_to_remove:
            try:
                del response.headers[header]
            except KeyError:
                pass
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.error(f"Validation error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": [{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in exc.errors()],
                "body": exc.body,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.error(f"Http error: {exc}", exc_info=True)
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(DocumentStoreError)
    async def document_store_exception_handler(request: Request, exc: DocumentStoreError) -> JSONResponse:
        logger.error(f"Document store error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(exc)},
        )

    @app.exception_handler(DatabaseConnectionError)
    async def db_operational_exception_handler(request: Request, exc: DatabaseConnectionError) -> JSONResponse:
        logger.error(f"DatabaseConnectionError: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": str(exc)},
        )

    @app.exception_handler(RAGError)
    async def rag_exception_handler(request: Request, exc: RAGError) -> JSONResponse:
        logger.error(f"RAG error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(exc)},
        )

    @app.exception_handler(PdfChunkingError)
    async def pdf_chunking_exception_handler(request: Request, exc: PdfChunkingError) -> JSONResponse:
        logger.error(f"PdfChunkingError : {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(exc)},
        )

    @app.exception_handler(ScanFailedException)
    async def scan_exception_handler(request: Request, exc: ScanFailedException) -> JSONResponse:
        logger.error(f"Scanner error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=exc.status,
            content={
                "message": settings.scan_failed_message,
                "scanner": exc.scanners,
                "scanned_content": exc.input_prompt,
            },
        )

    @app.get("/", tags=["MAIN"])
    async def read_root() -> dict[str, str]:
        return {"name": "Chat As Service, go to docs path for API detail"}

    @app.get(f"{settings.api_common_prefix}{settings.health_check}", tags=["MAIN"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(
        chatcompletion_router.router,
        prefix=settings.api_common_prefix,
        tags=["COMPLETION"],
    )
    app.include_router(
        embeddings_router.router,
        prefix=settings.api_common_prefix,
        tags=["EMBEDDING"],
    )
    app.include_router(
        document_store_router.router,
        prefix=settings.api_common_prefix,
        tags=["INDEXING_DELETE_SEARCH"],
    )

    app.include_router(
        rag_router.router,
        prefix=settings.api_common_prefix,
        tags=["RAG"],
    )

    app.include_router(
        file_processing_router.router,
        prefix=settings.api_common_prefix,
        tags=["FILE_PROCESSING"],
    )

    app.include_router(
        document_store_router_v2.router,
        prefix=settings.api_common_prefix,
        tags=["CREATE_INDEXING_SEARCH_V2"],
    )

    app.include_router(speech_to_text_router.router, prefix=settings.api_common_prefix, tags=["SPEECH_TO_TEXT"])

    app.include_router(text_to_speech_router.router, prefix=settings.api_common_prefix, tags=["TEXT_TO_SPEECH"])

    app.include_router(
        document_qna_router.router,
        prefix=f"{settings.api_common_prefix}/{settings.internal_api_url}",
        tags=["INTERNAL"],
    )

    app.include_router(
        upload_file_router.router,
        prefix=f"{settings.api_common_prefix}/{settings.internal_api_url}",
        tags=["INTERNAL"],
    )
    app.include_router(
        internal_chatcompletion_router.router,
        prefix=f"{settings.api_common_prefix}/{settings.internal_api_url}",
        tags=["INTERNAL"],
    )
    app.include_router(
        internal_file_processing_router.router,
        prefix=f"{settings.api_common_prefix}/{settings.internal_api_url}",
        tags=["INTERNAL"],
    )
    app.include_router(
        internal_generate_qna_router.router,
        prefix=f"{settings.api_common_prefix}/{settings.internal_api_url}",
        tags=["INTERNAL"],
    )
    app.include_router(
        internal_playground_router.router,
        prefix=f"{settings.api_common_prefix}/{settings.internal_api_url}",
        tags=["INTERNAL"],
    )
    app.include_router(
        export_traces_router.router,
        prefix=f"{settings.api_common_prefix}/{settings.internal_api_url}",
        tags=["INTERNAL"],
    )

    app.include_router(stt_ws_router, prefix=settings.ws_common_prefix)

    app.include_router(tts_ws_router, prefix=settings.ws_common_prefix)

    app.include_router(collection_router.router, prefix=settings.api_common_prefix, tags=["COLLECTION"])
    app.include_router(list_embedding_model_router.router, prefix=settings.api_common_prefix, tags=["EMBEDDING"])

    return app


app = create_app()

>> config.py
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    allowed_origins: str = "https://10.216.70.62,http://localhost:3000"
    allow_credentials: bool = True
    allow_methods: str = "GET,POST,OPTIONS,DELETE"
    allow_headers: str = "Authorization,Content-Type,Accept,Origin,User-Agent,X-Requested-With,X-API-Key,X-Session-Id,X-Usecase-Id,X-Correlation-ID,x-base-api-key"
    # database
    database_url: str
    platform_meta_db_name: str = "platform_meta_db"
    database_ssl_enabled: bool = False
    database_ssl_cert: str = ""
    database_ssl_key: str = ""
    database_ssl_root_cert: str = ""
    database_ssl_mode: str = ""
    pool_size: int = 10  # Number of connections to keep in pool
    max_overflow: int = 5  # Number of extra connections if pool exhausted
    pool_timeout: int = 30  # Wait up to 30 seconds for a connection
    pool_recycle: int = 1800  # Recycle connections every 30 minutes (to prevent stale connections)
    pool_pre_ping: bool = True  # Test connections before using (avoids broken connections)
    echo: bool = True
    engine_args: dict = {}
    pgvector_insert_default_batch_size: int = 1000
    # opik_api_key: str
    # opik_workspace: str
    # opik_project_name: str
    # opik_check_tls_certificate: bool
    # opik_url_override: str

    min_similarity_score: float = 0.75
    default_document_limit: int = 10

    api_common_prefix: str = "/v1/api"
    ws_common_prefix: str = "/v1/ws"
    health_check: str = "/health"

    default_model: str = "gemini-1.5-flash"
    default_model_embeddings: str = "BAAI/bge-m3"

    # llm endpoint
    base_api_url: str = "https://10.216.70.62/DEV/litellm"
    api_key_verification_endpoint: str = "/key/info"
    internal_api_url: str = "internal"
    # app specific
    deployment_env: str = "DEV"
    title: str = "GenAI Platform As a Service"
    description: str = "GenAI Platform As a Service"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "DEBUG"
    log_path: str = str(Path(__file__).parents[1] / "log" / "app.log")
    service_slug: str = "platform-services"
    validate_user_token_api: str = ""
    # completion endpoint
    completion_endpoint: str = "/text_completion"
    image_completion_endpoint: str = "/image_completion"
    playground_endpoints: str = "/playground"
    chatcompletion_endpoint: str = "/chatcompletion"

    # embedding endpoint
    embedding_endpoint: str = "/embeddings"

    # embedding endpoint
    search_endpoint: str = "/search"
    document_search_endpoint: str = "/v2/search_document"
    create_table_endpoint: str = "/v2/create_table"
    # rag endpoint
    rag_endpoint: str = "/rag"

    # document qna endpoint
    document_qna_endpoint: str = "/document_qna"

    create_collection: str = "/create_collection"
    delete_collection: str = "/delete_collection"
    list_embedding_model: str = "/list_embedding_models"

    vertexai_max_output_tokens: int = 65535
    vertexai_project: str = "hbl-dev-gcp-gen-ai-prj-spk-5a"
    vertexai_model: str = "gemini-2.5-flash-preview-05-20"
    vertexai_temperature: int = 0
    vertexai_top_p: int = 1
    vertexai_seed: int = 0
    vertexai_location: str = "global"

    # opik traces endpoints
    opik_traces: str = "/opik_traces"
    opik_traces_max_results: int = 5000

    # STT service URL
    stt_endpoint: str = "/speech-to-text"
    stt_api_url: str = (
        "https://10.216.70.62/DEV/voicing/stt/transcribe?backend=fast&model_name=large-v3&languages=en,hi,pa,mr"
    )
    stt_debugging_enabled: bool = False
    stt_ws_url: str = "wss://10.216.70.62/DEV/voicing/stt/ws?languages=en,hi,pa,mr"

    # TTS Service
    tts_endpoint: str = "/text-to-speech"
    tts_api_url: str = "https://10.216.70.62/DEV/voicing/tts/tts"
    tts_debugging_enabled: bool = False
    tts_ws_url: str = "wss://10.216.70.62/DEV/voicing/tts/tts"

    # chunk size
    chunk_size: int = 2048
    # chunk overlapping
    chunk_overlap: int = 256

    # store as a service endpoint
    storage_endpoint: str = "/index"
    document_index_endpoint: str = "/v2/index_document"
    # delete index
    delete_endpoint: str = "/delete_index"

    # file processing endpoint
    file_processing: str = "/file_processing"

    # guardrails specifics
    guardrails_endpoint: str = "https://10.216.70.62/DEV/guardrails/"
    guardrails_ssl_verify: bool = False
    guardrails_prompt_analyze_api: str = "api/v1/analyze/prompt"
    guardrails_output_analyze_api: str = "api/v1/analyze/output"

    default_api_key: str
    playground_api_key: str
    default_guardrail_id: str = "3"
    # redis
    redis_host: str = "localhost"
    redis_port: int = 6378
    ssl_ca_certs: str
    use_ssl: bool = True
    redis_auth_string: str = ""
    # qna utility
    cloud_service_provider: str = "gcp"
    generate_qna_endpoint: str = "/generate_qna"
    upload_file_limit: int = 10485760
    upload_bucket_name: str = "genai-ai-utilities-storage"
    voicing_upload_bucket_name: str = "voicing-genai-hdfc"
    upload_folder_name: str = "uploads"
    upload_object_endpoint: str = "/upload_object"
    # master api key
    master_api_key: str
    guardrail_enabled: bool = True
    prompt_hub_endpoint: str = "https://10.216.70.62/DEV/prompthub-service"
    prompt_hub_get_prompt_api: str = "/extenal/get-prompt-by-id/"
    prompt_hub_get_usecase_by_apikey: str = "/extenal/get-usecase-by-api-key"
    # max retries for rpm and tpm on lite llm
    max_retries: int = 0

    # Rate limiter configs
    playground_api_limit: str = "10/minute"
    pg_api_limit_exceed_message: str = "Too Many Request"
    pg_api_limit_exceed_status_code: int = 429

    scan_failed_message: str = (
        "Sorry, our safety filters blocked the assistant’s reply. Please rephrase or try a different question."
    )

    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", ".env"), env_file_encoding="utf-8", extra="allow"
    )

    def get(self, key: str, default: str | None = None) -> Optional[str]:
        return os.getenv(key) or default

    @property
    def all_env(self) -> dict:
        return dotenv_values(os.environ.get("ENV_FILE", ".env"))

    @property
    def verify(self) -> bool:
        return self.deployment_env == "PROD"

    @property
    def postgres_ssl_args(self) -> dict:
        self.engine_args = {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": self.pool_pre_ping,
            "echo": self.echo,
        }
        if self.database_ssl_enabled:
            ssl_args = {
                "sslcert": self.database_ssl_cert,
                "sslkey": self.database_ssl_key,
                "sslrootcert": self.database_ssl_root_cert,
                "sslmode": self.database_ssl_mode,
            }
            self.engine_args["connect_args"] = ssl_args
        return self.engine_args


@lru_cache()
def get_settings() -> Settings:
    return Settings()

>> collection_router.py
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select

from src.api.deps import check_embedding_model, validate_collection_access, validate_headers_and_api_key
from src.config import get_settings
from src.db.connection import create_session_platform
from src.db.platform_meta_tables import CollectionInfo, EmbeddingModels
from src.logging_config import Logger
from src.models.collection_payload import CreateCollections, DeleteCollection
from src.models.headers import HeaderInformation
from src.repository.document_repository import DocumentRepository, create_document_model

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()


@router.post(
    settings.create_collection,
    summary="Create a new collection in DB-B based on request.",
    status_code=status.HTTP_200_OK,
)
async def create_collection(
    request: CreateCollections,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
) -> dict:
    try:
        response = []

        with create_session_platform() as session_platform:
            for entry in request.collection_entries:
                model_dim = await check_embedding_model(entry.model)
                collection_uuid = str(uuid4())

                collection_info = CollectionInfo(
                    uuid=collection_uuid,
                    channel_id=entry.channel_id,
                    usecase_id=entry.usecase_id,
                    collection_name=entry.collection_name,
                    model=entry.model,
                )
                session_platform.add(collection_info)
                session_platform.commit()

                create_document_model(collection_uuid, embedding_dimensions=model_dim)
                logger.info(f"Created collection {collection_uuid} with dimensions {model_dim}")

                response.append({"collection_name": entry.collection_name, "uuid": collection_uuid})

        return {"collections": response}

    except Exception as e:
        logger.exception("Error creating collection.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    settings.delete_collection,
    summary="Delete a collection from DB-B and its metadata.",
    status_code=status.HTTP_200_OK,
)
async def delete_collection(
    request: DeleteCollection,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
) -> dict:
    await validate_collection_access(header_information.x_base_api_key, request.collection_uid)
    try:
        with create_session_platform() as session_platform:
            collection_query = session_platform.execute(
                select(CollectionInfo).where(
                    CollectionInfo.uuid == request.collection_uid)
            ).scalar_one_or_none()

            if not collection_query:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Collection with UID '{request.collection_uid}' not found.",
                )

            document_repository = DocumentRepository(request.collection_uid, create_if_not_exists=False)
            if not document_repository.check_table_exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Collection table '{request.collection_uid}' does not exist in the database.",
                )
            document_repository.delete_collection()

            session_platform.execute(
                delete(CollectionInfo).where(
                    CollectionInfo.uuid == request.collection_uid)
            )
            session_platform.commit()

            logger.info(f"Deleted collection with UID '{request.collection_uid}'")

        return {
            "message": "Collection has been deleted.",
            "collection": request.collection_uid}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting collection.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

>> deps.py
from functools import lru_cache
from typing import Any, Optional

import httpx
from fastapi import Depends, Header, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from google.genai import Client
from openai import AsyncOpenAI
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select

from src.config import get_settings
from src.db.connection import create_session_platform
from src.db.platform_meta_tables import CollectionInfo, EmbeddingModels
from src.integrations.open_ai_sdk import OpenAISdk
from src.integrations.redis_chatbot_memory import RedisShortTermMemory
from src.models.headers import HeaderInformation, InternalHeaderInformation
from src.services.embedding_service import EmbeddingService
from src.services.file_upload_service import FileUploadService
from src.services.vertexai_conversation_service import VertexAIConversationService
from src.utility.file_io import FileIO

settings = get_settings()


def get_rate_limiter() -> Limiter:
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=f"rediss://:{settings.redis_auth_string}@{settings.redis_host}:{settings.redis_port}/0?"
        f"ssl_ca_certs={settings.ssl_ca_certs}",
    )
    return limiter


async def validate_headers_and_api_key(
    session_id: Optional[str] = Header(None, alias="x-session-id"),
    x_base_api_key: Optional[str] = Header(..., alias="x-base-api-key"),
) -> HeaderInformation:
    missing_headers = []
    if not session_id:
        missing_headers.append("x-session-id")
    if not x_base_api_key:
        missing_headers.append("x-base-api-key")

    if missing_headers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required headers: {', '.join(missing_headers)}",
        )
    validation_url = f"{settings.base_api_url}{settings.api_key_verification_endpoint}"
    verify = False if settings.deployment_env != "PROD" else True
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.get(
            validation_url,
            params={"key": x_base_api_key},
            headers={"Authorization": f"Bearer {settings.master_api_key}"},
        )
    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")

    return HeaderInformation(x_session_id=session_id, x_base_api_key=x_base_api_key)


def build_openai_sdk(api_key: str) -> OpenAISdk:
    http_client = httpx.AsyncClient(http2=True, verify=settings.verify)
    openai_async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.base_api_url,
        http_client=http_client,
        max_retries=settings.max_retries,
    )
    return OpenAISdk(openai_async_client=openai_async_client)


@lru_cache()
def get_openai_service_internal() -> OpenAISdk:
    return build_openai_sdk(settings.default_api_key)


@lru_cache()
def get_openai_service(
    header_info: HeaderInformation = Depends(validate_headers_and_api_key),
) -> OpenAISdk:
    return build_openai_sdk(header_info.x_base_api_key)


@lru_cache()
def get_memory_client() -> RedisShortTermMemory:
    return RedisShortTermMemory(host=settings.redis_host, port=settings.redis_port)


@lru_cache()
def get_embedding_service(open_ai_sdk: OpenAISdk = Depends(get_openai_service)) -> EmbeddingService:
    return EmbeddingService(open_ai_sdk=open_ai_sdk)


def get_vertexai_service() -> VertexAIConversationService:
    client = Client(
        vertexai=True,
        project=settings.vertexai_project,
        location=settings.vertexai_location,
    )
    return VertexAIConversationService(client=client)


@lru_cache()
def get_file_io_service() -> FileIO:
    return FileIO()


@lru_cache()
def get_file_upload_service() -> FileUploadService:
    return FileUploadService()


user_token_header = APIKeyHeader(name="token", auto_error=False, scheme_name="TOKEN")


async def validate_request_user_token_params(
    session_id: Optional[str] = Header(default=None, alias="x-session-id"),
    x_user_token: Optional[str] = Header(..., alias="token"),
) -> InternalHeaderInformation:
    missing_header = []
    if not session_id:
        missing_header.append("x-session-id")
    if not x_user_token:
        missing_header.append("token")

    if missing_header:
        raise HTTPException(status_code=400, detail=f"Missing header(s): {', '.join(missing_header)}")

    return InternalHeaderInformation(x_session_id=session_id, x_user_token=x_user_token)


async def validate_user_token_api_call(x_user_token: str) -> str:
    async with httpx.AsyncClient(verify=False) as client:
        key_response = await client.get(
            settings.validate_user_token_api,
            headers={"Authorization": f"{x_user_token}"},
        )

    if key_response.status_code == 200:
        json_data: dict[str, Any] = key_response.json()
        user_data = json_data.get("data")

        if isinstance(user_data, str):
            return user_data
        else:
            raise HTTPException(status_code=403, detail="Invalid user token: missing data field")

    raise HTTPException(status_code=403, detail="Invalid user token")


async def check_embedding_model(modelname: str) -> int:
    with create_session_platform() as sess_plat:
        model_query = sess_plat.execute(
            select(EmbeddingModels).where(EmbeddingModels.model_name == modelname)
        ).scalar_one_or_none()
        if not model_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{modelname}' not found in embedding_models table.",
            )
        return int(model_query.dimensions)


async def validate_collection_access(base_api_key: str, collection_name: str) -> None:
    validation_url = f"{settings.prompt_hub_endpoint}{settings.prompt_hub_get_usecase_by_apikey}"
    verify = False if settings.deployment_env != "PROD" else True
    async with httpx.AsyncClient(verify=verify) as client:
        response = await client.get(
            validation_url,
            headers={"lite-llm-api-key": base_api_key},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")

        channel_id = response.json()["data"]["channel_id"]
        team_id = response.json()["data"]["team_id"]

    with create_session_platform() as sess_plat:
        collection_query = sess_plat.execute(
            select(CollectionInfo).where(CollectionInfo.uuid == collection_name)
        ).scalar_one_or_none()
        if not collection_query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found.",
            )
        elif str(collection_query.usecase_id) != team_id or str(collection_query.channel_id) != channel_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not authorized to access this collection.",
            )

>> connections.py
from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.db.base import BaseDBA

settings = get_settings()

ssl_args = {
    "sslcert": settings.database_ssl_cert,
    "sslkey": settings.database_ssl_key,
    "sslrootcert": settings.database_ssl_root_cert,
    "sslmode": settings.database_ssl_mode,
}
engine_args: dict[str, Any] = {
    "pool_size": 10,  # Number of connections to keep in pool
    "max_overflow": 5,  # Number of extra connections if pool exhausted
    "pool_timeout": 30,  # Wait up to 30 seconds for a connection
    "pool_recycle": 1800,  # Recycle connections every 30 minutes (to prevent stale connections)
    "pool_pre_ping": True,  # Test connections before using (avoids broken connections)
    "echo": True,
}
if settings.database_ssl_enabled:
    engine_args["connect_args"] = ssl_args
# Create the SQLAlchemy engine with connection pooling options
engine = create_engine(settings.database_url, **engine_args)
database_url_platform = settings.database_url.rsplit("/", 1)[0] + f"/{settings.platform_meta_db_name}"
engine_platform_meta_db = create_engine(database_url_platform, **engine_args)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
SessionLocal_PlatformMeta = sessionmaker(autocommit=False, autoflush=False, bind=engine_platform_meta_db)


# TODO future enhancement : make it Async
@contextmanager
def create_session() -> Generator[Session, None, None]:
    """Context manager for DB session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def create_session_platform() -> Generator[Session, None, None]:
    """Context manager for DB session."""
    session_platform = SessionLocal_PlatformMeta()
    try:
        yield session_platform
        session_platform.commit()
    except Exception:
        session_platform.rollback()
        raise
    finally:
        session_platform.close()


def initialize_platform_db() -> None:
    try:
        BaseDBA.metadata.create_all(bind=engine_platform_meta_db)
    except Exception as e:
        print(f"Error initializing Platform Meta Tables: {str(e)}")

>> platform_meta_tables.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from src.db.base import BaseDBA


class CollectionInfo(BaseDBA):  # type: ignore
    __tablename__ = "collection_info"

    uuid = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    channel_id = Column(Integer, nullable=False)
    usecase_id = Column(String, nullable=False)
    collection_name = Column(String, nullable=False)
    model = Column(String, nullable=False)


class EmbeddingModels(BaseDBA):  # type: ignore
    __tablename__ = "embedding_models"

    model_name = Column(String, primary_key=True, nullable=False)
    dimensions = Column(Integer, nullable=False)

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




I want you to go through them very very thoroghly and check if the following test cases holds good.

from fastapi import HTTPException
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.api.deps import validate_headers_and_api_key
from src.models.headers import HeaderInformation


# Fixtures
@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_headers():
    return {"x-session-id": "test-session", "x-base-api-key": "test-api-key"}


@pytest.fixture
def valid_header_info():
    return HeaderInformation(x_session_id="test-session", x_base_api_key="test-api-key", team_id="test_team")


@pytest.fixture
def override_dependencies(valid_header_info):
    app.dependency_overrides[validate_headers_and_api_key] = lambda: valid_header_info
    yield
    app.dependency_overrides = {}


@pytest.fixture
def create_collection_payload():
    return {
        "collection_entries": [
            {
                "channel_id": 1,
                "usecase_id": "test-usecase",
                "collection_name": "test-collection",
                "model": "test-model",
            }
        ]
    }


@pytest.fixture
def delete_collection_payload():
    return {"collection_uid": "test-collection-uuid"}


def test_create_collection_success(
        client, create_collection_payload, valid_headers, override_dependencies, mocker
):
    mocker.patch("src.api.deps.check_embedding_model", return_value=128)

    mocker.patch("src.repository.document_repository.create_document_model", return_value=None)

    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "collections" in response_data
    assert response_data["collections"][0]["collection_name"] == "test-collection"


def test_create_collection_model_not_found(
        client, create_collection_payload, valid_headers, override_dependencies, mocker
):
    mocker.patch(
        "src.api.deps.check_embedding_model",
        side_effect=HTTPException(
            status_code=400,
            detail="Model 'test-model' not found in embedding_models table."
        )
    )

    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert response.status_code == 400
    assert response.json()["error"] == "Model 'test-model' not found in embedding_models table."





<h2 style='color:green;'>##System##</h2>
The test cases you've provided seem to be well-designed, but let's go through each one step-by-step to ensure the code implementation satisfies the expectations of the test cases. I'll review each test case in the context of the provided code implementation.

---

### **Test Case 1: `test_create_collection_success`**

#### **Code Behavior**
1. **Flow:**
   - **Endpoint:** `/v1/api/create_collection`
   - **Steps in `create_collection` endpoint:**
     - `check_embedding_model(entry.model)` is called to validate the model.
     - A collection UUID is generated.
     - Metadata about the collection is added to the `CollectionInfo` table.
     - A document model is created using the `create_document_model` function.
     - The function returns a success response with the collection details.

2. **Mocks Used in Test:**
   - `check_embedding_model` is mocked to return a valid embedding dimension (`128`).
   - `create_document_model` is mocked to do nothing (`return_value=None`).

3. **Assertions in Test:**
   - The API call should return a `200` status.
   - The response should contain the collection name and UUID.

#### **Implementation Validation**
- The `check_embedding_model` function:
  - It queries the `EmbeddingModels` table for the model.
  - If the model exists, it returns its dimension (`128` in this case).
  - **Mocked correctly for the test.**

- The `create_document_model` function:
  - It creates a document table with the specified `embedding_dimensions`.
  - If the table already exists, it reuses it.
  - **Mocked to skip table creation for simplicity in this test.**

- The `create_collection` function:
  - Handles the flow as expected, including generating a UUID, adding metadata, and returning a proper response.
  - No issues were found.

#### **Conclusion**
- This test case holds good.
- The expected behavior matches the implementation.

---

### **Test Case 2: `test_create_collection_model_not_found`**

#### **Code Behavior**
1. **Flow:**
   - **Endpoint:** `/v1/api/create_collection`
   - During the `check_embedding_model` call:
     - If the model is not found in the `EmbeddingModels` table, an `HTTPException` with status code `400` is raised.

2. **Mocks Used in Test:**
   - `check_embedding_model` is mocked to raise an `HTTPException` with the message `"Model 'test-model' not found in embedding_models table."`

3. **Assertions in Test:**
   - The API call should return a `400` status.
   - The response body should contain the error message `"Model 'test-model' not found in embedding_models table."`

#### **Implementation Validation**
- The `check_embedding_model` function:
  - Queries the `EmbeddingModels` table for the model.
  - If the model is not found, it raises the exact `HTTPException` as mocked in the test.
  - **Mocked correctly for the test.**

- The `create_collection` function:
  - Calls `check_embedding_model` before proceeding.
  - If `check_embedding_model` raises an exception, it propagates the exception to the client.
  - **Handles the exception as expected.**

#### **Conclusion**
- This test case holds good.
- The expected behavior matches the implementation.

---

### **Additional Observations**
Here are some additional points to ensure robustness and proper functionality of the code:

1. **Validation of Headers (`validate_headers_and_api_key`):**
   - The `validate_headers_and_api_key` dependency validates the `x-session-id` and `x-base-api-key` headers.
   - If the headers are missing, an `HTTPException` with status `422` is raised.
   - The test cases provided assume valid headers (via the `valid_headers` and `override_dependencies` fixtures).
   - Consider testing invalid or missing headers in additional test cases.

2. **Database Session Management (`create_session_platform`):**
   - The `create_session_platform` context manager ensures proper session management.
   - If there are database connection issues, the endpoint will raise an appropriate exception.
   - This behavior is not explicitly tested in the provided cases but should be considered for edge cases.

3. **Model Caching (`create_document_model`):**
   - The `_document_model_cache` ensures that models are not recreated unnecessarily.
   - This is an optimization and does not impact the correctness of the test cases.

4. **Error Handling in `delete_collection`:**
   - Similar exception handling exists for the `delete_collection` endpoint.
   - For consistency, consider adding test cases for this endpoint as well.

---

### **Recommended Additional Test Cases**
1. **Missing Required Headers:**
   ```python
   def test_create_collection_missing_headers(client, create_collection_payload):
       response = client.post("/v1/api/create_collection", json=create_collection_payload)
       assert response.status_code == 422
       assert "Missing required headers" in response.json()["detail"]
   ```

2. **Database Connection Issues:**
   ```python
   def test_create_collection_db_error(client, create_collection_payload, valid_headers, override_dependencies, mocker):
       mocker.patch("src.db.connection.create_session_platform", side_effect=Exception("DB connection error"))
       response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
       assert response.status_code == 500
       assert "DB connection error" in response.json()["detail"]
   ```

3. **Delete Collection Success:**
   ```python
   def test_delete_collection_success(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
       mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=True)
       mocker.patch("src.repository.document_repository.DocumentRepository.delete_collection", return_value=1)
       
       response = client.delete("/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
       assert response.status_code == 200
       assert response.json()["message"] == "Collection has been deleted."
   ```

4. **Delete Collection Not Found:**
   ```python
   def test_delete_collection_not_found(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
       mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=False)
       
       response = client.delete("/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
       assert response.status_code == 404
       assert "Collection table" in response.json()["detail"]
   ```

---

### **Final Thoughts**
The provided test cases are well-constructed and validated against the implementation. With the additional test cases suggested above, you can ensure broader coverage and robustness of the service.

<h2 style='color:cyan;'>##User##</h2>


from fastapi import HTTPException
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.api.deps import validate_headers_and_api_key
from src.models.headers import HeaderInformation


# Fixtures
@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_headers():
    return {"x-session-id": "test-session", "x-base-api-key": "test-api-key"}


@pytest.fixture
def valid_header_info():
    return HeaderInformation(x_session_id="test-session", x_base_api_key="test-api-key", team_id="test_team")


@pytest.fixture
def override_dependencies(valid_header_info):
    app.dependency_overrides[validate_headers_and_api_key] = lambda: valid_header_info
    yield
    app.dependency_overrides = {}


@pytest.fixture
def create_collection_payload():
    return {
        "collection_entries": [
            {
                "channel_id": 1,
                "usecase_id": "test-usecase",
                "collection_name": "test-collection",
                "model": "test-model",
            }
        ]
    }


@pytest.fixture
def delete_collection_payload():
    return {"collection_uid": "test-collection-uuid"}


def test_create_collection_success(
        client, create_collection_payload, valid_headers, override_dependencies, mocker
):
    mocker.patch("src.api.deps.check_embedding_model", return_value=128)

    mocker.patch("src.repository.document_repository.create_document_model", return_value=None)

    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert response.status_code == 200
    response_data = response.json()
    assert "collections" in response_data
    assert response_data["collections"][0]["collection_name"] == "test-collection"


def test_create_collection_model_not_found(
        client, create_collection_payload, valid_headers, override_dependencies, mocker
):
    mocker.patch(
        "src.api.deps.check_embedding_model",
        side_effect=HTTPException(
            status_code=400,
            detail="Model 'test-model' not found in embedding_models table."
        )
    )

    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert response.status_code == 400
    assert response.json()["error"] == "Model 'test-model' not found in embedding_models table."

def test_create_collection_db_error(client, create_collection_payload, valid_headers, override_dependencies, mocker):
    mocker.patch("src.db.connection.create_session_platform", side_effect=Exception("DB connection error"))
    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    assert response.status_code == 500
    assert "DB connection error" in response.json()["detail"]


def test_delete_collection_success(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
    mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=True)
    mocker.patch("src.repository.document_repository.DocumentRepository.delete_collection", return_value=1)

    response = client.delete("/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
    assert response.status_code == 200
    assert response.json()["message"] == "Collection has been deleted."


def test_delete_collection_not_found(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
    mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=False)

    response = client.delete("/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
    assert response.status_code == 404
    assert "Collection table" in response.json()["detail"]



(genai-platform-services) (base) epfn119476@25C-LTP-H-39281 genai_platform_services % uv run pytest tests/unit/api/router/test_collection_router.py --cov=src --cov-report=term
================================================================================ test session starts ================================================================================
platform darwin -- Python 3.11.12, pytest-8.3.5, pluggy-1.5.0
rootdir: /Users/epfn119476/Documents/HDFC/genai_platform_services
configfile: pyproject.toml
plugins: env-1.1.5, asyncio-0.25.3, anyio-4.8.0, langsmith-0.3.42, mock-3.14.0, cov-6.1.1, opik-1.7.33
asyncio: mode=Mode.AUTO, asyncio_default_fixture_loop_scope=function
collected 5 items                                                                                                                                                                   

tests/unit/api/router/test_collection_router.py FFFFF
ERROR: Coverage failure: total of 50 is less than fail-under=75
                                                                                                                                                                              [100%]

===================================================================================== FAILURES ======================================================================================
__________________________________________________________________________ test_create_collection_success ___________________________________________________________________________

client = <starlette.testclient.TestClient object at 0x10caa6190>
create_collection_payload = {'collection_entries': [{'channel_id': 1, 'collection_name': 'test-collection', 'model': 'test-model', 'usecase_id': 'test-usecase'}]}
valid_headers = {'x-base-api-key': 'test-api-key', 'x-session-id': 'test-session'}, override_dependencies = None, mocker = <pytest_mock.plugin.MockerFixture object at 0x10bb5ebd0>

    def test_create_collection_success(
            client, create_collection_payload, valid_headers, override_dependencies, mocker
    ):
        mocker.patch("src.api.deps.check_embedding_model", return_value=128)
    
        mocker.patch("src.repository.document_repository.create_document_model", return_value=None)
    
        response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
>       assert response.status_code == 200
E       assert 500 == 200
E        +  where 500 = <Response [500 Internal Server Error]>.status_code

tests/unit/api/router/test_collection_router.py:59: AssertionError
------------------------------------------------------------------------------- Captured stdout call --------------------------------------------------------------------------------
2025-06-11 22:39:53,030 INFO sqlalchemy.engine.Engine select pg_catalog.version()
2025-06-11 22:39:53,031 INFO sqlalchemy.engine.Engine [raw sql] {}
2025-06-11 22:39:53,103 INFO sqlalchemy.engine.Engine select current_schema()
2025-06-11 22:39:53,104 INFO sqlalchemy.engine.Engine [raw sql] {}
2025-06-11 22:39:53,173 INFO sqlalchemy.engine.Engine show standard_conforming_strings
2025-06-11 22:39:53,174 INFO sqlalchemy.engine.Engine [raw sql] {}
2025-06-11 22:39:53,243 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2025-06-11 22:39:53,251 INFO sqlalchemy.engine.Engine SELECT embedding_models.model_name, embedding_models.dimensions 
FROM embedding_models 
WHERE embedding_models.model_name = %(model_name_1)s
2025-06-11 22:39:53,251 INFO sqlalchemy.engine.Engine [generated in 0.00049s] {'model_name_1': 'test-model'}
2025-06-11 22:39:53,321 INFO sqlalchemy.engine.Engine ROLLBACK
[ERROR] [2025-06-11 22:39:53,356] [src.api.routers.collection_router] [55] Error creating collection.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.
[ERROR] [2025-06-11 22:39:53,361] [src.main] [137] Http error: 500: 400: Model 'test-model' not found in embedding_models table.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 56, in create_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
fastapi.exceptions.HTTPException: 500: 400: Model 'test-model' not found in embedding_models table.
--------------------------------------------------------------------------------- Captured log call ---------------------------------------------------------------------------------
INFO     sqlalchemy.engine.Engine:base.py:1843 select pg_catalog.version()
INFO     sqlalchemy.engine.Engine:base.py:1843 [raw sql] {}
INFO     sqlalchemy.engine.Engine:base.py:1843 select current_schema()
INFO     sqlalchemy.engine.Engine:base.py:1843 [raw sql] {}
INFO     sqlalchemy.engine.Engine:base.py:1843 show standard_conforming_strings
INFO     sqlalchemy.engine.Engine:base.py:1843 [raw sql] {}
INFO     sqlalchemy.engine.Engine:base.py:2699 BEGIN (implicit)
INFO     sqlalchemy.engine.Engine:base.py:1843 SELECT embedding_models.model_name, embedding_models.dimensions 
FROM embedding_models 
WHERE embedding_models.model_name = %(model_name_1)s
INFO     sqlalchemy.engine.Engine:base.py:1843 [generated in 0.00049s] {'model_name_1': 'test-model'}
INFO     sqlalchemy.engine.Engine:base.py:2702 ROLLBACK
ERROR    src.api.routers.collection_router:collection_router.py:55 Error creating collection.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.
ERROR    src.main:main.py:137 Http error: 500: 400: Model 'test-model' not found in embedding_models table.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 56, in create_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
fastapi.exceptions.HTTPException: 500: 400: Model 'test-model' not found in embedding_models table.
______________________________________________________________________ test_create_collection_model_not_found _______________________________________________________________________

client = <starlette.testclient.TestClient object at 0x10ca9e910>
create_collection_payload = {'collection_entries': [{'channel_id': 1, 'collection_name': 'test-collection', 'model': 'test-model', 'usecase_id': 'test-usecase'}]}
valid_headers = {'x-base-api-key': 'test-api-key', 'x-session-id': 'test-session'}, override_dependencies = None, mocker = <pytest_mock.plugin.MockerFixture object at 0x10cb0bf50>

    def test_create_collection_model_not_found(
            client, create_collection_payload, valid_headers, override_dependencies, mocker
    ):
        mocker.patch(
            "src.api.deps.check_embedding_model",
            side_effect=HTTPException(
                status_code=400,
                detail="Model 'test-model' not found in embedding_models table."
            )
        )
    
        response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
>       assert response.status_code == 400
E       assert 500 == 400
E        +  where 500 = <Response [500 Internal Server Error]>.status_code

tests/unit/api/router/test_collection_router.py:77: AssertionError
------------------------------------------------------------------------------- Captured stdout call --------------------------------------------------------------------------------
2025-06-11 22:39:53,455 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2025-06-11 22:39:53,455 INFO sqlalchemy.engine.Engine SELECT embedding_models.model_name, embedding_models.dimensions 
FROM embedding_models 
WHERE embedding_models.model_name = %(model_name_1)s
2025-06-11 22:39:53,456 INFO sqlalchemy.engine.Engine [cached since 0.2051s ago] {'model_name_1': 'test-model'}
2025-06-11 22:39:53,562 INFO sqlalchemy.engine.Engine ROLLBACK
[ERROR] [2025-06-11 22:39:53,597] [src.api.routers.collection_router] [55] Error creating collection.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.
[ERROR] [2025-06-11 22:39:53,599] [src.main] [137] Http error: 500: 400: Model 'test-model' not found in embedding_models table.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 56, in create_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
fastapi.exceptions.HTTPException: 500: 400: Model 'test-model' not found in embedding_models table.
--------------------------------------------------------------------------------- Captured log call ---------------------------------------------------------------------------------
INFO     sqlalchemy.engine.Engine:base.py:2699 BEGIN (implicit)
INFO     sqlalchemy.engine.Engine:base.py:1843 SELECT embedding_models.model_name, embedding_models.dimensions 
FROM embedding_models 
WHERE embedding_models.model_name = %(model_name_1)s
INFO     sqlalchemy.engine.Engine:base.py:1843 [cached since 0.2051s ago] {'model_name_1': 'test-model'}
INFO     sqlalchemy.engine.Engine:base.py:2702 ROLLBACK
ERROR    src.api.routers.collection_router:collection_router.py:55 Error creating collection.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.
ERROR    src.main:main.py:137 Http error: 500: 400: Model 'test-model' not found in embedding_models table.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 56, in create_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
fastapi.exceptions.HTTPException: 500: 400: Model 'test-model' not found in embedding_models table.
__________________________________________________________________________ test_create_collection_db_error __________________________________________________________________________

client = <starlette.testclient.TestClient object at 0x10cabac10>
create_collection_payload = {'collection_entries': [{'channel_id': 1, 'collection_name': 'test-collection', 'model': 'test-model', 'usecase_id': 'test-usecase'}]}
valid_headers = {'x-base-api-key': 'test-api-key', 'x-session-id': 'test-session'}, override_dependencies = None, mocker = <pytest_mock.plugin.MockerFixture object at 0x109317910>

    def test_create_collection_db_error(client, create_collection_payload, valid_headers, override_dependencies, mocker):
        mocker.patch("src.db.connection.create_session_platform", side_effect=Exception("DB connection error"))
        response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
        assert response.status_code == 500
>       assert "DB connection error" in response.json()["detail"]
E       KeyError: 'detail'

tests/unit/api/router/test_collection_router.py:84: KeyError
------------------------------------------------------------------------------- Captured stdout call --------------------------------------------------------------------------------
2025-06-11 22:39:53,653 INFO sqlalchemy.engine.Engine BEGIN (implicit)
2025-06-11 22:39:53,654 INFO sqlalchemy.engine.Engine SELECT embedding_models.model_name, embedding_models.dimensions 
FROM embedding_models 
WHERE embedding_models.model_name = %(model_name_1)s
2025-06-11 22:39:53,654 INFO sqlalchemy.engine.Engine [cached since 0.4037s ago] {'model_name_1': 'test-model'}
2025-06-11 22:39:53,738 INFO sqlalchemy.engine.Engine ROLLBACK
[ERROR] [2025-06-11 22:39:53,772] [src.api.routers.collection_router] [55] Error creating collection.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.
[ERROR] [2025-06-11 22:39:53,773] [src.main] [137] Http error: 500: 400: Model 'test-model' not found in embedding_models table.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 56, in create_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
fastapi.exceptions.HTTPException: 500: 400: Model 'test-model' not found in embedding_models table.
--------------------------------------------------------------------------------- Captured log call ---------------------------------------------------------------------------------
INFO     sqlalchemy.engine.Engine:base.py:2699 BEGIN (implicit)
INFO     sqlalchemy.engine.Engine:base.py:1843 SELECT embedding_models.model_name, embedding_models.dimensions 
FROM embedding_models 
WHERE embedding_models.model_name = %(model_name_1)s
INFO     sqlalchemy.engine.Engine:base.py:1843 [cached since 0.4037s ago] {'model_name_1': 'test-model'}
INFO     sqlalchemy.engine.Engine:base.py:2702 ROLLBACK
ERROR    src.api.routers.collection_router:collection_router.py:55 Error creating collection.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.
ERROR    src.main:main.py:137 Http error: 500: 400: Model 'test-model' not found in embedding_models table.
Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 34, in create_collection
    model_dim = await check_embedding_model(entry.model)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/deps.py", line 161, in check_embedding_model
    raise HTTPException(
fastapi.exceptions.HTTPException: 400: Model 'test-model' not found in embedding_models table.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/epfn119476/Documents/HDFC/genai_platform_services/src/api/routers/collection_router.py", line 56, in create_collection
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
fastapi.exceptions.HTTPException: 500: 400: Model 'test-model' not found in embedding_models table.
__________________________________________________________________________ test_delete_collection_success ___________________________________________________________________________

client = <starlette.testclient.TestClient object at 0x10cabb090>, delete_collection_payload = {'collection_uid': 'test-collection-uuid'}
valid_headers = {'x-base-api-key': 'test-api-key', 'x-session-id': 'test-session'}, override_dependencies = None, mocker = <pytest_mock.plugin.MockerFixture object at 0x10bb7dd90>

    def test_delete_collection_success(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
        mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=True)
        mocker.patch("src.repository.document_repository.DocumentRepository.delete_collection", return_value=1)
    
>       response = client.delete("/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
E       TypeError: TestClient.delete() got an unexpected keyword argument 'json'

tests/unit/api/router/test_collection_router.py:91: TypeError
_________________________________________________________________________ test_delete_collection_not_found __________________________________________________________________________

client = <starlette.testclient.TestClient object at 0x10cbe3f50>, delete_collection_payload = {'collection_uid': 'test-collection-uuid'}
valid_headers = {'x-base-api-key': 'test-api-key', 'x-session-id': 'test-session'}, override_dependencies = None, mocker = <pytest_mock.plugin.MockerFixture object at 0x10cc0be90>

    def test_delete_collection_not_found(client, delete_collection_payload, valid_headers, override_dependencies, mocker):
        mocker.patch("src.repository.document_repository.DocumentRepository.check_table_exists", return_value=False)
    
>       response = client.delete("/v1/api/delete_collection", json=delete_collection_payload, headers=valid_headers)
E       TypeError: TestClient.delete() got an unexpected keyword argument 'json'

tests/unit/api/router/test_collection_router.py:99: TypeError
================================================================================= warnings summary ==================================================================================
src/db/base.py:3
  /Users/epfn119476/Documents/HDFC/genai_platform_services/src/db/base.py:3: MovedIn20Warning: The ``declarative_base()`` function is now available as sqlalchemy.orm.declarative_base(). (deprecated since: 2.0) (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)
    BaseDBA = declarative_base()

.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295
.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295
.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295
  /Users/epfn119476/Documents/HDFC/genai_platform_services/.venv/lib/python3.11/site-packages/pydantic/_internal/_config.py:295: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)

<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================================================== tests coverage ===================================================================================
_________________________________________________________________ coverage: platform darwin, python 3.11.12-final-0 _________________________________________________________________

Name                                                 Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------------
src/__init__.py                                          0      0   100%
src/api/__init__.py                                      0      0   100%
src/api/deps.py                                        105     59    44%   28-33, 40-62, 66-73, 78, 85, 90, 95, 99-104, 109, 114, 124-133, 137-152, 165, 169-192
src/api/routers/__init__.py                              0      0   100%
src/api/routers/chatcompletion_router.py                46     26    43%   42-96
src/api/routers/collection_router.py                    53     27    49%   35-52, 68-106
src/api/routers/document_store_router.py                48     26    46%   50-64, 86-97, 117-132
src/api/routers/embeddings_router.py                    17      2    88%   30-31
src/api/routers/file_processing_router.py               35     19    46%   35-56
src/api/routers/internal/__init__.py                     0      0   100%
src/api/routers/internal/chatcompletion_router.py       47     27    43%   42-106
src/api/routers/internal/document_qna_router.py         15      3    80%   28-30
src/api/routers/internal/export_traces_router.py        31     19    39%   28-74
src/api/routers/internal/file_processing_router.py      35     19    46%   35-56
src/api/routers/internal/generate_qna_router.py         34     15    56%   42-74
src/api/routers/internal/playground_router.py           33     15    55%   40-71
src/api/routers/internal/upload_file_router.py          23     11    52%   25-35
src/api/routers/list_embedding_model_router.py          25     12    52%   22-43
src/api/routers/rag_router.py                           35     15    57%   41-71
src/api/routers/speech_to_text_router.py                36     20    44%   32-53
src/api/routers/text_to_speech_router.py                34     17    50%   33-62
src/api/routers/v2/__init__.py                           0      0   100%
src/api/routers/v2/document_store_router_v2.py          71     47    34%   42-75, 98-116, 132-151
src/chunkers/__init__.py                                 0      0   100%
src/chunkers/recursive_chunker.py                       15     15     0%   1-21
src/client/opik_client.py                                7      1    86%   13
src/config.py                                          129      7    95%   170, 174, 182-198
src/db/__init__.py                                       0      0   100%
src/db/base.py                                           2      0   100%
src/db/connection.py                                    41     14    66%   27, 42-50, 59, 68-71
src/db/platform_meta_tables.py                          14      0   100%
src/exception/__init__.py                                0      0   100%
src/exception/document_store_exception.py               10      0   100%
src/exception/exceptions.py                              4      0   100%
src/exception/rag_exception.py                           8      0   100%
src/exception/scanner_exceptions.py                      8      6    25%   3-8
src/integrations/__init__.py                             0      0   100%
src/integrations/cloud_storage.py                       72     61    15%   12-22, 25-36, 39-53, 56-70, 73-100
src/integrations/open_ai_sdk.py                         79     59    25%   28-30, 39-106, 116-159, 166-169
src/integrations/redis_chatbot_memory.py                42     29    31%   20-29, 32, 35-44, 47-54, 57-77, 80-85, 89
src/logging_config.py                                   49     13    73%   37-39, 43-44, 48-49, 53-54, 58-59, 63-64
src/main.py                                            122     26    79%   56-63, 84, 103, 120-121, 126-127, 142-143, 150-151, 158-159, 166-167, 174-175, 186, 190
src/models/__init__.py                                   0      0   100%
src/models/collection_payload.py                        11      0   100%
src/models/completion_payload.py                        53      0   100%
src/models/create_table_payload.py                      10      0   100%
src/models/document_qna.py                               7      0   100%
src/models/embeddings_payload.py                        12      0   100%
src/models/generate_qna_payload.py                      18      0   100%
src/models/headers.py                                   11      0   100%
src/models/indexing_payload.py                          12      0   100%
src/models/rag_payload.py                               23      0   100%
src/models/registry_metadata.py                         11      0   100%
src/models/search_request.py                            28     28     0%   1-65
src/models/storage_payload.py                           56      0   100%
src/models/tts_payload.py                               26      0   100%
src/models/upload_object_payload.py                     43     16    63%   42-59, 63-69, 72
src/prompts/__init__.py                                  2      0   100%
src/repository/__init__.py                               0      0   100%
src/repository/document_repository.py                  114     72    37%   51, 57-87, 92-96, 99, 102-103, 107-116, 120-128, 132-133, 144-179, 183-186, 189
src/repository/registry/__init__.py                      2      0   100%
src/repository/registry/database.py                     22      6    73%   10, 14, 24, 28, 32, 36
src/repository/registry/pgvector.py                     71     46    35%   23-25, 30-31, 38-48, 51-53, 62-79, 82-83, 86, 89-100, 111, 116
src/services/__init__.py                                 0      0   100%
src/services/abstract_document_store.py                 13      3    77%   13, 23, 28
src/services/embedding_service.py                       10      4    60%   9, 12-14
src/services/file_upload_service.py                    112     92    18%   20-22, 25-67, 71-86, 89-97, 101-106, 109-151
src/services/pdf_processing_service.py                  20     20     0%   1-26
src/services/pgvector_document_store.py                 57     40    30%   29-31, 34-46, 49-70, 73-96
src/services/rag_service.py                             45     30    33%   19-20, 23-90, 100-116
src/services/vertexai_conversation_service.py           32     20    38%   15, 18-25, 28-65
src/utility/__init__.py                                  0      0   100%
src/utility/dynamic_model_utils.py                      47     32    32%   53-95
src/utility/file_io.py                                  50     34    32%   20-26, 30-62
src/utility/guardrails.py                               39     26    33%   15, 23-24, 31-42, 51-58, 68-79
src/utility/pdf_helpers.py                              18     13    28%   8-18, 22-26
src/utility/registry.py                                 32     13    59%   18, 25-29, 32-35, 38, 42-47
src/utility/registry_initializer.py                     37     24    35%   16-24, 28-43, 47
src/utility/utils.py                                    54     45    17%   17-65, 72-79
src/websockets/stt_websocket_router.py                  46     34    26%   19-68
src/websockets/tts_websocket_router.py                  92     77    16%   19-25, 30-129
----------------------------------------------------------------------------------
TOTAL                                                 2561   1285    50%
FAIL Required test coverage of 75.0% not reached. Total coverage: 49.82%
============================================================================== short test summary info ==============================================================================
FAILED tests/unit/api/router/test_collection_router.py::test_create_collection_success - assert 500 == 200
FAILED tests/unit/api/router/test_collection_router.py::test_create_collection_model_not_found - assert 500 == 400
FAILED tests/unit/api/router/test_collection_router.py::test_create_collection_db_error - KeyError: 'detail'
FAILED tests/unit/api/router/test_collection_router.py::test_delete_collection_success - TypeError: TestClient.delete() got an unexpected keyword argument 'json'
FAILED tests/unit/api/router/test_collection_router.py::test_delete_collection_not_found - TypeError: TestClient.delete() got an unexpected keyword argument 'json'
=========================================================================== 5 failed, 9 warnings in 1.64s ===========================================================================
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
(genai-platform-services) (base) epfn119476@25C-LTP-H-39281 genai_platform_services % 






<h2 style='color:green;'>##System##</h2>
From the test output, it's clear that there are multiple issues with the code and test setup, causing all the tests to fail. Let me break down each failure and provide fixes for these problems. Here's a comprehensive analysis of the issues and their solutions:

---

### **1. Test: `test_create_collection_success`**
#### **Issue:**
- The test is failing with the error:
  ```plaintext
  assert 500 == 200
  ```
- The response from the endpoint is a `500 Internal Server Error` instead of the expected `200 OK`.
- This is because the `check_embedding_model` function is throwing a `400` exception (`Model 'test-model' not found in embedding_models table`), which is not handled correctly in the `create_collection` function. This `400` exception is wrapped in a `500` error due to the generic exception handling.

#### **Fix:**
- The `create_collection` function should properly handle `HTTPException` exceptions raised by `check_embedding_model` and propagate the correct status code and message to the client.

- **Updated `create_collection` function in `collection_router.py`:**
  ```python
  async def create_collection(
      request: CreateCollections,
      header_information: HeaderInformation = Depends(validate_headers_and_api_key),
  ) -> dict:
      try:
          response = []

          with create_session_platform() as session_platform:
              for entry in request.collection_entries:
                  model_dim = await check_embedding_model(entry.model)
                  collection_uuid = str(uuid4())

                  collection_info = CollectionInfo(
                      uuid=collection_uuid,
                      channel_id=entry.channel_id,
                      usecase_id=entry.usecase_id,
                      collection_name=entry.collection_name,
                      model=entry.model,
                  )
                  session_platform.add(collection_info)
                  session_platform.commit()

                  create_document_model(collection_uuid, embedding_dimensions=model_dim)
                  logger.info(f"Created collection {collection_uuid} with dimensions {model_dim}")

                  response.append({"collection_name": entry.collection_name, "uuid": collection_uuid})

          return {"collections": response}

      except HTTPException as e:
          raise e  # Properly propagate HTTPException errors
      except Exception as e:
          logger.exception("Error creating collection.")
          raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
  ```

---

### **2. Test: `test_create_collection_model_not_found`**
#### **Issue:**
- The test is failing with the error:
  ```plaintext
  assert 500 == 400
  ```
- Similar to the previous test, the `400` error raised by `check_embedding_model` is being wrapped as a `500 Internal Server Error` due to generic exception handling in `create_collection`.

#### **Fix:**
- The fix for this issue is the same as in Test 1: propagate `HTTPException` errors properly in the `create_collection` function (as shown above).

---

### **3. Test: `test_create_collection_db_error`**
#### **Issue:**
- The test is failing with the error:
  ```plaintext
  KeyError: 'detail'
  ```
- This occurs because the `response.json()` does not contain a `"detail"` key. This is likely due to the `Exception` raised by the `create_session_platform` mock not being translated into a proper `HTTPException` in the `create_collection` function.

#### **Fix:**
- Ensure that all generic exceptions are converted into `HTTPException` with a proper `"detail"` key. This is already handled in the improved `create_collection` function shown above.

---

### **4. Test: `test_delete_collection_success`**
#### **Issue:**
- The test is failing with the error:
  ```plaintext
  TypeError: TestClient.delete() got an unexpected keyword argument 'json'
  ```
- This is because the `TestClient.delete` method does not accept the `json` argument. For `DELETE` requests, the payload should be passed using the `data` argument.

#### **Fix:**
- Update the test to use `data` instead of `json`:
  ```python
  response = client.delete("/v1/api/delete_collection", data=delete_collection_payload, headers=valid_headers)
  ```

---

### **5. Test: `test_delete_collection_not_found`**
#### **Issue:**
- The test is failing with the same error as the previous test:
  ```plaintext
  TypeError: TestClient.delete() got an unexpected keyword argument 'json'
  ```

#### **Fix:**
- Update the test to use `data` instead of `json`:
  ```python
  response = client.delete("/v1/api/delete_collection", data=delete_collection_payload, headers=valid_headers)
  ```

---

### **Additional Improvements**
1. **Mock SQLAlchemy Database Calls Properly:**
   - The `check_embedding_model` function queries the database, and it should be mocked properly to avoid actual database interaction during tests.
   - Use `mocker.patch` to mock the `session.execute` call:
     ```python
     mocker.patch(
         "src.db.connection.create_session_platform",
         return_value=MagicMock(execute=MagicMock(return_value=MockResult(model_name="test-model", dimensions=128)))
     )
     ```

2. **Update Test Coverage:**
   - The current test coverage is 50%, which is far below the required 75%. Add tests for the following:
     - Invalid headers (e.g., missing `x-session-id` or `x-base-api-key`).
     - Edge cases for `delete_collection` (e.g., trying to delete a collection that does not exist).

3. **Address Deprecation Warnings:**
   - The logs contain warnings about deprecated usage in SQLAlchemy and Pydantic. Update the code to use the new APIs:
     - Replace `declarative_base()` with `sqlalchemy.orm.declarative_base()`.
     - Use `ConfigDict` instead of `config` in Pydantic models.

---

### **Final Steps**
After applying the fixes:
1. Rerun the tests using `pytest`:
   ```bash
   pytest tests/unit/api/router/test_collection_router.py --cov=src --cov-report=term
   ```
2. Ensure all tests pass and the coverage meets the required threshold.

---

### **Expected Output After Fixes**
```plaintext
============================= test session starts ==============================
platform darwin -- Python 3.11.12, pytest-8.3.5, pluggy-1.5.0
rootdir: /Users/epfn119476/Documents/HDFC/genai_platform_services
plugins: cov-6.1.1, mock-3.14.0
collected 5 items                                                             

tests/unit/api/router/test_collection_router.py .....                     [100%]

========================== 5 passed in 1.23s ==========================
```

This will confirm that the fixes are successful. Let me know if you encounter further issues!

<h2 style='color:cyan;'>##User##</h2>


where should i use it?

mocker.patch(
    "src.db.connection.create_session_platform",
    return_value=MagicMock(execute=MagicMock(return_value=MockResult(model_name="test-model", dimensions=128)))
)




<h2 style='color:green;'>##System##</h2>
You should use the `mocker.patch` statement when you want to mock the `create_session_platform` function to avoid real database calls during your tests. Specifically, this mocking is needed for functions like `check_embedding_model` that query the database to fetch embedding model information.

---

### **Where to Use It**

You should use it in the test cases that involve database interactions, such as `test_create_collection_success`, `test_create_collection_model_not_found`, and possibly `test_delete_collection_success` (if the `delete` functionality queries the database).

For example, in the `test_create_collection_success` test case, `check_embedding_model` queries the `EmbeddingModels` table using the `create_session_platform` function. To mock this behavior, you would patch `create_session_platform` to return a mocked session that provides a mocked query result.

---

### **Updated Test Code**

Below is an example of how you would use the `mocker.patch` statement in your tests:

#### **Updated `test_create_collection_success`**
```python
from unittest.mock import MagicMock

def test_create_collection_success(
        client, create_collection_payload, valid_headers, override_dependencies, mocker
):
    # Mock the database session and query result for check_embedding_model
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar_one_or_none.return_value = MagicMock(
        model_name="test-model", dimensions=128
    )
    mocker.patch("src.db.connection.create_session_platform", return_value=mock_session)

    # Mock create_document_model to avoid actual table creation
    mocker.patch("src.repository.document_repository.create_document_model", return_value=None)

    # Make the API call
    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    
    # Assertions
    assert response.status_code == 200
    response_data = response.json()
    assert "collections" in response_data
    assert response_data["collections"][0]["collection_name"] == "test-collection"
```

---

#### **Updated `test_create_collection_model_not_found`**
```python
def test_create_collection_model_not_found(
        client, create_collection_payload, valid_headers, override_dependencies, mocker
):
    # Mock the database session to simulate model not found
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar_one_or_none.return_value = None
    mocker.patch("src.db.connection.create_session_platform", return_value=mock_session)

    # Make the API call
    response = client.post("/v1/api/create_collection", json=create_collection_payload, headers=valid_headers)
    
    # Assertions
    assert response.status_code == 400
    assert response.json()["detail"] == "Model 'test-model' not found in embedding_models table."
```

---

#### **Explanation of the Mocking**
1. **`create_session_platform` Mock:**
   - The `create_session_platform` function is mocked to return a `MagicMock` object.
   - The `MagicMock` object simulates the behavior of the session, including the `execute` method and its return value.

2. **`mock_session.execute.return_value.scalar_one_or_none.return_value`:**
   - This mocks the behavior of a database query and its result.
   - For `test_create_collection_success`, it returns a mocked object with `model_name` and `dimensions`.
   - For `test_create_collection_model_not_found`, it returns `None` to simulate a "model not found" scenario.

3. **Why Mock This?**
   - It avoids actual database queries during the test.
   - It allows you to control the behavior of the database session and simulate different scenarios.

---

### **Other Tests**
You can use the same approach to mock database queries for other tests, such as `test_delete_collection_success` or `test_delete_collection_not_found`, if those tests involve database calls through `create_session_platform`.

For example, in `test_delete_collection_success`, you can mock the session to simulate the `CollectionInfo` query result.

---

### **When Not to Mock It**
If the test does not involve a database call, you do not need to use this patch. For example, if the test focuses solely on input validation or logic that does not depend on the database, you can omit the `create_session_platform` mock.

Let me know if you need help applying this to other test cases!

