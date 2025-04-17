from contextlib import asynccontextmanager
from typing import AsyncGenerator, List
import os


from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from src.api.routers import (text_completion_router, generate_qna,
                             image_input_completion_router,
                             upload_file_router)

from .logging_config import Logger
from .config import \
    (ALLOWED_ORIGINS, ALLOW_CREDENTIALS,
     ALLOW_METHODS, ALLOW_HEADERS,
     API_COMMON_PREFIX, HEALTH_CHECK,
     ENV, SERVICE_SLUG)

allowed_origins: List[str] = ALLOWED_ORIGINS.split(",")
allowed_methods: List[str] = ALLOW_METHODS.split(",")
allowed_headers: List[str] = ALLOW_HEADERS.split(",")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan handler for startup and shutdown events."""
    Logger.info("Starting up application...")
    yield
    Logger.info("Shutting down application...")


def create_app(version: str = "0.1.0") -> FastAPI:
    app = FastAPI(
        title="Chat As Service",
        description="Chat As Service This Support Various LLM Models With OpenAi Specification",
        version=version,
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        lifespan=lifespan,
        root_path=f"/{ENV}/{SERVICE_SLUG}"

    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=ALLOW_CREDENTIALS,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers
    )

    @app.middleware("http")
    async def add_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = ",".join(allowed_origins)
        response.headers["Access-Control-Allow-Methods"] = ",".join(allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ",".join(allowed_headers)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "loc": err["loc"],
                        "msg": err["msg"],
                        "type": err["type"]
                    }
                    for err in exc.errors()
                ],
                "body": exc.body
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.get("/", tags=["Main"])
    async def read_root():
        return {"name": "Chat As Service, go to docs path for API detail"}

    @app.get(f"{API_COMMON_PREFIX}{HEALTH_CHECK}", tags=["Main"])
    async def read_root():
        return {"status": "ok"}

    app.include_router(text_completion_router.router, prefix=API_COMMON_PREFIX, tags=["text_completion"])
    app.include_router(image_input_completion_router.router, prefix=API_COMMON_PREFIX, tags=["image_completion"])
    app.include_router(upload_file_router.router, prefix=API_COMMON_PREFIX, tags=["upload_file"])
    app.include_router(generate_qna.router, prefix=API_COMMON_PREFIX, tags=['generate_qna'])
    return app


app = create_app()
