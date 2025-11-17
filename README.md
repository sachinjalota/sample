import json
import os
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Type

from src.logging_config import Logger
from src.services.base_class.vector_store_base import BaseVectorStore, VectorStoreConfig
from src.services.embedding_service import EmbeddingService

logger = Logger.create_logger(__name__)

VECTORSTORE_MODULES = os.getenv(
    "VECTORSTORE_MODULES",
    "src.services.strategies.vector_store_PG_strategy, src.services.strategies.vector_store_ES_strategy",
).split(",")


class VectorStoreNotFoundError(Exception):
    """Raised when a requested vector store backend is not registered."""
    pass


class VectorStoreFactory:
    """
    Factory for creating vector store backend instances.
    Supports registration, caching, and lazy loading of backends.
    """

    _instance: Optional["VectorStoreFactory"] = None
    _registry: Dict[str, Dict[str, Any]] = {}
    _loaded: bool = False
    _cache: Dict[str, BaseVectorStore] = {}

    def __new__(cls) -> "VectorStoreFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls, name: str, description: str = "", tags: Optional[List[str]] = None
    ) -> Callable[[Type[BaseVectorStore]], Type[BaseVectorStore]]:
        """
        Decorator to register a vector store backend.
        
        Usage:
            @VectorStoreFactory.register("pgvector", description="PostgreSQL backend")
            class PGVectorStore(BaseVectorStore):
                ...
        """
        def decorator(store_cls: Type[BaseVectorStore]) -> Type[BaseVectorStore]:
            cls._registry[name.lower()] = {
                "class": store_cls,
                "description": description or (store_cls.__doc__ or "").strip(),
                "tags": tags or [],
            }
            logger.debug(f"Registered vector store backend: {name}")
            return store_cls

        return decorator

    @classmethod
    def _load_backends(cls) -> None:
        """Lazy load all backend modules from environment configuration."""
        if not cls._loaded:
            for module in VECTORSTORE_MODULES:
                module = module.strip()
                if module:
                    try:
                        import_module(module)
                        logger.debug(f"Loaded vector store backend module: {module}")
                    except Exception as e:
                        logger.warning(f"Failed to load vector store backend {module}: {e}")
            cls._loaded = True

    @classmethod
    def list_backends(cls) -> List[Dict[str, Any]]:
        """
        List all registered vector store backends.
        
        Returns:
            List of dicts with backend metadata (name, description, tags)
        """
        cls._load_backends()
        return [
            {
                "name": k,
                "description": v["description"],
                "tags": v["tags"]
            }
            for k, v in cls._registry.items()
        ]

    @classmethod
    def _cache_key(cls, name: str, config: VectorStoreConfig) -> str:
        """Generate cache key from backend name and config."""
        return f"{name}:{json.dumps(config.dict(), sort_keys=True, default=str)}"

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[VectorStoreConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> BaseVectorStore:
        """
        Create or retrieve a cached vector store backend instance.
        
        Args:
            name: Backend name (e.g., "pgvector", "elasticsearch")
            config: Configuration for the backend
            embedding_service: Embedding service instance
            
        Returns:
            BaseVectorStore instance
            
        Raises:
            VectorStoreNotFoundError: If backend is not registered
        """
        cls._load_backends()

        meta = cls._registry.get(name.lower())
        if not meta:
            raise VectorStoreNotFoundError(
                f"Vector store backend '{name}' not found. "
                f"Registered backends: {list(cls._registry.keys())}"
            )

        config = config or VectorStoreConfig(backend=name)
        key = cls._cache_key(name, config)

        # Return cached instance if available
        if key in cls._cache:
            logger.debug(f"Using cached backend instance: {name}")
            return cls._cache[key]

        # Create new instance
        backend_cls: Type[BaseVectorStore] = meta["class"]
        instance = backend_cls(config, embedding_service)
        cls._cache[key] = instance

        logger.info(f"Created new vector store backend: {name}")
        return instance

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached backend instances."""
        cls._cache.clear()
        logger.info("Cleared vector store backend cache")

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a backend is registered."""
        cls._load_backends()
        return name.lower() in cls._registry
