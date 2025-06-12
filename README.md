from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union
from fastapi import HTTPException, status
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeMeta

from src.db.connection import create_session_platform
from src.logging_config import Logger

T = TypeVar("T", bound=DeclarativeMeta)
logger = Logger.create_logger(__name__)

class DBManager:
    """Generic CRUD operations for any SQLAlchemy model."""

    @staticmethod
    def _handle_error(exc: Exception, msg: str):
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @staticmethod
    def _build_filters(
        model: Type[T],
        filters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None
    ) -> Sequence[Any]:
        if not filters:
            return []
        if isinstance(filters, dict):
            return [getattr(model, k) == v for k, v in filters.items()]
        return list(filters)  # assume SQLAlchemy expressions

    @classmethod
    def select_one(
        cls,
        model: Type[T],
        filters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        raise_not_found: bool = True
    ) -> Optional[T]:
        try:
            with create_session_platform() as session:
                stmt = select(model).where(*cls._build_filters(model, filters))
                result = session.execute(stmt).scalar_one_or_none()

            if result is None and raise_not_found:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"{model.__name__} not found for filters {filters}"
                )
            return result

        except HTTPException:
            raise
        except Exception as exc:
            cls._handle_error(exc, f"Error selecting one {model.__name__}")

    @classmethod
    def select_many(
        cls,
        model: Type[T],
        filters: Optional[Union[Dict[str, Any], Sequence[Any]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        try:
            with create_session_platform() as session:
                stmt = select(model).where(*cls._build_filters(model, filters))
                if limit is not None:
                    stmt = stmt.limit(limit)
                if offset is not None:
                    stmt = stmt.offset(offset)
                return session.execute(stmt).scalars().all()

        except Exception as exc:
            cls._handle_error(exc, f"Error selecting many {model.__name__}")

    @classmethod
    def insert_one(cls, model: Type[T], data: Dict[str, Any]) -> Any:
        try:
            with create_session_platform() as session:
                stmt = insert(model).values(**data)
                result = session.execute(stmt)
                session.commit()
                return result.inserted_primary_key
        except Exception as exc:
            cls._handle_error(exc, f"Error inserting into {model.__name__}")

    @classmethod
    def update_many(
        cls,
        model: Type[T],
        filters: Optional[Union[Dict[str, Any], Sequence[Any]]],
        data: Dict[str, Any]
    ) -> int:
        try:
            with create_session_platform() as session:
                stmt = update(model).where(*cls._build_filters(model, filters)).values(**data)
                result = session.execute(stmt)
                session.commit()
                return result.rowcount
        except Exception as exc:
            cls._handle_error(exc, f"Error updating {model.__name__}")

    @classmethod
    def delete_many(
        cls,
        model: Type[T],
        filters: Optional[Union[Dict[str, Any], Sequence[Any]]]
    ) -> int:
        try:
            with create_session_platform() as session:
                stmt = delete(model).where(*cls._build_filters(model, filters))
                result = session.execute(stmt)
                session.commit()
                return result.rowcount
        except Exception as exc:
            cls._handle_error(exc, f"Error deleting from {model.__name__}")
