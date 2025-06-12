
I have following files of a service from a project based on fastapi 

>> collection_router.py
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select

from src.api.deps import (
    check_embedding_model,
    validate_collection_access,
    validate_headers_and_api_key,
)
from src.config import get_settings
from src.db.connection import create_session_platform
from src.db.platform_meta_tables import CollectionInfo
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

    except HTTPException as e:
        raise e

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
                select(CollectionInfo).where(CollectionInfo.uuid == request.collection_uid)
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

            session_platform.execute(delete(CollectionInfo).where(CollectionInfo.uuid == request.collection_uid))
            session_platform.commit()

            logger.info(f"Deleted collection with UID '{request.collection_uid}'")

        return {"message": "Collection has been deleted.", "collection": request.collection_uid}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting collection.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

>> deps.py
async def check_embedding_model(embedd_model: str) -> int:
    with create_session_platform() as sess_plat:
        model_query = sess_plat.execute(
            select(EmbeddingModels).where(EmbeddingModels.model_name == embedd_model)
        ).scalar_one_or_none()
        if not model_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{embedd_model}' not found in embedding_models table.",
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



Can we create a class like DB Manager that handles all the db related code statements and keep the router and deps files clean?
