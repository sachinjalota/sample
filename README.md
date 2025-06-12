from fastapi import APIRouter, Depends, HTTPException, status
from src.api.deps import (
    validate_headers_and_api_key,
    check_embedding_model,
    validate_collection_access,
)
from src.db.db_manager import DBManager
from src.repository.document_repository import DocumentRepository, create_document_model
from src.models.collection_payload import CreateCollections, DeleteCollection
from src.models.headers import HeaderInformation
from src.logging_config import Logger

router = APIRouter()
logger = Logger.create_logger(__name__)

@router.post(
    "/collections",
    summary="Create one or more collections",
    status_code=status.HTTP_200_OK,
)
async def create_collection(
    request: CreateCollections,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
):
    response = []

    for entry in request.collection_entries:
        # 1) Validate & get embedding dims
        dims = await check_embedding_model(entry.model)

        # 2) Insert metadata row
        pk = DBManager.insert_one(
            model=CollectionInfo,
            data={
                "uuid": entry.generated_uuid := str(uuid4()),
                "channel_id": entry.channel_id,
                "usecase_id": entry.usecase_id,
                "collection_name": entry.collection_name,
                "model": entry.model,
            }
        )
        # pk is a tuple; uuid is explicitly in payload so you can ignore pk

        # 3) Create the actual document table & indexes
        create_document_model(entry.generated_uuid, embedding_dimensions=dims)

        logger.info(f"Created collection {entry.generated_uuid}")
        response.append({
            "collection_name": entry.collection_name,
            "uuid": entry.generated_uuid,
        })

    return {"collections": response}


@router.delete(
    "/collections/{collection_uid}",
    summary="Delete a collection by UUID",
    status_code=status.HTTP_200_OK,
)
async def delete_collection(
    request: DeleteCollection,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
):
    # 1) Ensure the client is allowed to delete this collection
    await validate_collection_access(header_information.x_base_api_key, request.collection_uid)

    # 2) Ensure the physical table exists, then drop it
    repo = DocumentRepository(request.collection_uid, create_if_not_exists=False)
    if not repo.check_table_exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection table '{request.collection_uid}' missing."
        )
    repo.delete_collection()

    # 3) Delete the metadata row
    deleted = DBManager.delete_many(
        model=CollectionInfo,
        filters={"uuid": request.collection_uid}
    )
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{request.collection_uid}' not found."
        )

    logger.info(f"Deleted collection {request.collection_uid}")
    return {
        "message": "Collection has been deleted.",
        "collection": request.collection_uid
    }
