from fastapi import APIRouter, HTTPException, Response, status, Depends
from src.models.upload_object_payload import UploadObjectPayload
from src.cloud_storage.storage_service import CloudStorage
from src.api.utility.file_upload_service import FileUploadService
import src.config as config
from src.logging_config import Logger
from fastapi.responses import JSONResponse

router = APIRouter()
logger = Logger.create_logger(__name__)


def get_storage_service():
    return CloudStorage(config.CLOUD_STORAGE_PROVIDER)


def get_file_upload_service(
        storage_service=Depends(get_storage_service),
):
    return FileUploadService(storage_service, logger)


@router.post(f"/{config.UPLOAD_OBJECT_ENDPOINT}",
             summary="Upload Files ",
             response_description="Unique ID: Unique ID , object_path: uploaded gcp path",
             status_code=status.HTTP_200_OK,
             )
async def upload_object(payload: UploadObjectPayload,
                        file_upload_service: FileUploadService = Depends(get_file_upload_service)):
    logger.info(f"Entry into upload_object with input {payload.file_name}")
    try:
        response = await file_upload_service.upload_object(payload)
        logger.info(f"File uploaded successfully with response : {response}")
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail} while uploading the file")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while uploading the file")
