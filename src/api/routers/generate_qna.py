from fastapi import APIRouter, HTTPException, Response, status, Depends
from src.models.generate_qna_payload import CompletionPayload
from src.cloud_storage.storage_service import CloudStorage
# from src.providers.cloud_llm import get_llm_service
from src.llm.openai_client import OpenAILLM
from src.utility.helper import get_markdown_from_pdf
from src.api.utility.file_upload_service import FileUploadService
import src.config as config
from src.logging_config import Logger
from src.api.prompts.default_prompts import (QNA_GENERATE_USER_PROMPT,
                                             QNA_GENERATE_SYSTEM_PROMPT)
from fastapi.responses import JSONResponse

router = APIRouter()


def get_logger():
    return Logger.create_logger(__name__)


# Dependency for cloud storage
def get_storage_service():
    return CloudStorage(config.CLOUD_STORAGE_PROVIDER)


# Dependency for LLM
def get_llm():
    return OpenAILLM()


# Dependency for FileUploadService
def get_file_upload_service(
        storage: CloudStorage = Depends(get_storage_service),
        logger=Depends(get_logger),
):
    return FileUploadService(storage, logger)


@router.post(f"/{config.GENERATE_QNA_ENDPOINT}",
             summary="Endpoint for QNA Generation",
             response_description="List object containing  qna ",
             status_code=status.HTTP_200_OK,
             )
async def generate_qna(
        payload: CompletionPayload,
        file_upload_service: FileUploadService = Depends(get_file_upload_service),
        llm: OpenAILLM = Depends(get_llm),
        logger=Depends(get_logger)
):
    try:
        storage_object_path = payload.object_path
        file_path = file_upload_service.download_file(storage_object_path)
        markdown_string = get_markdown_from_pdf(file_path)
        payload.user_prompt = QNA_GENERATE_USER_PROMPT
        payload.system_prompt = QNA_GENERATE_SYSTEM_PROMPT.format(document_content=markdown_string)
        response = llm.generate_response(payload)
        logger.info(f"Response generated successfully with response : {response}")
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the qna")
    finally:
        file_upload_service.clean_temp_file(file_path)
