from fastapi import APIRouter, HTTPException, status
from litellm import completion
from fastapi.responses import JSONResponse

import src.config as config
from src.logging_config import Logger
from src.api.prompts.default_prompts import DEFAULT_SYSTEM_PROMPT
from src.models.image_input import ImageCompletionRequest

logger = Logger.create_logger("image_input")

router = APIRouter()


@router.post(f"{config.IMAGE_COMPLETION_ENDPOINT}",
             summary="Endpoint for Image  completion request ",
             response_description="content: response from llm ",
             status_code=status.HTTP_200_OK,
             )
async def image_completion(request: ImageCompletionRequest):
    try:
        messages = [
            {"role": "system",
             "content":
                 request.system_prompt if request.system_prompt else DEFAULT_SYSTEM_PROMPT},
            {"role": "user",
             "content":
                 [{
                     "type": "text",
                     "text": request.user_query},
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": request.image_url,
                             "format": "image/jpeg"
                         }
                     }
                 ]}
        ]
        response = completion(
            model=request.model_name,
            messages=messages,
            api_key=config.BASE_API_KEY,
            base_url=config.BASE_API_URL,
            headers=config.BASE_API_HEADERS
        )
        logger.info(f"completion Response Generated Successful : {response}")
        return JSONResponse(content=response.dict(), status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading the file : {str(e)}")
