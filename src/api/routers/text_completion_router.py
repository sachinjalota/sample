from litellm import completion
from fastapi import APIRouter, HTTPException, Response, status, Depends
from src.logging_config import Logger
import src.config as config
from src.api.prompts.default_prompts import DEFAULT_SYSTEM_PROMPT
from src.models.text_input import TextCompletionRequest
from fastapi.responses import JSONResponse

router = APIRouter()
logger = Logger.create_logger("text_input")


@router.post(f"{config.TEXT_COMPLETION_ENDPOINT}",
             summary= "Endpoint for text completion request ",
             response_description="content: response from llm ",
             status_code=status.HTTP_200_OK,
             )
async def text_completion(request: TextCompletionRequest):
    try:
        response = completion(
            model=request.model_name,
            messages=[
                {"role": "system",
                 "content": request.system_prompt if request.system_prompt else DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": request.user_prompt}
            ],
            stream=request.stream,
            api_key=config.BASE_API_KEY,
            base_url=config.BASE_API_URL,
            headers=config.BASE_API_HEADERS)
        logger.info(f"completion Response Generated Successful : {response}")
        return JSONResponse(content=response.dict(), status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while uploading the file : {str(e)}")
