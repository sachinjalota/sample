from typing import Optional, TypeAlias, Literal, List

from pydantic import BaseModel, Field

from src.config import DEFAULT_IMAGE_COMPLETION_MODEL

ChatCompletionModality: TypeAlias = Literal["text", "audio"]


class ImageCompletionRequest(BaseModel):
    model_name: str = Field(min_length=5, description="Must be at least 5 characters long",
                            default=DEFAULT_IMAGE_COMPLETION_MODEL)
    system_prompt: str = Optional[str]
    user_query: str = Field(..., min_length=5, description="Must be at least 5 characters long")
    image_url: str = Field(..., description="Must be a Accessible Image")
    auth_params: BaseModel
    max_completion_tokens: Optional[int] = -1
    max_tokens: Optional[int] = -1
    modalities: Optional[List[ChatCompletionModality]] = ["text"]
