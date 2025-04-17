from typing import Optional, TypeAlias, Literal, List

from pydantic import BaseModel, Field
from src.config import DEFAULT_TEXT_COMPLETION_MODEL

ChatCompletionModality: TypeAlias = Literal["text", "audio"]


class Message(BaseModel):
    role: str
    content: str


class HistoryModel(BaseModel):
    data: List[Message]


class AuthorizationAttribute(BaseModel):
    bearer: str = "12345"
    organization_id: str
    project_id: str
    use_case_id: str


# class OptionalGenerationParams(BaseModel):
#     temperature: Optional[float] = 1,
#     max_completion_tokens: Optional[int] = -1,
#     max_tokens: Optional[int] = -1,
#     modalities: Optional[List[ChatCompletionModality]] = ["text"],


class TextCompletionRequest(BaseModel):
    model_name: str = Field(min_length=5, description="Must be at least 5 characters long",
                            default=DEFAULT_TEXT_COMPLETION_MODEL)
    system_prompt: str = Optional[str]
    user_prompt: str = Field(..., min_length=5, description="Must be at least 5 characters long")
    auth_params: BaseModel
    # optional_generation_params: OptionalGenerationParams
    temperature: Optional[float] = 1
    max_completion_tokens: Optional[int] = -1
    max_tokens: Optional[int] = -1
    modalities: Optional[List[ChatCompletionModality]] = ["text"]
    stream: bool = False
    history: Optional[HistoryModel] = None
