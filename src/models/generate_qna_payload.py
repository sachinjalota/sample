from typing import Optional
from pydantic import BaseModel, Field


class CompletionPayload(BaseModel):
    session_id: Optional[str] = Field(..., description="Session ID")
    usecase_id: Optional[str] = Field(..., description="Use case ID")
    object_path: str = Field(
        ..., description="the upload object path on storage"
    )
    mime_type: str = Field(
        ..., description="mime type like application/pdf"
    )
    max_completion_tokens: Optional[int] = Field(
        8192,
        description="Upper bound for generated tokens.",
    )
    no_of_qna: Optional[int] = Field(
        10, description="Number of generated qna"
    )
    seed: Optional[int] = Field(1, description="Seed value for deterministic results.")
    temperature: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for randomness (higher = more random).",
    )
    top_p: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (top probability mass considered).",
    )
    question_context: Optional[str] = Field(
        "", description="give additional question context"
    )
    user_prompt: Optional[str] = Field(
        "", description="user_prompt"
    )
    system_prompt: Optional[str] = Field(
        "", description="user_prompt"
    )
