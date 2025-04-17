from src.api.routers import image_input_completion_router
from fastapi.testclient import TestClient
from src.main import app
from src.config import API_COMMON_PREFIX, IMAGE_COMPLETION_ENDPOINT
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.anyio
async def test_text_completion():
    with TestClient(app) as client:
        data = {
          "model_name": "openai/gemini-1.5-flash",
          "system_prompt": "string",
          "user_query": "What is in the Image",
          "image_url": "https://general-purpose-public.s3.ap-south-1.amazonaws.com/gemini_test_images/aadharCard.jpeg",
          "auth_params": {},
          "max_completion_tokens": -1,
          "max_tokens": -1,
          "modalities": [
            "text"
          ]
        }
        async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(f"{API_COMMON_PREFIX}{IMAGE_COMPLETION_ENDPOINT}", json=data)
        assert response.status_code == 200


if __name__ == '__main__':
    test_text_completion()
