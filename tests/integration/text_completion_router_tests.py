from src.api.routers import text_completion_router
from fastapi.testclient import TestClient
from src.main import app
from src.config import API_COMMON_PREFIX, TEXT_COMPLETION_ENDPOINT
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.anyio
async def test_text_completion():
    with TestClient(app) as client:
        data = {
            "model_name": "openai/gemini-1.5-flash",
            "system_prompt": "string",
            "user_prompt": "Who is the Pm of India",
            "auth_params": {},
            "temperature": 1,
            "max_completion_tokens": -1,
            "max_tokens": -1,
            "modalities": [
                "text"
            ],
            "stream": "false",
            "history": {
                "data": [
                    {
                        "role": "string",
                        "content": "string"
                    }
                ]
            }
        }
        async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(f"{API_COMMON_PREFIX}{TEXT_COMPLETION_ENDPOINT}", json=data)
        assert response.status_code == 200


if __name__ == '__main__':
    test_text_completion()
