from fastapi.testclient import TestClient
from src.main import app
from src.config import API_COMMON_PREFIX, HEALTH_CHECK
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.anyio
async def test_health_check_manual():
    async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get(f"{API_COMMON_PREFIX}{HEALTH_CHECK}")
        print(response)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


if __name__ == '__main__':
    test_health_check_manual()
