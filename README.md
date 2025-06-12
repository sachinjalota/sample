import httpx
from fastapi import HTTPException, status, Depends
from src.config import get_settings
from src.db.db_manager import DBManager
from src.db.platform_meta_tables import CollectionInfo
from src.db.connection import create_session_platform  # only if you still need sessions elsewhere

settings = get_settings()

async def validate_headers_and_api_key(...) -> HeaderInformation:
    # your existing header validation logic
    ...

async def check_embedding_model(model_name: str) -> int:
    # Re-use the generic select_one to fetch the model row:
    row = DBManager.select_one(
        model=EmbeddingModels,
        filters={"model_name": model_name},
        raise_not_found=False
    )
    if not row:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Embedding model '{model_name}' not found."
        )
    return int(row.dimensions)

async def validate_collection_access(api_key: str, collection_uuid: str) -> None:
    # 1) Check with Prompt-Hub
    validation_url = f"{settings.prompt_hub_endpoint}{settings.prompt_hub_get_usecase_by_apikey}"
    verify = settings.deployment_env == "PROD"
    async with httpx.AsyncClient(verify=verify) as client:
        resp = await client.get(validation_url, headers={"lite-llm-api-key": api_key})
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized API key")

    data = resp.json()["data"]
    channel_id = data["channel_id"]
    team_id = data["team_id"]

    # 2) Fetch the collection row
    col = DBManager.select_one(
        model=CollectionInfo,
        filters={"uuid": collection_uuid}
    )

    # 3) Verify ownership
    if str(col.channel_id) != channel_id or str(col.usecase_id) != team_id:
        raise HTTPException(status_code=403, detail="Not authorized for this collection.")
