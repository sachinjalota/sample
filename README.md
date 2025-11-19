# src/api/routers/async_file_processing_router.py - NEW FILE

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import ORJSONResponse

from src.api.deps import get_cloud_storage_service, validate_headers_and_api_key
from src.config import get_settings
from src.db.connection import create_session
from src.db.platform_meta_tables import (
    BatchUploadJob,
    FileProcessingTask,
    FileTaskStatus,
    JobStatus,
    VectorStoreInfo,
)
from src.integrations.cloud_storage import CloudStorage
from src.logging_config import Logger
from src.models.headers import HeaderInformation
from src.repository.base_repository import BaseRepository
from src.utility.vector_store_helpers import check_embedding_model, get_usecase_id_by_api_key
from src.utility.vector_store_utils import create_file_info_tbl_model

router = APIRouter()
logger = Logger.create_logger(__name__)
settings = get_settings()


async def check_file_version_conflict(
    vector_store_id: str,
    file_name: str,
    storage_backend: str,
    override: bool
) -> tuple[bool, Optional[int]]:
    """
    Check if file exists in vector store
    Returns: (conflict_exists, current_version)
    """
    try:
        # Get vector store name
        vs_record = BaseRepository.select_one(
            db_tbl=VectorStoreInfo,
            filters={"id": vector_store_id, "vector_db": storage_backend}
        )
        if not vs_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector store '{vector_store_id}' not found"
            )
        
        store_name = vs_record["name"]
        
        if storage_backend == "pgvector":
            file_info_tbl = create_file_info_tbl_model(f"{store_name}_file_info")
            existing = BaseRepository.select_one(
                db_tbl=file_info_tbl,
                filters={"file_name": file_name, "vs_id": vector_store_id},
                session_factory=create_session
            )
            
            if existing:
                if not override:
                    return True, existing.get("file_version", 1)
                else:
                    # Will need to delete old version during indexing
                    return False, existing.get("file_version", 1) + 1
        
        elif storage_backend == "elasticsearch":
            from src.db.elasticsearch_connection import get_elasticsearch_client
            client = get_elasticsearch_client()
            
            file_info_index = f"{store_name}_file_info"
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"file_name.keyword": file_name}},
                            {"term": {"vs_id": vector_store_id}}
                        ]
                    }
                }
            }
            
            response = client.search(index=file_info_index, body=query, size=1)
            hits = response.get("hits", {}).get("hits", [])
            
            if hits:
                if not override:
                    current_version = hits[0]["_source"].get("file_version", 1)
                    return True, current_version
                else:
                    current_version = hits[0]["_source"].get("file_version", 1)
                    return False, current_version + 1
        
        return False, 1  # No conflict, start at version 1
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error checking file version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check file version"
        )


@router.post(
    "/vector_stores/{store_id}/files/batch",
    summary="Upload multiple files for async processing",
    description=(
        "Uploads files to GCS and queues them for background processing. "
        "Returns immediately with job_id for status tracking."
    ),
    response_class=ORJSONResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def batch_upload_files(
    store_id: str,
    files: List[UploadFile] = File(...),
    override: bool = Query(False, description="Override existing files with same name"),
    storage_backend: str = Query(..., description="pgvector or elasticsearch"),
    attributes: Optional[dict] = Query(None, description="Custom file attributes"),
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
    cloud_service: CloudStorage = Depends(get_cloud_storage_service),
) -> ORJSONResponse:
    """
    Async file upload endpoint
    1. Validates VS and user access
    2. Checks file versions and conflicts
    3. Uploads to GCS
    4. Creates job and task records
    5. Enqueues Celery tasks
    6. Returns job_id immediately
    """
    
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 files per batch"
        )
    
    try:
        # 1. Validate vector store and get metadata
        usecase_id = await get_usecase_id_by_api_key(header_information.x_base_api_key)
        
        vs_record = BaseRepository.select_one(
            db_tbl=VectorStoreInfo,
            filters={"id": store_id, "vector_db": storage_backend}
        )
        
        if not vs_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector store '{store_id}' not found for backend '{storage_backend}'"
            )
        
        if vs_record["usecase_id"] != usecase_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this vector store"
            )
        
        store_name = vs_record["name"]
        embedding_model = vs_record["model_name"]
        
        # Get embedding model details
        model_path, embedding_dimensions, context_length = await check_embedding_model(embedding_model)
        
        # 2. Validate files and check for conflicts
        validated_files = []
        conflicts = []
        
        for file in files:
            # Validate file type
            if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_file_types):
                conflicts.append({
                    "file_name": file.filename,
                    "reason": f"Unsupported file type. Allowed: {', '.join(settings.allowed_file_types)}"
                })
                continue
            
            # Validate file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset
            
            if file_size > settings.max_file_size_bytes:
                conflicts.append({
                    "file_name": file.filename,
                    "reason": f"File too large. Max: {settings.max_file_size_bytes / 1024 / 1024}MB"
                })
                continue
            
            # Check version conflict
            has_conflict, new_version = await check_file_version_conflict(
                store_id, file.filename, storage_backend, override
            )
            
            if has_conflict:
                conflicts.append({
                    "file_name": file.filename,
                    "reason": f"File already exists (version {new_version - 1}). Use override=true to replace."
                })
                continue
            
            validated_files.append((file, new_version))
        
        if conflicts and len(validated_files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "All files have conflicts", "conflicts": conflicts}
            )
        
        # 3. Create batch job record
        job_id = uuid4()
        now_dt = datetime.now(ZoneInfo(settings.timezone))
        
        job_data = {
            "job_id": str(job_id),
            "vector_store_id": store_id,
            "usecase_id": usecase_id,
            "created_at": now_dt,
            "status": JobStatus.QUEUED.value,
            "total_files": len(validated_files),
            "files_queued": len(validated_files),
            "storage_backend": storage_backend,
            "embedding_model": embedding_model,
            "chunking_strategy": vs_record.get("chunking_strategy"),
        }
        
        BaseRepository.insert_one(db_tbl=BatchUploadJob, data=job_data)
        logger.info(f"Created batch job {job_id} for {len(validated_files)} files")
        
        # 4. Upload files to GCS and create task records
        uploaded_files = []
        task_contexts = []
        
        for file, new_version in validated_files:
            file_id = uuid4()
            task_id = uuid4()
            
            # Upload to GCS
            gcs_object_name = f"vector_store_files/{store_id}/{file_id}_{file.filename}"
            
            file.file.seek(0)
            gcs_path = cloud_service.upload_object(
                file.file,
                bucket_name=settings.upload_bucket_name,
                object_name=gcs_object_name
            )
            
            # Create task record
            task_data = {
                "task_id": str(task_id),
                "job_id": str(job_id),
                "file_id": str(file_id),
                "file_name": file.filename,
                "gcs_path": gcs_path,
                "status": FileTaskStatus.QUEUED.value,
                "file_version": new_version,
                "created_at": now_dt,
            }
            
            BaseRepository.insert_one(db_tbl=FileProcessingTask, data=task_data)
            
            # Prepare task context for Celery
            task_context = {
                "task_id": str(task_id),
                "job_id": str(job_id),
                "file_id": str(file_id),
                "file_name": file.filename,
                "gcs_path": gcs_path,
                "vector_store_id": store_id,
                "vector_store_name": store_name,
                "usecase_id": usecase_id,
                "storage_backend": storage_backend,
                "embedding_model": embedding_model,
                "model_path": model_path,
                "embedding_dimensions": embedding_dimensions,
                "context_length": context_length,
                "chunking_strategy": vs_record.get("chunking_strategy"),
                "attributes": attributes or {},
                "file_version": new_version,
            }
            
            task_contexts.append(task_context)
            
            uploaded_files.append({
                "file_id": str(file_id),
                "task_id": str(task_id),
                "file_name": file.filename,
                "status": "queued",
                "version": new_version,
            })
        
        # 5. Enqueue Celery tasks
        from src.celery_tasks.file_processing import process_single_file
        from celery import group
        
        # Create task group for parallel processing
        job = group([
            process_single_file.s(context)
            for context in task_contexts
        ])
        
        result = job.apply_async()
        
        logger.info(f"Enqueued {len(task_contexts)} file processing tasks for job {job_id}")
        
        # 6. Return response
        response_data = {
            "job_id": str(job_id),
            "status": "queued",
            "total_files": len(validated_files),
            "files_queued": len(validated_files),
            "files_uploaded": uploaded_files,
            "conflicts": conflicts if conflicts else None,
            "message": f"Successfully queued {len(validated_files)} files for processing"
        }
        
        return ORJSONResponse(content=response_data, status_code=status.HTTP_202_ACCEPTED)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Batch upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upload failed: {str(e)}"
        )


@router.get(
    "/jobs/{job_id}/status",
    summary="Get job processing status",
    description="Returns detailed status of batch upload job and individual file tasks",
    response_class=ORJSONResponse,
)
async def get_job_status(
    job_id: str,
    header_information: HeaderInformation = Depends(validate_headers_and_api_key),
) -> ORJSONResponse:
    """
    Status check endpoint
    Returns:
    - Overall job status
    - Per-file progress
    - Error details (user-friendly)
    """
    try:
        # Validate job exists and user has access
        job = BaseRepository.select_one(
            db_tbl=BatchUploadJob,
            filters={"job_id": job_id}
        )
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found"
            )
        
        # Verify user access
        usecase_id = await get_usecase_id_by_api_key(header_information.x_base_api_key)
        if job["usecase_id"] != usecase_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this job"
            )
        
        # Get file tasks
        tasks = BaseRepository.select_many(
            db_tbl=FileProcessingTask,
            filters={"job_id": job_id},
        )
        
        # Format file statuses
        files_status = []
        for task in tasks:
            file_status = {
                "file_id": task["file_id"],
                "task_id": task["task_id"],
                "file_name": task["file_name"],
                "status": task["status"],
                "current_phase": task.get("current_phase"),
                "retry_count": task.get("retry_count", 0),
                "created_at": task["created_at"].isoformat() if task.get("created_at") else None,
                "started_at": task["started_at"].isoformat() if task.get("started_at") else None,
                "completed_at": task["completed_at"].isoformat() if task.get("completed_at") else None,
                "chunks_count": task.get("chunks_count"),
                "usage_bytes": task.get("usage_bytes"),
            }
            
            # Add error info if failed
            if task["status"] == FileTaskStatus.FAILED.value:
                file_status["error"] = {
                    "message": task.get("error_message"),
                    "stage": task.get("error_details", {}).get("stage") if task.get("error_details") else None,
                }
            
            files_status.append(file_status)
        
        # Calculate progress percentage
        progress_pct = 0
        if job["total_files"] > 0:
            progress_pct = round((job["files_completed"] / job["total_files"]) * 100, 2)
        
        response_data = {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"].isoformat(),
            "updated_at": job["updated_at"].isoformat() if job.get("updated_at") else None,
            "progress": {
                "total_files": job["total_files"],
                "files_queued": job["files_queued"],
                "files_processing": job["files_processing"],
                "files_completed": job["files_completed"],
                "files_failed": job["files_failed"],
                "progress_percentage": progress_pct,
            },
            "vector_store_id": job["vector_store_id"],
            "storage_backend": job["storage_backend"],
            "embedding_model": job["embedding_model"],
            "files": files_status,
        }
        
        return ORJSONResponse(content=response_data, status_code=status.HTTP_200_OK)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}"
        )
