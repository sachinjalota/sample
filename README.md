# src/config.py - ADD THESE SETTINGS

class Settings(BaseSettings):
    # ... existing settings ...
    
    # Celery Configuration
    celery_broker_url: str = f"rediss://:{redis_auth_string}@{redis_host}:{redis_port}/1"
    celery_result_backend: str = f"rediss://:{redis_auth_string}@{redis_host}:{redis_port}/2"
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_accept_content: list = ["json"]
    celery_timezone: str = "Asia/Kolkata"
    celery_enable_utc: bool = True
    celery_task_track_started: bool = True
    celery_task_time_limit: int = 1800  # 30 minutes
    celery_task_soft_time_limit: int = 1700  # 28 minutes (warning)
    celery_worker_prefetch_multiplier: int = 1  # Process one task at a time
    celery_worker_max_tasks_per_child: int = 50  # Restart worker after 50 tasks
    
    # File Processing Limits
    max_concurrent_files_per_vs: int = 5
    max_file_size_bytes: int = 52428800  # 50MB
    allowed_file_types: list = [".pdf", ".docx", ".doc"]
    
    # Retry Configuration
    max_task_retries: int = 1
    task_retry_delay: int = 60  # seconds
    
    # GCS Management
    delete_gcs_after_indexing: bool = False  # Configurable flag
    
    # Async Job Endpoints
    batch_upload_endpoint: str = "/vector_stores/{store_id}/files/batch"
    job_status_endpoint: str = "/jobs/{job_id}/status"
