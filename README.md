# src/db/platform_meta_tables.py - ADD THESE TABLES

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
import enum

class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"
    CANCELLED = "cancelled"

class FileTaskStatus(str, enum.Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"

class BatchUploadJob(BaseDBA):
    """Tracks batch file upload jobs"""
    __tablename__ = "batch_upload_jobs"
    
    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vector_store_id = Column(UUID(as_uuid=True), nullable=False)
    usecase_id = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=False), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=False), onupdate=func.now())
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    
    # Summary counts
    total_files = Column(Integer, nullable=False, default=0)
    files_queued = Column(Integer, nullable=False, default=0)
    files_processing = Column(Integer, nullable=False, default=0)
    files_completed = Column(Integer, nullable=False, default=0)
    files_failed = Column(Integer, nullable=False, default=0)
    
    # Metadata
    storage_backend = Column(String(50), nullable=False)
    embedding_model = Column(String(255), nullable=False)
    chunking_strategy = Column(JSONB, nullable=True)
    
    # Error tracking
    error_summary = Column(JSONB, nullable=True)  # {file_id: error_msg}


class FileProcessingTask(BaseDBA):
    """Tracks individual file processing tasks"""
    __tablename__ = "file_processing_tasks"
    
    task_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("batch_upload_jobs.job_id"), nullable=False)
    file_id = Column(UUID(as_uuid=True), nullable=False, unique=True)
    file_name = Column(String(255), nullable=False)
    gcs_path = Column(String(512), nullable=False)
    
    # Status tracking
    status = Column(Enum(FileTaskStatus), nullable=False, default=FileTaskStatus.QUEUED)
    current_phase = Column(String(50), nullable=True)  # Human-readable phase
    retry_count = Column(Integer, nullable=False, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=False), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=False), nullable=True)
    completed_at = Column(DateTime(timezone=False), nullable=True)
    
    # Intermediate data storage (for retry)
    extracted_text = Column(Text, nullable=True)  # Store on extraction success
    
    # Results
    chunks_count = Column(Integer, nullable=True)
    usage_bytes = Column(BigInteger, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)  # User-friendly error
    error_details = Column(JSONB, nullable=True)  # Technical details
    
    # Version tracking
    file_version = Column(Integer, nullable=False, default=1)


class CeleryTaskTracking(BaseDBA):
    """Optional: Track Celery task IDs for monitoring"""
    __tablename__ = "celery_task_tracking"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_task_id = Column(UUID(as_uuid=True), ForeignKey("file_processing_tasks.task_id"), nullable=False)
    celery_task_id = Column(String(255), nullable=False, unique=True)
    task_name = Column(String(255), nullable=False)  # e.g., "extract_pdf", "generate_embeddings"
    created_at = Column(DateTime(timezone=False), nullable=False, server_default=func.now())
