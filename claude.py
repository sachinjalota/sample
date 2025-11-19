## 9. Usage Example
# Upload files
response = requests.post(
    "https://your-api.com/v1/api/vector_stores/{store_id}/files/batch",
    headers={
        "x-base-api-key": "your-api-key",
        "x-session-id": "session-123"
    },
    params={
        "storage_backend": "pgvector",
        "override": False
    },
    files=[
        ("files", ("doc1.pdf", open("doc1.pdf", "rb"), "application/pdf")),
        ("files", ("doc2.pdf", open("doc2.pdf", "rb"), "application/pdf")),
    ]
)

job_data = response.json()
# {
#   "job_id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "queued",
#   "total_files": 2,
#   "files_queued": 2,
#   "files_uploaded": [
#     {"file_id": "...", "task_id": "...", "file_name": "doc1.pdf", "status": "queued"},
#     {"file_id": "...", "task_id": "...", "file_name": "doc2.pdf", "status": "queued"}
#   ]
# }

# Check status
status_response = requests.get(
    f"https://your-api.com/v1/api/jobs/{job_data['job_id']}/status",
    headers={
        "x-base-api-key": "your-api-key",
        "x-session-id": "session-123"
    }
)

status_data = status_response.json()
# {
#   "job_id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "processing",
#   "progress": {
#     "total_files": 2,
#     "files_completed": 1,
#     "files_processing": 1,
#     "progress_percentage": 50.0
#   },
#   "files": [
#     {
#       "file_id": "...",
#       "file_name": "doc1.pdf",
#       "status": "completed",
#       "current_phase": "Successfully indexed to vector store",
#       "chunks_count": 45,
#       "usage_bytes": 123456
#     },
#     {
#       "file_id": "...",
#       "file_name": "doc2.pdf",
#       "status": "embedding",
#       "current_phase": "Generating embeddings",
#       "retry_count": 0
#     }
#   ]
# }

## 11. Enhanced Celery Tasks with Better Error Messages
# src/celery_tasks/file_processing.py - UPDATE ERROR HANDLING

from src.utility.error_messages import get_user_friendly_error

# In extract_file_content task, update exception handling:
except SoftTimeLimitExceeded:
    error_msg = get_user_friendly_error("extraction", "timeout")
    logger.error(f"[Task {task_id}] Extraction timeout")
    update_task_status(
        task_id,
        FileTaskStatus.FAILED,
        error_message=error_msg,
        error_details={"stage": "extraction", "reason": "timeout", "technical": "SoftTimeLimitExceeded"}
    )
    update_job_summary(task_context["job_id"])
    raise

except ValueError as e:
    if "empty" in str(e).lower():
        error_msg = get_user_friendly_error("extraction", "empty_content")
    else:
        error_msg = get_user_friendly_error("extraction", "corrupted", str(e))
    
    logger.error(f"[Task {task_id}] {error_msg}")
    update_task_status(
        task_id,
        FileTaskStatus.FAILED,
        error_message=error_msg,
        error_details={"stage": "extraction", "exception": str(e)}
    )
    update_job_summary(task_context["job_id"])
    raise

# Similar pattern for other tasks...

## 14. Testing Strategy
# tests/test_async_processing.py - NEW FILE

import pytest
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from src.celery_tasks.file_processing import (
    extract_file_content,
    chunk_text,
    generate_embeddings,
    index_to_vectorstore,
)
from src.db.platform_meta_tables import FileTaskStatus


class TestFileProcessingTasks:
    """Test suite for Celery file processing tasks"""
    
    @pytest.fixture
    def mock_task_context(self):
        """Sample task context"""
        return {
            "task_id": str(uuid4()),
            "job_id": str(uuid4()),
            "file_id": str(uuid4()),
            "file_name": "test.pdf",
            "gcs_path": "gs://bucket/test.pdf",
            "vector_store_id": str(uuid4()),
            "vector_store_name": "test_store",
            "usecase_id": "team123",
            "storage_backend": "pgvector",
            "embedding_model": "BAAI/bge-m3",
            "model_path": "gs://models/bge-m3",
            "embedding_dimensions": 1024,
            "context_length": 8192,
            "chunking_strategy": {"type": "auto"},
            "attributes": {},
            "file_version": 1,
        }
    
    @patch("src.celery_tasks.file_processing.CloudStorage")
    @patch("src.celery_tasks.file_processing.PDFExtractionService")
    def test_extract_pdf_success(self, mock_pdf_service, mock_cloud, mock_task_context):
        """Test successful PDF extraction"""
        # Setup mocks
        mock_cloud.return_value.download_object.return_value = b"fake pdf content"
        mock_pdf_service.return_value.extract_from_bytes.return_value = {
            "extracted_text": "This is extracted text from PDF"
        }
        
        # Execute
        result = extract_file_content(mock_task_context)
        
        # Assertions
        assert "extracted_text" in result
        assert len(result["extracted_text"]) > 0
        assert result["extracted_text"] == "This is extracted text from PDF"
    
    def test_chunk_text_success(self, mock_task_context):
        """Test successful text chunking"""
        mock_task_context["extracted_text"] = "This is a test document. " * 1000  # Long text
        
        # Execute
        result = chunk_text(mock_task_context)
        
        # Assertions
        assert "chunks" in result
        assert len(result["chunks"]) > 0
        assert all(isinstance(chunk, str) for chunk in result["chunks"])
    
    @patch("src.celery_tasks.file_processing.EmbeddingService")
    def test_generate_embeddings_success(self, mock_embedding_service, mock_task_context):
        """Test successful embedding generation"""
        mock_task_context["chunks"] = ["chunk1", "chunk2", "chunk3"]
        
        # Setup mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.data = [
            Mock(embedding=[0.1] * 1024),
            Mock(embedding=[0.2] * 1024),
            Mock(embedding=[0.3] * 1024),
        ]
        mock_embedding_service.return_value.get_embeddings = AsyncMock(return_value=mock_embeddings)
        
        # Execute
        result = generate_embeddings(mock_task_context)
        
        # Assertions
        assert "documents" in result
        assert len(result["documents"]) == 3
        assert all("embedding" in doc for doc in result["documents"])
    
    @patch("src.celery_tasks.file_processing.VectorStoreService")
    def test_indexing_with_rollback_on_failure(self, mock_vs_service, mock_task_context):
        """Test rollback on indexing failure"""
        mock_task_context["documents"] = [
            {"content": "chunk1", "embedding": [0.1] * 1024, "links": [], "topics": [], "author": None, "meta_data": {}}
        ]
        
        # Simulate indexing failure
        mock_vs_service.return_value.create_store_file = AsyncMock(side_effect=Exception("Index failed"))
        
        # Execute and expect exception
        with pytest.raises(Exception):
            index_to_vectorstore(mock_task_context)
        
        # Verify rollback was attempted (check logs or DB)
        # This would require checking that delete operations were called


class TestAsyncFileUploadAPI:
    """Test suite for async file upload endpoints"""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.main import app
        return TestClient(app)
    
    def test_batch_upload_success(self, client):
        """Test successful batch file upload"""
        # Create mock files
        files = [
            ("files", ("test1.pdf", b"fake pdf content", "application/pdf")),
            ("files", ("test2.pdf", b"fake pdf content", "application/pdf")),
        ]
        
        response = client.post(
            "/v1/api/vector_stores/test-store-id/files/batch",
            params={"storage_backend": "pgvector"},
            files=files,
            headers={
                "x-base-api-key": "test-key",
                "x-session-id": "test-session"
            }
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["total_files"] == 2
    
    def test_batch_upload_with_conflicts(self, client):
        """Test batch upload with version conflicts"""
        # Mock existing file
        with patch("src.api.routers.async_file_processing_router.check_file_version_conflict") as mock_check:
            mock_check.return_value = (True, 1)  # Conflict detected
            
            files = [("files", ("existing.pdf", b"content", "application/pdf"))]
            
            response = client.post(
                "/v1/api/vector_stores/test-store-id/files/batch",
                params={"storage_backend": "pgvector", "override": False},
                files=files,
                headers={"x-base-api-key": "test-key", "x-session-id": "test-session"}
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "conflicts" in data["detail"]
    
    def test_get_job_status(self, client):
        """Test job status retrieval"""
        job_id = str(uuid4())
        
        # Mock database records
        with patch("src.repository.base_repository.BaseRepository.select_one") as mock_select:
            mock_select.return_value = {
                "job_id": job_id,
                "usecase_id": "team123",
                "status": "processing",
                "total_files": 2,
                "files_completed": 1,
                "files_processing": 1,
                "created_at": "2025-01-01T00:00:00",
            }
            
            response = client.get(
                f"/v1/api/jobs/{job_id}/status",
                headers={"x-base-api-key": "test-key", "x-session-id": "test-session"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == job_id
            assert data["progress"]["progress_percentage"] == 50.0


# Run tests with: pytest tests/test_async_processing.py -v

## 16. Supervisor Configuration (Optional)
# supervisor.conf - Run FastAPI + Celery in same container

[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[program:fastapi]
command=uvicorn src.main:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/fastapi.err.log
stdout_logfile=/var/log/supervisor/fastapi.out.log

[program:celery_worker]
command=python celery_worker.py
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/celery.err.log
stdout_logfile=/var/log/supervisor/celery.out.log
stopwaitsecs=600
killasgroup=true

## 17. Dockerfile Update (if needed)
# Add to existing Dockerfile

# Install Celery
RUN pip install celery[redis]==5.3.4

# Copy supervisor config (if using supervisor)
COPY supervisor.conf /etc/supervisor/conf.d/

# Expose ports
EXPOSE 8000

# Start supervisor (runs both FastAPI + Celery)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
