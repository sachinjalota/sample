# src/celery_tasks/file_processing.py - NEW FILE

import traceback
from datetime import datetime
from typing import Any, Dict
from uuid import UUID
from zoneinfo import ZoneInfo

from celery import chain, group
from celery.exceptions import SoftTimeLimitExceeded

from src.celery_app import celery_app
from src.config import get_settings
from src.db.connection import create_session_platform
from src.db.platform_meta_tables import (
    BatchUploadJob,
    FileProcessingTask,
    FileTaskStatus,
    JobStatus,
)
from src.integrations.cloud_storage import CloudStorage
from src.logging_config import Logger
from src.repository.base_repository import BaseRepository
from src.services.embedding_service import EmbeddingService
from src.services.factory.chunking_factory import ChunkingConfig, ChunkingFactory
from src.services.pdf_extraction_service import PDFExtractionService
from src.services.service_layer.vector_store_service import VectorStoreService
from src.models.vector_store_payload import Document, CreateVectorStoreFileRequest

logger = Logger.create_logger(__name__)
settings = get_settings()


def update_task_status(
    task_id: str,
    status: FileTaskStatus,
    current_phase: str = None,
    error_message: str = None,
    error_details: Dict = None,
    **kwargs
) -> None:
    """Update file processing task status in DB"""
    try:
        update_data = {
            "status": status.value,
            "current_phase": current_phase,
        }
        
        if status == FileTaskStatus.FAILED:
            update_data["error_message"] = error_message
            update_data["error_details"] = error_details
            update_data["completed_at"] = datetime.now(ZoneInfo(settings.timezone))
        elif status == FileTaskStatus.COMPLETED:
            update_data["completed_at"] = datetime.now(ZoneInfo(settings.timezone))
        elif status in [FileTaskStatus.EXTRACTING, FileTaskStatus.CHUNKING, FileTaskStatus.EMBEDDING, FileTaskStatus.INDEXING]:
            if not kwargs.get("started_at"):
                update_data["started_at"] = datetime.now(ZoneInfo(settings.timezone))
        
        update_data.update(kwargs)
        
        BaseRepository.update_many(
            db_tbl=FileProcessingTask,
            filters={"task_id": task_id},
            data=update_data,
        )
    except Exception as e:
        logger.error(f"Failed to update task status for {task_id}: {e}")


def update_job_summary(job_id: str) -> None:
    """Recalculate and update batch job summary counts"""
    try:
        tasks = BaseRepository.select_many(
            db_tbl=FileProcessingTask,
            filters={"job_id": job_id},
        )
        
        status_counts = {
            "files_queued": 0,
            "files_processing": 0,
            "files_completed": 0,
            "files_failed": 0,
        }
        
        for task in tasks:
            if task["status"] == FileTaskStatus.QUEUED.value:
                status_counts["files_queued"] += 1
            elif task["status"] in [FileTaskStatus.EXTRACTING.value, FileTaskStatus.CHUNKING.value, 
                                    FileTaskStatus.EMBEDDING.value, FileTaskStatus.INDEXING.value]:
                status_counts["files_processing"] += 1
            elif task["status"] == FileTaskStatus.COMPLETED.value:
                status_counts["files_completed"] += 1
            elif task["status"] == FileTaskStatus.FAILED.value:
                status_counts["files_failed"] += 1
        
        # Determine overall job status
        total = len(tasks)
        if status_counts["files_completed"] == total:
            job_status = JobStatus.COMPLETED
        elif status_counts["files_failed"] == total:
            job_status = JobStatus.FAILED
        elif status_counts["files_failed"] > 0 and status_counts["files_completed"] > 0:
            job_status = JobStatus.PARTIAL_FAILURE
        elif status_counts["files_processing"] > 0:
            job_status = JobStatus.PROCESSING
        else:
            job_status = JobStatus.QUEUED
        
        BaseRepository.update_many(
            db_tbl=BatchUploadJob,
            filters={"job_id": job_id},
            data={
                "status": job_status.value,
                "updated_at": datetime.now(ZoneInfo(settings.timezone)),
                **status_counts
            },
        )
    except Exception as e:
        logger.error(f"Failed to update job summary for {job_id}: {e}")


@celery_app.task(bind=True, name="src.celery_tasks.file_processing.extract_file_content")
def extract_file_content(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 1: Extract text content from PDF/DOCX files
    Returns: Updated task_context with extracted_text
    """
    task_id = task_context["task_id"]
    file_name = task_context["file_name"]
    gcs_path = task_context["gcs_path"]
    
    logger.info(f"[Task {task_id}] Starting extraction for {file_name}")
    update_task_status(task_id, FileTaskStatus.EXTRACTING, "Extracting text from file")
    
    try:
        # Download file from GCS
        cloud_storage = CloudStorage()
        file_content = cloud_storage.download_object(gcs_path)
        
        # Extract based on file type
        if file_name.lower().endswith('.pdf'):
            pdf_service = PDFExtractionService()
            result = pdf_service.extract_from_bytes(file_content)
            extracted_text = result.get("extracted_text", "")
        elif file_name.lower().endswith(('.docx', '.doc')):
            # Use existing DOCX conversion logic
            from src.services.file_upload_service import FileUploadService
            upload_service = FileUploadService()
            # Convert DOCX to PDF, then extract
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            pdf_bytes = upload_service.convert_to_pdf(file_content, file_name)
            pdf_service = PDFExtractionService()
            result = pdf_service.extract_from_bytes(pdf_bytes.read())
            extracted_text = result.get("extracted_text", "")
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise ValueError("Extracted text is empty or too short")
        
        # Store extracted text for potential retry
        update_task_status(
            task_id,
            FileTaskStatus.EXTRACTING,
            "Text extraction completed",
            extracted_text=extracted_text
        )
        
        task_context["extracted_text"] = extracted_text
        logger.info(f"[Task {task_id}] Extraction completed: {len(extracted_text)} chars")
        return task_context
        
    except SoftTimeLimitExceeded:
        error_msg = f"File extraction timed out after {settings.celery_task_soft_time_limit}s"
        logger.error(f"[Task {task_id}] {error_msg}")
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=error_msg,
            error_details={"stage": "extraction", "reason": "timeout"}
        )
        update_job_summary(task_context["job_id"])
        raise
    except Exception as e:
        error_msg = f"Failed to extract content: {str(e)}"
        logger.exception(f"[Task {task_id}] {error_msg}")
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=error_msg,
            error_details={
                "stage": "extraction",
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
        )
        update_job_summary(task_context["job_id"])
        raise


@celery_app.task(bind=True, name="src.celery_tasks.file_processing.chunk_text")
def chunk_text(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 2: Chunk extracted text using configured strategy
    Returns: Updated task_context with chunks
    """
    task_id = task_context["task_id"]
    extracted_text = task_context.get("extracted_text")
    
    if not extracted_text:
        # Try to retrieve from DB (retry scenario)
        task_record = BaseRepository.select_one(
            db_tbl=FileProcessingTask,
            filters={"task_id": task_id},
        )
        extracted_text = task_record.get("extracted_text") if task_record else None
        
        if not extracted_text:
            raise ValueError("No extracted text available for chunking")
    
    logger.info(f"[Task {task_id}] Starting chunking")
    update_task_status(task_id, FileTaskStatus.CHUNKING, "Chunking text into segments")
    
    try:
        chunking_strategy = task_context.get("chunking_strategy", {"type": "auto"})
        
        if chunking_strategy["type"] == "static":
            config = ChunkingConfig(
                chunk_size=chunking_strategy.get("max_chunk_size_tokens", 2048),
                overlap=chunking_strategy.get("chunk_overlap_tokens", 256),
            )
            strategy = ChunkingFactory.create("recursive", config=config)
        else:  # auto
            config = ChunkingConfig(chunk_size=2048, overlap=256)
            strategy = ChunkingFactory.create("recursive", config=config)
        
        # Chunk synchronously (no await needed)
        import asyncio
        chunks = asyncio.run(strategy.chunk(extracted_text))
        
        if not chunks or len(chunks) == 0:
            raise ValueError("Chunking produced no results")
        
        task_context["chunks"] = chunks
        update_task_status(
            task_id,
            FileTaskStatus.CHUNKING,
            f"Chunking completed: {len(chunks)} chunks created"
        )
        
        logger.info(f"[Task {task_id}] Chunking completed: {len(chunks)} chunks")
        return task_context
        
    except SoftTimeLimitExceeded:
        error_msg = "Text chunking timed out"
        logger.error(f"[Task {task_id}] {error_msg}")
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=error_msg,
            error_details={"stage": "chunking", "reason": "timeout"}
        )
        update_job_summary(task_context["job_id"])
        raise
    except Exception as e:
        error_msg = f"Failed to chunk text: {str(e)}"
        logger.exception(f"[Task {task_id}] {error_msg}")
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=error_msg,
            error_details={
                "stage": "chunking",
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
        )
        update_job_summary(task_context["job_id"])
        raise


@celery_app.task(bind=True, name="src.celery_tasks.file_processing.generate_embeddings")
def generate_embeddings(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 3: Generate embeddings for chunks
    Returns: Updated task_context with documents (chunks + embeddings)
    """
    task_id = task_context["task_id"]
    chunks = task_context.get("chunks", [])
    
    logger.info(f"[Task {task_id}] Starting embedding generation for {len(chunks)} chunks")
    update_task_status(task_id, FileTaskStatus.EMBEDDING, "Generating embeddings")
    
    try:
        from src.api.deps import get_openai_service_internal
        from src.models.vector_store_payload import Document
        
        embedding_service = EmbeddingService(open_ai_sdk=get_openai_service_internal())
        embedding_model = task_context["embedding_model"]
        
        # Generate embeddings in batch
        import asyncio
        embeddings_result = asyncio.run(
            embedding_service.get_embeddings(model_name=embedding_model, batch=chunks)
        )
        
        # Construct documents with embeddings
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "embedding": embeddings_result.data[i].embedding,
                "links": [task_context["file_name"]],
                "topics": [],
                "author": None,
                "meta_data": {
                    "file_id": task_context["file_id"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        task_context["documents"] = documents
        update_task_status(
            task_id,
            FileTaskStatus.EMBEDDING,
            f"Embeddings generated for {len(documents)} chunks"
        )
        
        logger.info(f"[Task {task_id}] Embedding generation completed")
        return task_context
        
    except SoftTimeLimitExceeded:
        error_msg = "Embedding generation timed out"
        logger.error(f"[Task {task_id}] {error_msg}")
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=error_msg,
            error_details={"stage": "embedding", "reason": "timeout"}
        )
        update_job_summary(task_context["job_id"])
        raise
    except Exception as e:
        error_msg = f"Failed to generate embeddings: {str(e)}"
        logger.exception(f"[Task {task_id}] {error_msg}")
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=error_msg,
            error_details={
                "stage": "embedding",
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
        )
        update_job_summary(task_context["job_id"])
        raise


@celery_app.task(bind=True, name="src.celery_tasks.file_processing.index_to_vectorstore")
def index_to_vectorstore(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task 4: Index documents to vector store (PGVector or Elasticsearch)
    Returns: Final task_context with indexing results
    """
    task_id = task_context["task_id"]
    documents = task_context.get("documents", [])
    
    logger.info(f"[Task {task_id}] Starting indexing of {len(documents)} documents")
    update_task_status(task_id, FileTaskStatus.INDEXING, "Indexing to vector store")
    
    try:
        from src.api.deps import get_openai_service_internal
        from src.models.vector_store_payload import CreateVectorStoreFileRequest, Document as VectorStoreDocument
        
        # Convert documents to VectorStoreDocument format
        file_contents = [
            VectorStoreDocument(
                content=doc["content"],
                links=doc.get("links"),
                topics=doc.get("topics"),
                author=doc.get("author"),
                metadata=doc.get("meta_data")
            )
            for doc in documents
        ]
        
        # Create indexing request
        index_request = CreateVectorStoreFileRequest(
            storage_backend=task_context["storage_backend"],
            file_id=task_context["file_id"],
            file_name=task_context["file_name"],
            file_contents=file_contents,
            attributes=task_context.get("attributes", {}),
            chunking_strategy=task_context.get("chunking_strategy")
        )
        
        # Index via VectorStoreService
        embedding_service = EmbeddingService(open_ai_sdk=get_openai_service_internal())
        vector_service = VectorStoreService(
            backend_name=task_context["storage_backend"],
            embedding_service=embedding_service
        )
        
        import asyncio
        result = asyncio.run(
            vector_service.create_store_file(
                payload=index_request,
                store_id=task_context["vector_store_id"],
                usecase_id=task_context["usecase_id"],
                model_name=task_context["embedding_model"],
                context_length=task_context["context_length"],
                model_path=task_context["model_path"],
                embedding_dimensions=task_context["embedding_dimensions"]
            )
        )
        
        # Update task with final results
        update_task_status(
            task_id,
            FileTaskStatus.COMPLETED,
            "Successfully indexed to vector store",
            chunks_count=len(documents),
            usage_bytes=result.usage_bytes
        )
        
        # Update job summary
        update_job_summary(task_context["job_id"])
        
        # Optional: Delete from GCS if configured
        if settings.delete_gcs_after_indexing:
            try:
                cloud_storage = CloudStorage()
                # Implement delete logic here
                logger.info(f"[Task {task_id}] Would delete GCS file: {task_context['gcs_path']}")
            except Exception as del_err:
                logger.warning(f"[Task {task_id}] Failed to delete GCS file: {del_err}")
        
        logger.info(f"[Task {task_id}] Indexing completed successfully")
        return task_context
        
    except SoftTimeLimitExceeded:
        error_msg = "Indexing to vector store timed out"
        logger.error(f"[Task {task_id}] {error_msg}")
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=error_msg,
            error_details={"stage": "indexing", "reason": "timeout"}
        )
        update_job_summary(task_context["job_id"])
        raise
    except Exception as e:
        error_msg = f"Failed to index to vector store: {str(e)}"
        logger.exception(f"[Task {task_id}] {error_msg}")
        
        # Rollback: Delete any partially indexed data
        try:
            from src.repository.document_repository import DocumentRepository
            store_name = task_context["vector_store_name"]
            
            if task_context["storage_backend"] == "pgvector":
                from src.utility.vector_store_utils import create_chunks_tbl_model, create_file_info_tbl_model
                
                # Delete chunks
                vs_chunks_tbl = create_chunks_tbl_model(f"{store_name}_chunks", dimensions=0)
                BaseRepository.delete(
                    db_tbl=vs_chunks_tbl,
                    filters={"file_id": task_context["file_id"]},
                    session_factory=create_session,
                )
                
                # Delete file info
                vs_file_info_tbl = create_file_info_tbl_model(f"{store_name}_file_info")
                BaseRepository.delete(
                    db_tbl=vs_file_info_tbl,
                    filters={"file_id": task_context["file_id"]},
                    session_factory=create_session,
                )
                logger.info(f"[Task {task_id}] Rolled back partial indexing (PGVector)")
                
            elif task_context["storage_backend"] == "elasticsearch":
                from src.repository.elasticsearch_dml import ElasticsearchDML
                
                chunks_index = f"{store_name}_chunks"
                file_info_index = f"{store_name}_file_info"
                
                # Delete chunks
                chunk_query = {"query": {"term": {"file_id": task_context["file_id"]}}}
                ElasticsearchDML.delete(index_name=chunks_index, query=chunk_query)
                
                # Delete file info
                ElasticsearchDML.delete(index_name=file_info_index, doc_id=task_context["file_id"])
                logger.info(f"[Task {task_id}] Rolled back partial indexing (Elasticsearch)")
                
        except Exception as rollback_err:
            logger.error(f"[Task {task_id}] Rollback failed: {rollback_err}")
        
        update_task_status(
            task_id,
            FileTaskStatus.FAILED,
            error_message=f"{error_msg} (Partial data rolled back)",
            error_details={
                "stage": "indexing",
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
        )
        update_job_summary(task_context["job_id"])
        raise


@celery_app.task(name="src.celery_tasks.file_processing.process_single_file")
def process_single_file(task_context: Dict[str, Any]) -> None:
    """
    Orchestrator task: Chains all file processing steps
    """
    task_id = task_context["task_id"]
    logger.info(f"[Task {task_id}] Starting file processing chain")
    
    try:
        # Create task chain
        processing_chain = chain(
            extract_file_content.s(task_context),
            chunk_text.s(),
            generate_embeddings.s(),
            index_to_vectorstore.s()
        )
        
        # Execute chain with retry logic
        result = processing_chain.apply_async(
            retry=True,
            retry_policy={
                'max_retries': settings.max_task_retries,
                'interval_start': settings.task_retry_delay,
                'interval_step': settings.task_retry_delay,
            }
        )
        
        return result.get()  # Wait for chain completion
        
    except Exception as e:
        logger.exception(f"[Task {task_id}] Processing chain failed: {e}")
        # Final failure state is already set by individual tasks
        update_job_summary(task_context["job_id"])
        raise
