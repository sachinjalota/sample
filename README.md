# celery_worker.py - NEW FILE (root level)

from src.celery_app import celery_app
from src.logging_config import Logger

logger = Logger.create_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting Celery worker for GenAI Platform")
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "--concurrency=4",
        "--max-tasks-per-child=50",
        "-Q", "file_processing,default",
    ])
