# src/celery_app.py - NEW FILE

from celery import Celery
from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "genai_platform_workers",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer=settings.celery_task_serializer,
    result_serializer=settings.celery_result_serializer,
    accept_content=settings.celery_accept_content,
    timezone=settings.celery_timezone,
    enable_utc=settings.celery_enable_utc,
    task_track_started=settings.celery_task_track_started,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
    worker_max_tasks_per_child=settings.celery_worker_max_tasks_per_child,
    task_routes={
        "src.celery_tasks.file_processing.*": {"queue": "file_processing"},
    },
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
)

celery_app.autodiscover_tasks(["src.celery_tasks"])
