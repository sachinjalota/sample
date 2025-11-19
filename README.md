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
