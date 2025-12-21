# Database module
from ai_service.db.mongo import (
    connect,
    get_collection,
    health_check,
    insert_job,
    update_job,
    get_job,
    create_job_document
)
