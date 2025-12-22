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
from ai_service.db.history import (
    get_user_history,
    get_history_count,
    add_favorite,
    get_favorites,
    remove_favorite,
    add_feedback,
    get_user_feedback_stats,
)
from ai_service.db.wardrobe import (
    create_wardrobe_item,
    get_wardrobe_items,
    get_wardrobe_item,
    delete_wardrobe_item,
    find_duplicate_in_wardrobe,
    get_wardrobe_count,
    get_wardrobe_by_category,
    build_wardrobe_context,
    compute_phash,
)
