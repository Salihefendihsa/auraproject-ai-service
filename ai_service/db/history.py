"""
History & Favorites Module (v2.1.0)
Manages outfit history and user favorites in MongoDB.
"""
import logging
from typing import Optional, List
from datetime import datetime

from ai_service.db import mongo

logger = logging.getLogger(__name__)


# ==================== OUTFIT HISTORY ====================

def get_user_history(
    user_id: str,
    limit: int = 20,
    offset: int = 0
) -> List[dict]:
    """
    Get user's outfit generation history.
    
    Args:
        user_id: Owner user ID
        limit: Max results (default 20)
        offset: Skip results for pagination
    
    Returns:
        List of job summaries
    """
    try:
        collection = mongo.get_collection("jobs")
        if collection is None:
            return []
        
        cursor = collection.find(
            {"owner_user_id": user_id},
            {
                "_id": 0,
                "job_id": 1,
                "created_at": 1,
                "event": 1,
                "city": 1,
                "weather": 1,
                "status": 1,
                "outfits": {"$slice": 1}  # Just first outfit for summary
            }
        ).sort("created_at", -1).skip(offset).limit(limit)
        
        history = []
        for job in cursor:
            # Create summary
            outfit_count = len(job.get("outfits", []))
            first_outfit = job.get("outfits", [{}])[0] if job.get("outfits") else {}
            
            history.append({
                "job_id": job.get("job_id"),
                "created_at": job.get("created_at"),
                "event": job.get("event"),
                "city": job.get("city"),
                "weather_condition": job.get("weather", {}).get("condition") if job.get("weather") else None,
                "status": job.get("status"),
                "outfit_count": outfit_count,
                "style_preview": first_outfit.get("style_tag", "")
            })
        
        return history
        
    except Exception as e:
        logger.error(f"Failed to get history for {user_id}: {e}")
        return []


def get_history_count(user_id: str) -> int:
    """Get total history count for pagination."""
    try:
        collection = mongo.get_collection("jobs")
        if collection is None:
            return 0
        
        return collection.count_documents({"owner_user_id": user_id})
        
    except Exception as e:
        logger.error(f"Failed to count history: {e}")
        return 0


# ==================== FAVORITES ====================

def add_favorite(
    user_id: str,
    job_id: str,
    outfit_index: int,
    note: Optional[str] = None
) -> Optional[dict]:
    """
    Save an outfit as favorite.
    
    Args:
        user_id: Owner user ID
        job_id: Job containing the outfit
        outfit_index: Index of outfit in the job (1-based)
        note: Optional user note
    
    Returns:
        Favorite document or None
    """
    try:
        # Verify job ownership
        jobs_collection = mongo.get_collection("jobs")
        if jobs_collection is None:
            return None
        
        job = jobs_collection.find_one({"job_id": job_id})
        if not job:
            logger.warning(f"Job not found: {job_id}")
            return None
        
        if job.get("owner_user_id") != user_id:
            logger.warning(f"Job ownership mismatch: {job_id}")
            return None
        
        # Get the specific outfit
        outfits = job.get("outfits", [])
        if outfit_index < 1 or outfit_index > len(outfits):
            logger.warning(f"Invalid outfit index: {outfit_index}")
            return None
        
        outfit = outfits[outfit_index - 1]
        
        # Create favorite document
        favorites_collection = mongo.get_collection("favorites")
        if favorites_collection is None:
            return None
        
        favorite = {
            "user_id": user_id,
            "job_id": job_id,
            "outfit_index": outfit_index,
            "outfit_snapshot": outfit,
            "note": note,
            "created_at": datetime.utcnow().isoformat(),
            "event": job.get("event"),
            "city": job.get("city")
        }
        
        # Check for duplicate
        existing = favorites_collection.find_one({
            "user_id": user_id,
            "job_id": job_id,
            "outfit_index": outfit_index
        })
        
        if existing:
            logger.info(f"Favorite already exists: {job_id}:{outfit_index}")
            existing.pop("_id", None)
            return existing
        
        favorites_collection.insert_one(favorite)
        logger.info(f"Favorite added: {job_id}:{outfit_index}")
        
        favorite.pop("_id", None)
        return favorite
        
    except Exception as e:
        logger.error(f"Failed to add favorite: {e}")
        return None


def get_favorites(
    user_id: str,
    limit: int = 50,
    offset: int = 0
) -> List[dict]:
    """
    Get user's favorite outfits.
    
    Args:
        user_id: Owner user ID
        limit: Max results
        offset: Skip for pagination
    
    Returns:
        List of favorite documents
    """
    try:
        collection = mongo.get_collection("favorites")
        if collection is None:
            return []
        
        cursor = collection.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("created_at", -1).skip(offset).limit(limit)
        
        return list(cursor)
        
    except Exception as e:
        logger.error(f"Failed to get favorites: {e}")
        return []


def remove_favorite(user_id: str, job_id: str, outfit_index: int) -> bool:
    """Remove a favorite."""
    try:
        collection = mongo.get_collection("favorites")
        if collection is None:
            return False
        
        result = collection.delete_one({
            "user_id": user_id,
            "job_id": job_id,
            "outfit_index": outfit_index
        })
        
        return result.deleted_count > 0
        
    except Exception as e:
        logger.error(f"Failed to remove favorite: {e}")
        return False


# ==================== FEEDBACK ====================

def add_feedback(
    user_id: str,
    job_id: str,
    outfit_index: int,
    feedback_type: str,
    reason: Optional[str] = None
) -> Optional[dict]:
    """
    Add feedback (like/dislike) for an outfit.
    
    Args:
        user_id: Owner user ID
        job_id: Job containing the outfit
        outfit_index: Index of outfit (1-based)
        feedback_type: "like" or "dislike"
        reason: Optional reason (especially for dislikes)
    
    Returns:
        Feedback document or None
    """
    try:
        # Verify job ownership
        jobs_collection = mongo.get_collection("jobs")
        if jobs_collection is None:
            return None
        
        job = jobs_collection.find_one({"job_id": job_id})
        if not job or job.get("owner_user_id") != user_id:
            return None
        
        feedback_collection = mongo.get_collection("feedback")
        if feedback_collection is None:
            return None
        
        feedback = {
            "user_id": user_id,
            "job_id": job_id,
            "outfit_index": outfit_index,
            "feedback_type": feedback_type,
            "reason": reason,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Upsert - update if exists
        feedback_collection.update_one(
            {
                "user_id": user_id,
                "job_id": job_id,
                "outfit_index": outfit_index
            },
            {"$set": feedback},
            upsert=True
        )
        
        logger.info(f"Feedback added: {job_id}:{outfit_index} - {feedback_type}")
        return feedback
        
    except Exception as e:
        logger.error(f"Failed to add feedback: {e}")
        return None


def get_user_feedback_stats(user_id: str) -> dict:
    """Get feedback statistics for a user."""
    try:
        collection = mongo.get_collection("feedback")
        if collection is None:
            return {"likes": 0, "dislikes": 0, "top_dislike_reasons": []}
        
        likes = collection.count_documents({"user_id": user_id, "feedback_type": "like"})
        dislikes = collection.count_documents({"user_id": user_id, "feedback_type": "dislike"})
        
        # Get top dislike reasons
        pipeline = [
            {"$match": {"user_id": user_id, "feedback_type": "dislike", "reason": {"$ne": None}}},
            {"$group": {"_id": "$reason", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        
        reasons = list(collection.aggregate(pipeline))
        top_reasons = [{"reason": r["_id"], "count": r["count"]} for r in reasons]
        
        return {
            "likes": likes,
            "dislikes": dislikes,
            "top_dislike_reasons": top_reasons
        }
        
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        return {"likes": 0, "dislikes": 0, "top_dislike_reasons": []}
