"""
Authentication Module (v2.0.0)
Simple API Key authentication for AuraProject AI Service.
"""
import os
import secrets
import logging
from typing import Optional
from datetime import datetime
from fastapi import Header, HTTPException, Request

from ai_service.db import mongo

logger = logging.getLogger(__name__)

# Configuration
API_KEY_HEADER = "X-API-Key"
API_KEY_LENGTH = 32
BYPASS_AUTH = os.getenv("AURA_BYPASS_AUTH", "false").lower() == "true"


class User:
    """Authenticated user representation."""
    
    def __init__(self, user_id: str, name: str, api_key: str, created_at: str):
        self.user_id = user_id
        self.name = name
        self.api_key = api_key
        self.created_at = created_at
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at
        }


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"aura_{secrets.token_hex(API_KEY_LENGTH)}"


def create_user(name: str) -> Optional[dict]:
    """
    Create a new user with API key.
    
    Args:
        name: User display name
    
    Returns:
        User document with API key (only shown once!)
    """
    try:
        collection = mongo.get_collection("users")
        if collection is None:
            logger.error("MongoDB not available for user creation")
            return None
        
        user_id = secrets.token_hex(8)
        api_key = generate_api_key()
        
        user_doc = {
            "user_id": user_id,
            "name": name,
            "api_key": api_key,
            "created_at": datetime.utcnow().isoformat(),
            "active": True,
            "rate_limit_tier": "default",
            "concurrent_jobs": 0
        }
        
        collection.insert_one(user_doc)
        logger.info(f"User created: {user_id} ({name})")
        
        # Return with api_key visible (only this once!)
        return {
            "user_id": user_id,
            "name": name,
            "api_key": api_key,
            "message": "Save this API key - it won't be shown again!"
        }
        
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return None


def get_user_by_api_key(api_key: str) -> Optional[User]:
    """
    Look up user by API key.
    
    Args:
        api_key: The API key to validate
    
    Returns:
        User object if found and active, None otherwise
    """
    try:
        collection = mongo.get_collection("users")
        if collection is None:
            return None
        
        user_doc = collection.find_one({"api_key": api_key, "active": True})
        
        if user_doc:
            return User(
                user_id=user_doc["user_id"],
                name=user_doc["name"],
                api_key=user_doc["api_key"],
                created_at=user_doc["created_at"]
            )
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get user by API key: {e}")
        return None


def get_user_concurrent_jobs(user_id: str) -> int:
    """Get current concurrent job count for user."""
    try:
        collection = mongo.get_collection("users")
        if collection is None:
            return 0
        
        user = collection.find_one({"user_id": user_id})
        return user.get("concurrent_jobs", 0) if user else 0
        
    except Exception as e:
        logger.error(f"Failed to get concurrent jobs: {e}")
        return 0


def increment_concurrent_jobs(user_id: str) -> bool:
    """Increment concurrent job count for user."""
    try:
        collection = mongo.get_collection("users")
        if collection is None:
            return False
        
        collection.update_one(
            {"user_id": user_id},
            {"$inc": {"concurrent_jobs": 1}}
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to increment concurrent jobs: {e}")
        return False


def decrement_concurrent_jobs(user_id: str) -> bool:
    """Decrement concurrent job count for user."""
    try:
        collection = mongo.get_collection("users")
        if collection is None:
            return False
        
        collection.update_one(
            {"user_id": user_id},
            {"$inc": {"concurrent_jobs": -1}}
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to decrement concurrent jobs: {e}")
        return False


async def get_current_user(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> User:
    """
    FastAPI dependency for authentication.
    
    Usage:
        @router.post("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            ...
    
    Raises:
        HTTPException 401: If no API key or invalid API key
    """
    # Bypass auth for development
    if BYPASS_AUTH:
        logger.warning("Auth bypass enabled - DEV MODE")
        return User(
            user_id="dev_user",
            name="Development User",
            api_key="dev_key",
            created_at=datetime.utcnow().isoformat()
        )
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    user = get_user_by_api_key(x_api_key)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    logger.debug(f"Authenticated user: {user.user_id}")
    return user


async def get_optional_user(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[User]:
    """
    FastAPI dependency for optional authentication.
    Returns None if no API key provided.
    """
    if not x_api_key:
        return None
    
    return get_user_by_api_key(x_api_key)


def check_job_ownership(job: dict, user: User) -> bool:
    """
    Check if user owns the job.
    
    Args:
        job: Job document from MongoDB
        user: Current authenticated user
    
    Returns:
        True if user owns job or job has no owner
    """
    owner_id = job.get("owner_user_id")
    
    # Jobs without owner are accessible to all (legacy)
    if owner_id is None:
        return True
    
    return owner_id == user.user_id
