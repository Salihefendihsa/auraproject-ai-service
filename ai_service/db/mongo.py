"""
MongoDB Connection Module (v1.4.3)
Persistent job storage using MongoDB.
"""
import os
import logging
from typing import Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "aura_ai")

# Global client
_client = None
_db = None


def connect() -> bool:
    """
    Connect to MongoDB.
    
    Returns:
        True if connected, False otherwise
    """
    global _client, _db
    
    try:
        from pymongo import MongoClient
        
        logger.info(f"Connecting to MongoDB: {MONGO_URI[:30]}...")
        
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection
        _client.admin.command('ping')
        
        _db = _client[MONGO_DB_NAME]
        
        logger.info(f"✓ Connected to MongoDB database: {MONGO_DB_NAME}")
        return True
        
    except ImportError:
        logger.warning("pymongo not installed - MongoDB disabled")
        return False
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        return False


def get_collection(name: str = "jobs"):
    """Get a MongoDB collection."""
    global _db
    
    if _db is None:
        connect()
    
    if _db is None:
        return None
    
    return _db[name]


def health_check() -> dict:
    """Check MongoDB connection health."""
    global _client
    
    try:
        if _client is None:
            connect()
        
        if _client:
            _client.admin.command('ping')
            return {"status": "connected", "uri": MONGO_URI[:30] + "..."}
        else:
            return {"status": "disconnected", "reason": "client not initialized"}
            
    except Exception as e:
        return {"status": "disconnected", "reason": str(e)}


def insert_job(job_data: dict) -> bool:
    """Insert a new job document."""
    try:
        collection = get_collection("jobs")
        if collection is None:
            return False
        
        collection.insert_one(job_data)
        logger.info(f"Job inserted: {job_data.get('job_id')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert job: {e}")
        return False


def update_job(job_id: str, update_data: dict) -> bool:
    """Update an existing job document."""
    try:
        collection = get_collection("jobs")
        if collection is None:
            return False
        
        collection.update_one(
            {"job_id": job_id},
            {"$set": update_data}
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to update job {job_id}: {e}")
        return False


def get_job(job_id: str) -> Optional[dict]:
    """Get a job document by ID."""
    try:
        collection = get_collection("jobs")
        if collection is None:
            return None
        
        job = collection.find_one({"job_id": job_id})
        
        if job:
            # Remove MongoDB _id for JSON serialization
            job.pop("_id", None)
            return job
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        return None


def create_job_document(job_id: str) -> dict:
    """Create initial job document structure."""
    return {
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending",
        "detected_clothing": {},
        "attributes": {},
        "outfits": [],
        "assets": {
            "input_image": "",
            "masks": {},
            "renders": []
        },
        "provider_used": "",
        "cached": False,
        "error": None
    }
