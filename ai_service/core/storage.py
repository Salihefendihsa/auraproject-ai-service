"""
Storage Manager for job data.

============================================================================
CDN READINESS DOCUMENTATION
============================================================================

This module manages the file storage structure for job data and assets.
All paths are designed to be CDN-friendly with predictable, absolute URLs.

STORAGE STRUCTURE (CDN CANDIDATES):
-----------------------------------
base_data_dir/
├── jobs/{job_id}/
│   ├── renders/           <- CDN CANDIDATE: rendered outfit images
│   │   └── outfit_*.png      Served at: /ai/assets/jobs/{job_id}/renders/*
│   ├── masks/             <- CDN CANDIDATE: segmentation masks
│   │   └── *.png             Served at: /ai/assets/jobs/{job_id}/masks/*
│   ├── input.jpg          <- CDN CANDIDATE: input image
│   │   └──                   Served at: /ai/assets/jobs/{job_id}/input.jpg
│   └── job.json           <- NOT CDN: job metadata
└── wardrobe/{user_id}/    <- CDN CANDIDATE: user wardrobe items
    └── *.png                 Served at: /ai/assets/wardrobe/{user_id}/*

RENDER_URL FORMAT:
------------------
All render_url values follow this absolute path pattern:
  /ai/assets/jobs/{job_id}/renders/{filename}

This ensures:
- Predictable, cacheable URLs for CDN
- Path traversal protection via sanitize_asset_path()
- No need for URL rewriting at CDN level

============================================================================
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Optional

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file storage for jobs."""
    
    def __init__(self, base_dir: str = None):
        # Use path relative to this file
        if base_dir is None:
            module_dir = Path(__file__).parent.parent
            base_dir = module_dir / "data"
        self.base_data_dir = Path(base_dir)
        self.jobs_dir = self.base_data_dir / "jobs"
    
    def ensure_directories(self):
        """Create required directories."""
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Data directories initialized: {self.base_data_dir}")
    
    def create_job(self) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        job_path = self.jobs_dir / job_id
        job_path.mkdir(parents=True, exist_ok=True)
        (job_path / "masks").mkdir(exist_ok=True)
        return job_id
    
    def get_job_path(self, job_id: str) -> Path:
        """Get path to job directory."""
        return self.jobs_dir / job_id
    
    def save_input_image(self, job_id: str, file: BinaryIO) -> Path:
        """Save uploaded image to job directory."""
        job_path = self.get_job_path(job_id)
        job_path.mkdir(parents=True, exist_ok=True)
        
        image_path = job_path / "input.jpg"
        
        # Read and save
        content = file.read()
        with open(image_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved input image: {image_path}")
        return image_path
    
    def get_file_path(self, relative_path: str) -> Path:
        """Get full path for a relative path, with security check."""
        full_path = (self.base_data_dir / relative_path).resolve()
        base_resolved = self.base_data_dir.resolve()
        
        # Security: ensure path is within base directory
        if not str(full_path).startswith(str(base_resolved)):
            raise ValueError("Path traversal not allowed")
        
        return full_path
    
    # ==================== SEED JOB HELPERS ====================
    
    def create_seed_job(self, job_id: str) -> Path:
        """
        Create a seed job folder structure.
        
        Args:
            job_id: UUID for the job
        
        Returns:
            Path to job directory
        """
        job_path = self.jobs_dir / job_id
        job_path.mkdir(parents=True, exist_ok=True)
        (job_path / "masks").mkdir(exist_ok=True)
        (job_path / "renders").mkdir(exist_ok=True)
        logger.info(f"Created seed job folder: {job_path}")
        return job_path
    
    def save_seed_job_json(
        self,
        job_id: str,
        seed_image_path: Optional[str],
        person_image_path: Optional[str],
        gender: str,
        event: Optional[str],
        season: Optional[str],
        mode: str,
        subject_type: str
    ) -> Path:
        """
        Save job.json for a seed job.
        
        Args:
            job_id: Job UUID
            seed_image_path: Path to seed image (if provided)
            person_image_path: Path to person image (if provided)
            gender: male or female
            event: Optional event type
            season: Optional season
            mode: mock, partial_tryon, or full_tryon
            subject_type: person or mannequin
        
        Returns:
            Path to job.json
        """
        job_path = self.get_job_path(job_id)
        job_json_path = job_path / "job.json"
        
        data = {
            "job_id": job_id,
            "version": "3.0.0",
            "inputs": {
                "seed_image_path": seed_image_path,
                "person_image_path": person_image_path,
                "gender": gender,
                "event": event,
                "season": season,
                "mode": mode
            },
            "subject_type": subject_type,
            "current_stage": "initialized",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        with open(job_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved job.json: {job_json_path}")
        return job_json_path


# Global instance
storage = StorageManager()
