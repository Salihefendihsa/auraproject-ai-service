"""
Storage Manager for job data.
"""
import logging
import uuid
from pathlib import Path
from typing import BinaryIO

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


# Global instance
storage = StorageManager()
