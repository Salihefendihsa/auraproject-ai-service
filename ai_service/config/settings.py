"""
Settings Module (v1.4.1)
Centralized configuration from environment variables.
"""

# ============================================================================
# DEMO BASELINE FREEZE v1.0
# ============================================================================
# This version is a HARD FREEZE as of 2024-12-23.
# All logic marked with "DO NOT MODIFY BASELINE LOGIC BELOW" is locked.
# Changes require explicit approval and versioning.
# ============================================================================
DEMO_BASELINE_VERSION = "v1.0"

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings from environment variables."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # LLM Configuration
    llm_enabled: bool = True
    llm_primary: str = "openai"
    llm_secondary: str = "gemini"
    llm_daily_limit: int = 200
    
    # Feature Flags
    user_photo_tryon_enabled: bool = False  # ISOLATED EXTENSION: User photo try-on mode
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            # API Keys
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            
            # LLM Configuration
            llm_enabled=os.getenv("AURA_LLM_ENABLED", "true").lower() == "true",
            llm_primary=os.getenv("AURA_LLM_PRIMARY", "openai").lower(),
            llm_secondary=os.getenv("AURA_LLM_SECONDARY", "gemini").lower(),
            llm_daily_limit=int(os.getenv("AURA_LLM_DAILY_LIMIT", "200")),
            
            # Feature Flags
            user_photo_tryon_enabled=os.getenv("AURA_USER_PHOTO_TRYON_ENABLED", "false").lower() == "true",
        )
    
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)
    
    def has_gemini(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.gemini_api_key)
    
    def to_dict(self) -> dict:
        """Export settings as dict (without sensitive keys)."""
        return {
            "llm_enabled": self.llm_enabled,
            "llm_primary": self.llm_primary,
            "llm_secondary": self.llm_secondary,
            "llm_daily_limit": self.llm_daily_limit,
            "openai_configured": self.has_openai(),
            "gemini_configured": self.has_gemini(),
        }


def get_settings() -> Settings:
    """Get application settings (cached singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        logger.info(f"Settings loaded: {_settings.to_dict()}")
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings.from_env()
    logger.info(f"Settings reloaded: {_settings.to_dict()}")
    return _settings


# Singleton instance
_settings: Optional[Settings] = None
