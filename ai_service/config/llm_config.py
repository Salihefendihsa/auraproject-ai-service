"""
LLM Configuration Layer (v2.5.0)
Model-agnostic config for Planner (outfit generation) and Judge (visual evaluation).

Environment Variables:
  PLANNER (outfit generation):
    - AURA_LLM_PROVIDER: "openai" | "gemini" (default: openai)
    - AURA_LLM_MODEL: Override default model (optional)
    - AURA_LLM_FALLBACK_MODEL: Override fallback model (optional)
  
  JUDGE (visual evaluation):
    - AURA_JUDGE_PROVIDER: "openai" | "gemini" (default: gemini)
    - AURA_JUDGE_MODEL: Override default model (optional)
    - AURA_JUDGE_FALLBACK_MODEL: Override fallback model (optional)
"""
import os
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ==================== ENUMS ====================

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMRole(Enum):
    """LLM usage role."""
    PLANNER = "planner"      # Outfit generation
    JUDGE = "judge"          # Visual evaluation
    REGENERATOR = "regenerator"  # Outfit regeneration


# ==================== PROVIDER CONFIGS ====================

@dataclass
class OpenAIConfig:
    """OpenAI model configuration."""
    default_model: str = "gpt-4o-mini"
    future_model: str = "gpt-5"
    fallback_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2500
    available_models: tuple = field(default_factory=lambda: (
        "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-5",
    ))


@dataclass
class GeminiConfig:
    """Gemini model configuration."""
    default_model: str = "gemini-1.5-pro"
    future_model: str = "gemini-3-pro"
    fallback_model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 2500
    available_models: tuple = field(default_factory=lambda: (
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-3-pro",
    ))


# ==================== ACTIVE CONFIG ====================

@dataclass
class ActiveLLMConfig:
    """Active LLM configuration for a specific role."""
    role: LLMRole
    provider: LLMProvider
    model: str
    fallback_model: str
    temperature: float
    max_tokens: int
    
    @classmethod
    def from_env(cls, role: LLMRole = LLMRole.PLANNER) -> "ActiveLLMConfig":
        """Resolve configuration from environment variables."""
        
        if role == LLMRole.JUDGE:
            # Judge defaults to Gemini
            provider_env = "AURA_JUDGE_PROVIDER"
            model_env = "AURA_JUDGE_MODEL"
            fallback_env = "AURA_JUDGE_FALLBACK_MODEL"
            default_provider = "gemini"
        else:
            # Planner/Regenerator defaults to OpenAI
            provider_env = "AURA_LLM_PROVIDER"
            model_env = "AURA_LLM_MODEL"
            fallback_env = "AURA_LLM_FALLBACK_MODEL"
            default_provider = "openai"
        
        provider_str = os.getenv(provider_env, default_provider).lower()
        
        if provider_str == "gemini":
            provider = LLMProvider.GEMINI
            defaults = GeminiConfig()
        else:
            provider = LLMProvider.OPENAI
            defaults = OpenAIConfig()
        
        model = os.getenv(model_env, defaults.default_model)
        fallback = os.getenv(fallback_env, defaults.fallback_model)
        temperature = float(os.getenv("AURA_LLM_TEMPERATURE", str(defaults.temperature)))
        max_tokens = int(os.getenv("AURA_LLM_MAX_TOKENS", str(defaults.max_tokens)))
        
        config = cls(
            role=role,
            provider=provider,
            model=model,
            fallback_model=fallback,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(f"LLM Config [{role.value}]: provider={provider.value}, model={model}")
        return config
    
    def resolve_model(self, use_fallback: bool = False) -> str:
        return self.fallback_model if use_fallback else self.model
    
    def is_openai(self) -> bool:
        return self.provider == LLMProvider.OPENAI
    
    def is_gemini(self) -> bool:
        return self.provider == LLMProvider.GEMINI
    
    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "provider": self.provider.value,
            "model": self.model,
            "fallback_model": self.fallback_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# ==================== SINGLETON INSTANCES ====================

_planner_config: Optional[ActiveLLMConfig] = None
_judge_config: Optional[ActiveLLMConfig] = None


def get_llm_config(role: LLMRole = LLMRole.PLANNER) -> ActiveLLMConfig:
    """Get active LLM configuration for a role."""
    global _planner_config, _judge_config
    
    if role == LLMRole.JUDGE:
        if _judge_config is None:
            _judge_config = ActiveLLMConfig.from_env(LLMRole.JUDGE)
        return _judge_config
    else:
        if _planner_config is None:
            _planner_config = ActiveLLMConfig.from_env(LLMRole.PLANNER)
        return _planner_config


def get_planner_config() -> ActiveLLMConfig:
    """Get planner (outfit generation) config."""
    return get_llm_config(LLMRole.PLANNER)


def get_judge_config() -> ActiveLLMConfig:
    """Get judge (visual evaluation) config."""
    return get_llm_config(LLMRole.JUDGE)


def reset_llm_config():
    """Reset all configs (for testing)."""
    global _planner_config, _judge_config
    _planner_config = None
    _judge_config = None


def get_all_configs_dict() -> dict:
    """Get all configs as dict for /health endpoint."""
    return {
        "planner": get_planner_config().to_dict(),
        "judge": get_judge_config().to_dict()
    }


# ==================== CONTROLNET CONFIG (v2.6.0) ====================

class ControlNetType(Enum):
    """ControlNet conditioning type."""
    POSE = "pose"
    EDGE = "edge"
    DEPTH = "depth"


@dataclass
class ControlNetConfig:
    """
    ControlNet configuration for pose locking.
    
    Environment Variables:
    - AURA_CONTROLNET_ENABLED: true|false (default: false)
    - AURA_CONTROLNET_TYPE: pose|edge|depth (default: pose)
    - AURA_CONTROLNET_SCALE: float (default: 1.0)
    """
    enabled: bool = False
    control_type: ControlNetType = ControlNetType.POSE
    model_pose: str = "lllyasviel/control_v11p_sd15_openpose"
    model_edge: str = "lllyasviel/control_v11p_sd15_canny"
    model_depth: str = "lllyasviel/control_v11f1p_sd15_depth"
    conditioning_scale: float = 1.0
    
    @classmethod
    def from_env(cls) -> "ControlNetConfig":
        """Load ControlNet config from environment."""
        enabled = os.getenv("AURA_CONTROLNET_ENABLED", "false").lower() == "true"
        
        type_str = os.getenv("AURA_CONTROLNET_TYPE", "pose").lower()
        try:
            control_type = ControlNetType(type_str)
        except ValueError:
            control_type = ControlNetType.POSE
        
        scale = float(os.getenv("AURA_CONTROLNET_SCALE", "1.0"))
        
        config = cls(
            enabled=enabled,
            control_type=control_type,
            conditioning_scale=scale
        )
        
        if enabled:
            logger.info(f"ControlNet Config: type={control_type.value}, scale={scale}")
        
        return config
    
    def get_model_id(self) -> str:
        """Get the model ID for current control type."""
        if self.control_type == ControlNetType.POSE:
            return self.model_pose
        elif self.control_type == ControlNetType.EDGE:
            return self.model_edge
        elif self.control_type == ControlNetType.DEPTH:
            return self.model_depth
        return self.model_pose
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "type": self.control_type.value,
            "conditioning_scale": self.conditioning_scale
        }


_controlnet_config: Optional[ControlNetConfig] = None


def get_controlnet_config() -> ControlNetConfig:
    """Get ControlNet configuration (singleton)."""
    global _controlnet_config
    if _controlnet_config is None:
        _controlnet_config = ControlNetConfig.from_env()
    return _controlnet_config


def reset_controlnet_config():
    """Reset ControlNet config (for testing)."""
    global _controlnet_config
    _controlnet_config = None
