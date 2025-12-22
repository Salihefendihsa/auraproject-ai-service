"""
LLM Initialization Adapter (v2.5.0)
Unified LLM client with Gemini 3 Pro support and automatic fallback.

Supports both OpenAI and Gemini providers with role-based configuration.
"""
import os
import json
import logging
from typing import Optional, Any, Dict
from functools import wraps

from ai_service.config.llm_config import (
    get_llm_config, get_planner_config, get_judge_config,
    LLMProvider, LLMRole, ActiveLLMConfig
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client interface for any provider/model.
    
    Usage:
        client = LLMClient(role=LLMRole.PLANNER)
        client.initialize()
        response = await client.generate_text(system, user)
    """
    
    def __init__(self, role: LLMRole = LLMRole.PLANNER):
        self.role = role
        self.config = get_llm_config(role)
        self._openai_client = None
        self._gemini_model = None
        self._current_model = None
        self._fallback_used = False
        self._initialized = False
    
    def initialize(self, use_fallback: bool = False):
        """Initialize the LLM client based on configuration."""
        model = self.config.resolve_model(use_fallback)
        self._fallback_used = use_fallback
        self._current_model = model
        
        if self.config.is_openai():
            self._init_openai(model)
        elif self.config.is_gemini():
            self._init_gemini(model)
        
        self._initialized = True
    
    def _init_openai(self, model: str):
        """Initialize OpenAI client."""
        from openai import AsyncOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self._openai_client = AsyncOpenAI(api_key=api_key)
        logger.info(f"OpenAI [{self.role.value}]: model={model}, fallback={self._fallback_used}")
    
    def _init_gemini(self, model: str):
        """Initialize Gemini client."""
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self._gemini_model = genai.GenerativeModel(model)
        logger.info(f"Gemini [{self.role.value}]: model={model}, fallback={self._fallback_used}")
    
    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text response."""
        if not self._initialized:
            self.initialize()
        
        try:
            return await self._generate_impl(system_prompt, user_prompt, json_mode=False)
        except Exception as e:
            if not self._fallback_used:
                logger.warning(f"Primary model failed ({e}), trying fallback...")
                self.initialize(use_fallback=True)
                return await self._generate_impl(system_prompt, user_prompt, json_mode=False)
            raise
    
    async def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate JSON response with robust parsing."""
        if not self._initialized:
            self.initialize()
        
        try:
            text = await self._generate_impl(system_prompt, user_prompt, json_mode=True)
            return self._parse_json(text)
        except Exception as e:
            if not self._fallback_used:
                logger.warning(f"Primary model failed ({e}), trying fallback...")
                self.initialize(use_fallback=True)
                text = await self._generate_impl(system_prompt, user_prompt, json_mode=True)
                return self._parse_json(text)
            raise
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks."""
        text = text.strip()
        
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text.strip())
    
    async def _generate_impl(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        """Internal generation implementation."""
        if self.config.is_openai():
            return await self._generate_openai(system_prompt, user_prompt, json_mode)
        elif self.config.is_gemini():
            return await self._generate_gemini(system_prompt, user_prompt, json_mode)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def _generate_openai(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        """Generate using OpenAI."""
        kwargs = {
            "model": self._current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self._openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    async def _generate_gemini(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        """Generate using Gemini."""
        combined = f"{system_prompt}\n\n{user_prompt}"
        
        if json_mode:
            combined += "\n\nRespond with valid JSON only, no markdown code blocks."
        
        response = self._gemini_model.generate_content(combined)
        return response.text
    
    async def generate_with_images(
        self,
        prompt: str,
        images: list,
        json_mode: bool = False
    ) -> str:
        """Generate with image inputs (Gemini vision)."""
        if not self._initialized:
            self.initialize()
        
        if self.config.is_gemini():
            return await self._generate_gemini_vision(prompt, images, json_mode)
        else:
            # OpenAI vision (if needed in future)
            raise NotImplementedError("OpenAI vision not implemented")
    
    async def _generate_gemini_vision(self, prompt: str, images: list, json_mode: bool) -> str:
        """Generate using Gemini with images."""
        if json_mode:
            prompt += "\n\nRespond with valid JSON only, no markdown."
        
        content = [prompt] + images
        response = self._gemini_model.generate_content(content)
        return response.text
    
    def get_status(self) -> dict:
        """Get current client status."""
        return {
            "role": self.role.value,
            "provider": self.config.provider.value,
            "model": self._current_model,
            "fallback_used": self._fallback_used,
            "initialized": self._initialized
        }


# ==================== SINGLETON INSTANCES ====================

_planner_client: Optional[LLMClient] = None
_judge_client: Optional[LLMClient] = None


def get_llm_client(role: LLMRole = LLMRole.PLANNER) -> LLMClient:
    """Get initialized LLM client for a role."""
    global _planner_client, _judge_client
    
    if role == LLMRole.JUDGE:
        if _judge_client is None:
            _judge_client = LLMClient(role=LLMRole.JUDGE)
            _judge_client.initialize()
        return _judge_client
    else:
        if _planner_client is None:
            _planner_client = LLMClient(role=LLMRole.PLANNER)
            _planner_client.initialize()
        return _planner_client


def get_planner_client() -> LLMClient:
    """Get planner (outfit generation) client."""
    return get_llm_client(LLMRole.PLANNER)


def get_judge_client() -> LLMClient:
    """Get judge (visual evaluation) client."""
    return get_llm_client(LLMRole.JUDGE)


def reset_llm_clients():
    """Reset all clients (for testing)."""
    global _planner_client, _judge_client
    _planner_client = None
    _judge_client = None


def get_all_clients_status() -> dict:
    """Get all client statuses for /health endpoint."""
    return {
        "planner": get_planner_client().get_status(),
        "judge": get_judge_client().get_status()
    }
