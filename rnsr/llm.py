"""
LLM Configuration - Multi-Provider LLM and Embedding Support

Supports:
- OpenAI (GPT-4, text-embedding-3-small)
- Anthropic (Claude)
- Google Gemini (gemini-pro, text-embedding-004)

Usage:
    from rnsr.llm import get_llm, get_embed_model, LLMProvider
    
    # Auto-detect based on environment variables
    llm = get_llm()
    embed = get_embed_model()
    
    # Or specify provider explicitly
    llm = get_llm(provider=LLMProvider.GEMINI)
    embed = get_embed_model(provider=LLMProvider.GEMINI)
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    
    # Look for .env in the project root (parent of rnsr package)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system environment

logger = structlog.get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AUTO = "auto"  # Auto-detect from environment


# Default models per provider (updated January 2026)
DEFAULT_MODELS = {
    LLMProvider.OPENAI: {
        "llm": "gpt-4.1-mini",  # Fast, affordable - use "gpt-5-mini" for latest
        "embed": "text-embedding-3-small",
    },
    LLMProvider.ANTHROPIC: {
        "llm": "claude-sonnet-4-5",  # Smart model for agents/coding
        "embed": None,  # Anthropic doesn't have embeddings, fall back to OpenAI/Gemini
    },
    LLMProvider.GEMINI: {
        "llm": "gemini-2.5-flash",  # Best price-performance, use "gemini-3-flash" for latest
        "embed": "text-embedding-004",
    },
}


def detect_provider() -> LLMProvider:
    """
    Auto-detect LLM provider from environment variables.
    
    Checks for API keys in order:
    1. GOOGLE_API_KEY -> Gemini
    2. ANTHROPIC_API_KEY -> Anthropic
    3. OPENAI_API_KEY -> OpenAI
    
    Returns:
        Detected LLMProvider.
        
    Raises:
        ValueError: If no API key is found.
    """
    if os.getenv("GOOGLE_API_KEY"):
        logger.info("provider_detected", provider="gemini")
        return LLMProvider.GEMINI
    
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("provider_detected", provider="anthropic")
        return LLMProvider.ANTHROPIC
    
    if os.getenv("OPENAI_API_KEY"):
        logger.info("provider_detected", provider="openai")
        return LLMProvider.OPENAI
    
    raise ValueError(
        "No LLM API key found. Set one of: "
        "GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY"
    )


def get_llm(
    provider: LLMProvider = LLMProvider.AUTO,
    model: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Get an LLM instance for the specified provider.
    
    Args:
        provider: LLM provider (openai, anthropic, gemini, or auto).
        model: Model name override. Uses default if not specified.
        **kwargs: Additional arguments passed to the LLM constructor.
        
    Returns:
        LlamaIndex-compatible LLM instance.
        
    Example:
        llm = get_llm(provider=LLMProvider.GEMINI)
        response = await llm.acomplete("Hello!")
    """
    if provider == LLMProvider.AUTO:
        provider = detect_provider()
    
    model = model or DEFAULT_MODELS[provider]["llm"]
    
    if provider == LLMProvider.OPENAI:
        return _get_openai_llm(model, **kwargs)
    elif provider == LLMProvider.ANTHROPIC:
        return _get_anthropic_llm(model, **kwargs)
    elif provider == LLMProvider.GEMINI:
        return _get_gemini_llm(model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embed_model(
    provider: LLMProvider = LLMProvider.AUTO,
    model: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Get an embedding model for the specified provider.
    
    Args:
        provider: LLM provider (openai, gemini, or auto).
        model: Model name override. Uses default if not specified.
        **kwargs: Additional arguments passed to the embedding constructor.
        
    Returns:
        LlamaIndex-compatible embedding model.
        
    Note:
        Anthropic doesn't have embeddings. Falls back to OpenAI or Gemini.
        
    Example:
        embed = get_embed_model(provider=LLMProvider.GEMINI)
        vector = embed.get_text_embedding("Hello world")
    """
    if provider == LLMProvider.AUTO:
        provider = detect_provider()
    
    # Anthropic doesn't have embeddings, fall back
    if provider == LLMProvider.ANTHROPIC:
        if os.getenv("GOOGLE_API_KEY"):
            provider = LLMProvider.GEMINI
            logger.info("anthropic_no_embeddings", fallback="gemini")
        elif os.getenv("OPENAI_API_KEY"):
            provider = LLMProvider.OPENAI
            logger.info("anthropic_no_embeddings", fallback="openai")
        else:
            raise ValueError(
                "Anthropic doesn't provide embeddings. "
                "Set GOOGLE_API_KEY or OPENAI_API_KEY for embeddings."
            )
    
    model = model or DEFAULT_MODELS[provider]["embed"]
    
    if provider == LLMProvider.OPENAI:
        return _get_openai_embed(model, **kwargs)
    elif provider == LLMProvider.GEMINI:
        return _get_gemini_embed(model, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# =============================================================================
# Provider-Specific Implementations
# =============================================================================


def _get_openai_llm(model: str, **kwargs: Any) -> Any:
    """Get OpenAI LLM instance."""
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI LLM not installed. "
            "Install with: pip install llama-index-llms-openai"
        )
    
    logger.debug("initializing_llm", provider="openai", model=model)
    return OpenAI(model=model, **kwargs)


def _get_anthropic_llm(model: str, **kwargs: Any) -> Any:
    """Get Anthropic LLM instance."""
    try:
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "Anthropic LLM not installed. "
            "Install with: pip install llama-index-llms-anthropic"
        )
    
    logger.debug("initializing_llm", provider="anthropic", model=model)
    return Anthropic(model=model, **kwargs)


def _get_gemini_llm(model: str, **kwargs: Any) -> Any:
    """Get Google Gemini LLM instance using the new google-genai SDK."""
    logger.debug("initializing_llm", provider="gemini", model=model)
    
    # Try the new google-genai SDK first (recommended)
    try:
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Create a wrapper that matches LlamaIndex LLM interface
        class GeminiWrapper:
            """Wrapper for google-genai to match LlamaIndex LLM interface."""
            
            def __init__(self, model_name: str, api_key: str):
                self.client = genai.Client(api_key=api_key)
                self.model_name = model_name
            
            def complete(self, prompt: str, **kw: Any) -> str:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text or ""
            
            def chat(self, messages: list, **kw: Any) -> str:
                # Convert to genai format
                contents = []
                for msg in messages:
                    role = "user" if msg.get("role") == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                ) 
                return response.text or ""
        
        return GeminiWrapper(model, api_key)
        
    except ImportError:
        # Fall back to llama-index-llms-gemini (deprecated)
        try:
            from llama_index.llms.gemini import Gemini
            return Gemini(model=model, **kwargs)
        except ImportError:
            raise ImportError(
                "Neither google-genai nor llama-index-llms-gemini installed. "
                "Install with: pip install google-genai"
            )


def _get_openai_embed(model: str, **kwargs: Any) -> Any:
    """Get OpenAI embedding model."""
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError:
        raise ImportError(
            "OpenAI embeddings not installed. "
            "Install with: pip install llama-index-embeddings-openai"
        )
    
    logger.debug("initializing_embed", provider="openai", model=model)
    return OpenAIEmbedding(model=model, **kwargs)


def _get_gemini_embed(model: str, **kwargs: Any) -> Any:
    """Get Google Gemini embedding model."""
    try:
        from llama_index.embeddings.gemini import GeminiEmbedding
    except ImportError:
        raise ImportError(
            "Gemini embeddings not installed. "
            "Install with: pip install llama-index-embeddings-gemini"
        )
    
    logger.debug("initializing_embed", provider="gemini", model=model)
    return GeminiEmbedding(model_name=f"models/{model}", **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_provider_info() -> dict[str, Any]:
    """
    Get information about available providers.
    
    Returns:
        Dictionary with provider availability and configuration.
    """
    info = {
        "available": [],
        "default_provider": None,
        "models": DEFAULT_MODELS,
    }
    
    if os.getenv("OPENAI_API_KEY"):
        info["available"].append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        info["available"].append("anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        info["available"].append("gemini")
    
    if info["available"]:
        try:
            info["default_provider"] = detect_provider().value
        except ValueError:
            pass
    
    return info


def validate_provider(provider: LLMProvider) -> bool:
    """
    Check if a provider is available (has API key set).
    
    Args:
        provider: Provider to check.
        
    Returns:
        True if provider is available.
    """
    if provider == LLMProvider.OPENAI:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == LLMProvider.ANTHROPIC:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == LLMProvider.GEMINI:
        return bool(os.getenv("GOOGLE_API_KEY"))
    elif provider == LLMProvider.AUTO:
        return any([
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
        ])
    return False
