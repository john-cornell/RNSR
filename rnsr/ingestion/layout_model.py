"""
LayoutLM Model - Visual Document Structure Analysis

Implements LayoutLMv3 for multimodal document understanding using:
- Text: Token sequences from OCR
- Layout: 2D bounding box coordinates
- Image: Visual features from document patches

The model "sees" that text is bold, 24pt font, and centered, allowing
it to classify blocks as Header, Body, Title, Caption, etc.

Usage:
    from rnsr.ingestion.layout_model import get_layout_model, classify_layout_blocks
    
    # Auto-load default model (layoutlmv3-large)
    model = get_layout_model()
    
    # Or specify model explicitly
    model = get_layout_model(model_name="microsoft/layoutlmv3-base")
    
    # Classify document blocks
    labels = classify_layout_blocks(page_image, bboxes, text_spans)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# =============================================================================
# Model Configuration
# =============================================================================

# Default models
LAYOUT_MODEL_BASE = "microsoft/layoutlmv3-base"  # 133M params, 400MB
LAYOUT_MODEL_LARGE = "microsoft/layoutlmv3-large"  # 368M params, 1.2GB

DEFAULT_LAYOUT_MODEL = LAYOUT_MODEL_LARGE  # Large by default for 16GB+ RAM

# Label mapping for document structure
LABEL_NAMES = [
    "O",          # Other/None
    "B-TITLE",    # Beginning of title
    "I-TITLE",    # Inside title
    "B-HEADER",   # Beginning of header
    "I-HEADER",   # Inside header
    "B-BODY",     # Beginning of body text
    "I-BODY",     # Inside body text
    "B-CAPTION",  # Beginning of caption
    "I-CAPTION",  # Inside caption
    "B-FOOTER",   # Beginning of footer
    "I-FOOTER",   # Inside footer
    "B-TABLE",    # Beginning of table
    "I-TABLE",    # Inside table
]

# =============================================================================
# Global Model Cache
# =============================================================================

_LAYOUT_MODEL_CACHE: dict[str, Any] = {}


def detect_device() -> str:
    """
    Auto-detect best available device for inference.
    
    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)
    
    Returns:
        Device string ("cuda", "mps", or "cpu").
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info("device_detected", device="cuda", gpus=torch.cuda.device_count())
            return "cuda"
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("device_detected", device="mps", note="Apple Silicon GPU")
            return "mps"
        
        logger.info("device_detected", device="cpu", note="No GPU available")
        return "cpu"
        
    except ImportError:
        logger.warning("torch_not_installed", fallback="cpu")
        return "cpu"


def get_model_name_from_env() -> str:
    """
    Get model name from environment variable or default.
    
    Environment variables:
    - RNSR_LAYOUT_MODEL: Model name or path
      Examples: "microsoft/layoutlmv3-base", "microsoft/layoutlmv3-large"
    
    Returns:
        Model name or path.
    """
    model = os.getenv("RNSR_LAYOUT_MODEL")
    if model:
        logger.info("layout_model_from_env", model=model)
        return model
    
    return DEFAULT_LAYOUT_MODEL


def get_device_from_env() -> str:
    """
    Get device from environment variable or auto-detect.
    
    Environment variables:
    - RNSR_LAYOUT_DEVICE: Device override
      Options: "cuda", "mps", "cpu", "auto"
    
    Returns:
        Device string.
    """
    device = os.getenv("RNSR_LAYOUT_DEVICE", "auto").lower()
    
    if device == "auto":
        return detect_device()
    
    if device not in ("cuda", "mps", "cpu"):
        logger.warning("invalid_device", device=device, fallback="auto")
        return detect_device()
    
    logger.info("layout_device_from_env", device=device)
    return device


def get_layout_model(
    model_name: str | None = None,
    device: str | None = None,
    force_reload: bool = False,
) -> Any:
    """
    Get LayoutLMv3 model instance with caching.
    
    Args:
        model_name: Model name or path. Uses env var or default if None.
        device: Device for inference ("cuda", "mps", "cpu", "auto").
        force_reload: Force reload model even if cached.
        
    Returns:
        LayoutLMv3 model instance.
        
    Raises:
        ImportError: If transformers or torch not installed.
        RuntimeError: If model cannot be loaded.
        
    Example:
        # Default (layoutlmv3-large, auto device)
        model = get_layout_model()
        
        # Custom model
        model = get_layout_model(model_name="microsoft/layoutlmv3-base")
        
        # Force CPU
        model = get_layout_model(device="cpu")
    """
    # Resolve model name and device
    model_name = model_name or get_model_name_from_env()
    device = device or get_device_from_env()
    
    # Check cache
    cache_key = f"{model_name}:{device}"
    if not force_reload and cache_key in _LAYOUT_MODEL_CACHE:
        logger.debug("layout_model_from_cache", model=model_name, device=device)
        return _LAYOUT_MODEL_CACHE[cache_key]
    
    # Import dependencies
    try:
        from transformers import AutoModelForTokenClassification, AutoProcessor
        import torch
    except ImportError as e:
        raise ImportError(
            "transformers and torch required for LayoutLM. "
            "Install with: pip install transformers torch torchvision"
        ) from e
    
    logger.info(
        "loading_layout_model",
        model=model_name,
        device=device,
        note="First load downloads ~1.2GB" if "large" in model_name else "First load downloads ~400MB"
    )
    
    try:
        # Load model and processor
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL_NAMES),
        )
        processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        
        # Move to device
        if device != "cpu":
            model = model.to(device)
        
        model.eval()  # Set to evaluation mode
        
        # Cache model and processor together
        _LAYOUT_MODEL_CACHE[cache_key] = {
            "model": model,
            "processor": processor,
            "device": device,
        }
        
        logger.info("layout_model_loaded", model=model_name, device=device)
        return _LAYOUT_MODEL_CACHE[cache_key]
        
    except Exception as e:
        logger.error("layout_model_load_failed", model=model_name, error=str(e))
        raise RuntimeError(f"Failed to load LayoutLM model: {e}") from e


def classify_layout_blocks(
    page_image: Image.Image,
    bboxes: list[tuple[float, float, float, float]],
    text_spans: list[str],
    model_name: str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Classify layout blocks using LayoutLMv3.
    
    Args:
        page_image: PIL Image of the document page.
        bboxes: List of bounding boxes as (x0, y0, x1, y1) tuples.
        text_spans: List of text content for each bounding box.
        model_name: Model name override (uses default if None).
        device: Device override (uses auto-detect if None).
        
    Returns:
        List of classification results with structure:
        [
            {
                "text": str,
                "bbox": tuple,
                "label": str,  # "TITLE", "HEADER", "BODY", etc.
                "confidence": float,
            },
            ...
        ]
        
    Example:
        from PIL import Image
        
        image = Image.open("page.png")
        bboxes = [(10, 10, 100, 30), (10, 50, 200, 70)]
        texts = ["Document Title", "This is the introduction."]
        
        results = classify_layout_blocks(image, bboxes, texts)
        for r in results:
            print(f"{r['label']}: {r['text']}")
    """
    if len(bboxes) != len(text_spans):
        raise ValueError("Number of bboxes must match number of text_spans")
    
    if not bboxes:
        logger.warning("no_bboxes_to_classify")
        return []
    
    # Load model
    model_dict = get_layout_model(model_name, device)
    model = model_dict["model"]
    processor = model_dict["processor"]
    device_str = model_dict["device"]
    
    try:
        import torch
        
        # Normalize bboxes to 0-1000 scale (LayoutLM format)
        width, height = page_image.size
        normalized_bboxes = []
        for x0, y0, x1, y1 in bboxes:
            normalized_bboxes.append([
                int((x0 / width) * 1000),
                int((y0 / height) * 1000),
                int((x1 / width) * 1000),
                int((y1 / height) * 1000),
            ])
        
        # Prepare inputs
        encoding = processor(
            page_image,
            text_spans,
            boxes=normalized_bboxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        
        # Move to device
        if device_str != "cpu":
            encoding = {k: v.to(device_str) for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Extract results
        results = []
        for i, (text, bbox) in enumerate(zip(text_spans, bboxes)):
            pred_idx = int(predictions[0, i].item())
            confidence = float(probabilities[0, i, pred_idx].item())
            
            label = LABEL_NAMES[pred_idx] if pred_idx < len(LABEL_NAMES) else "O"
            
            # Simplify label (remove B-/I- prefix)
            simplified_label = label.split("-")[-1] if "-" in label else label
            
            results.append({
                "text": text,
                "bbox": bbox,
                "label": simplified_label,
                "confidence": confidence,
            })
        
        logger.debug("layout_classification_complete", blocks=len(results))
        return results
        
    except Exception as e:
        logger.error("layout_classification_failed", error=str(e))
        raise RuntimeError(f"Failed to classify layout blocks: {e}") from e


def check_layout_model_available() -> bool:
    """
    Check if LayoutLM dependencies are available.
    
    Returns:
        True if transformers and torch are installed.
    """
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def get_layout_model_info() -> dict[str, Any]:
    """
    Get information about LayoutLM configuration.
    
    Returns:
        Dictionary with model configuration and availability.
    """
    info = {
        "available": check_layout_model_available(),
        "default_model": DEFAULT_LAYOUT_MODEL,
        "models": {
            "base": LAYOUT_MODEL_BASE,
            "large": LAYOUT_MODEL_LARGE,
        },
        "device": get_device_from_env(),
        "env_model": os.getenv("RNSR_LAYOUT_MODEL"),
        "env_device": os.getenv("RNSR_LAYOUT_DEVICE"),
        "cache_dir": os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    }
    
    return info
