"""
Test script for LayoutLM visual analysis integration.

Tests:
1. Layout complexity detection
2. LayoutLM model loading
3. Visual analysis classification
"""

from pathlib import Path

from rnsr.ingestion import (
    detect_layout_complexity,
    get_layout_model_info,
    check_layout_model_available,
)


def test_layout_model_info():
    """Test that LayoutLM is available and configured correctly."""
    print("\n=== Layout Model Info ===")
    info = get_layout_model_info()
    
    print(f"LayoutLM Available: {info['available']}")
    print(f"Default Model: {info['default_model']}")
    print(f"Device: {info['device']}")
    print(f"Cache Directory: {info['cache_dir']}")
    
    if info['available']:
        print("✓ LayoutLM dependencies installed")
    else:
        print("✗ LayoutLM dependencies missing (install transformers and torch)")
    
    return info['available']


def test_complexity_detection():
    """Test layout complexity detection on sample PDF."""
    print("\n=== Layout Complexity Detection ===")
    
    # Find a test PDF
    test_dir = Path(__file__).parent / "benchmark_results"
    if not test_dir.exists():
        print("No benchmark results directory found, skipping complexity test")
        return
    
    # Look for any PDF in the workspace
    workspace = Path(__file__).parent
    pdfs = list(workspace.rglob("*.pdf"))
    
    if not pdfs:
        print("No PDF files found in workspace for testing")
        return
    
    test_pdf = pdfs[0]
    print(f"Testing: {test_pdf.name}")
    
    try:
        complexity = detect_layout_complexity(test_pdf, threshold=0.3)
        
        print(f"Complexity Score: {complexity.complexity_score:.2f}")
        print(f"Needs Visual Analysis: {complexity.needs_visual_analysis}")
        print(f"Reason: {complexity.reason}")
        print(f"Multi-column: {complexity.has_multi_column}")
        print(f"Empty Pages: {complexity.has_empty_pages} ({complexity.empty_page_ratio:.1%})")
        print(f"Complex Wrapping: {complexity.has_complex_wrapping}")
        print(f"BBox Overlap Score: {complexity.bbox_overlap_score:.2f}")
        
        if complexity.needs_visual_analysis:
            print("✓ Would trigger LayoutLM visual analysis")
        else:
            print("✓ Would use simple font histogram")
            
    except Exception as e:
        print(f"✗ Complexity detection failed: {e}")


def test_model_loading():
    """Test that LayoutLM model can be loaded."""
    print("\n=== Layout Model Loading ===")
    
    if not check_layout_model_available():
        print("✗ LayoutLM not available (dependencies missing)")
        return
    
    try:
        from rnsr.ingestion.layout_model import get_layout_model
        
        print("Loading layoutlmv3-large model (this may take a few minutes on first run)...")
        model_dict = get_layout_model()
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {model_dict['device']}")
        print(f"  Model type: {type(model_dict['model']).__name__}")
        print(f"  Processor type: {type(model_dict['processor']).__name__}")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")


def test_environment_config():
    """Test environment variable configuration."""
    print("\n=== Environment Configuration ===")
    import os
    
    layout_model = os.getenv("RNSR_LAYOUT_MODEL")
    layout_device = os.getenv("RNSR_LAYOUT_DEVICE")
    hf_home = os.getenv("HF_HOME")
    
    print(f"RNSR_LAYOUT_MODEL: {layout_model or '(default: layoutlmv3-large)'}")
    print(f"RNSR_LAYOUT_DEVICE: {layout_device or '(auto-detect)'}")
    print(f"HF_HOME: {hf_home or '(default: ~/.cache/huggingface)'}")
    
    print("\nTo configure:")
    print("  export RNSR_LAYOUT_MODEL='microsoft/layoutlmv3-base'  # Use smaller model")
    print("  export RNSR_LAYOUT_DEVICE='cpu'  # Force CPU inference")
    print("  export HF_HOME='/path/to/cache'  # Custom cache directory")


if __name__ == "__main__":
    print("=" * 60)
    print("RNSR LayoutLM Visual Analysis Test Suite")
    print("=" * 60)
    
    # Test 1: Check dependencies and configuration
    available = test_layout_model_info()
    
    # Test 2: Environment configuration
    test_environment_config()
    
    # Test 3: Layout complexity detection
    test_complexity_detection()
    
    # Test 4: Model loading (only if available)
    if available:
        print("\nNote: Model loading test will download ~1.2GB on first run")
        user_input = input("Load LayoutLM model now? (y/n): ")
        if user_input.lower() == 'y':
            test_model_loading()
    
    print("\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)
