"""
RNSR Custom Exceptions

All module-specific exceptions inherit from RNSRError.
"""


class RNSRError(Exception):
    """Base exception for all RNSR errors."""

    pass


# Ingestion Exceptions
class IngestionError(RNSRError):
    """Base exception for ingestion errors."""

    pass


class FontAnalysisError(IngestionError):
    """Raised when font histogram analysis fails."""

    pass


class SegmentationError(IngestionError):
    """Raised when page segmentation fails."""

    pass


class OCRError(IngestionError):
    """Raised when OCR fallback fails."""

    pass


# Indexing Exceptions
class IndexingError(RNSRError):
    """Base exception for indexing errors."""

    pass


class SummaryGenerationError(IndexingError):
    """Raised when LLM summary generation fails."""

    pass


class KVStoreError(IndexingError):
    """Raised when KV store operations fail."""

    pass


# Agent Exceptions
class AgentError(RNSRError):
    """Base exception for agent errors."""

    pass


class VariableNotFoundError(AgentError):
    """Raised when a variable pointer cannot be resolved."""

    pass


class NavigationError(AgentError):
    """Raised when document navigation fails."""

    pass
