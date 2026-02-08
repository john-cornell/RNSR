"""
RNSR Extraction Module

Entity and relationship extraction for ontological document understanding.

## Recommended: RLMUnifiedExtractor

Use the unified RLM extractor for all extraction needs:

```python
from rnsr.extraction import RLMUnifiedExtractor, extract_entities_and_relationships

# Simple API
result = extract_entities_and_relationships(node_id, doc_id, header, content)

# Full control
extractor = RLMUnifiedExtractor()
result = extractor.extract(node_id, doc_id, header, content)
```

This extractor:
1. LLM writes extraction code based on document (adaptive)
2. Code executes on DOC_VAR (grounded in text)
3. ToT validates with probabilities (accurate)
4. Cross-validates entities and relationships (comprehensive)
5. Learns new types from usage (domain-adaptive)

## Adaptive Learning

The system learns from your document workload:
- Entity types: `LearnedTypeRegistry`
- Relationship types: `LearnedRelationshipTypeRegistry`
- Normalization patterns: `LearnedNormalizationPatterns`
- Stop words: `LearnedStopWords`
- Header thresholds: `LearnedHeaderThresholds`
- Query patterns: `LearnedQueryPatterns`

All learned data persists in `~/.rnsr/`.
"""

from rnsr.extraction.models import (
    Entity,
    EntityLink,
    EntityType,
    ExtractionResult,
    Mention,
    Relationship,
    RelationType,
)

# Primary extractor (recommended)
from rnsr.extraction.rlm_unified_extractor import (
    RLMUnifiedExtractor,
    RLMUnifiedResult,
    extract_entities_and_relationships,
)

# Legacy utility functions (classes removed — use RLMUnifiedExtractor instead)
from rnsr.extraction.entity_extractor import merge_entities
from rnsr.extraction.grounded_extractor import (
    GroundedEntityExtractor,
    ValidationMode,
)
from rnsr.extraction.unified_extractor import (
    UnifiedGroundedExtractor,
    UnifiedExtractionResult,
)
from rnsr.extraction.rlm_extractor import (
    RLMEntityExtractor,
    RLMExtractionResult,
    LightweightREPL,
)
from rnsr.extraction.tot_validator import (
    TotEntityValidator,
    TotBatchResult,
    TotValidationResult,
)
from rnsr.extraction.relationship_validator import (
    RelationshipValidator,
    RelationshipValidationResult,
    RelationshipBatchResult,
)
from rnsr.extraction.candidate_extractor import (
    CandidateExtractor,
    EntityCandidate,
    extract_candidates_from_text,
)
from rnsr.extraction.relationship_patterns import (
    RelationshipPatternExtractor,
    RelationshipCandidate,
    extract_relationship_candidates,
)
from rnsr.extraction.relationship_extractor import extract_implicit_relationships
from rnsr.extraction.entity_linker import (
    EntityLinker,
    LearnedNormalizationPatterns,
    get_learned_normalization_patterns,
)
from rnsr.extraction.learned_types import (
    LearnedTypeRegistry,
    LearnedRelationshipTypeRegistry,
    get_learned_type_registry,
    get_learned_relationship_type_registry,
    record_learned_type,
    record_learned_relationship_type,
)

__all__ = [
    # Models
    "Entity",
    "EntityLink",
    "EntityType",
    "ExtractionResult",
    "Mention",
    "Relationship",
    "RelationType",
    
    # PRIMARY EXTRACTOR (recommended)
    "RLMUnifiedExtractor",
    "RLMUnifiedResult",
    "extract_entities_and_relationships",  # Simple function API
    
    # Alternative extractors
    "UnifiedGroundedExtractor",
    "UnifiedExtractionResult",
    "RLMEntityExtractor",
    "RLMExtractionResult",
    "GroundedEntityExtractor",
    "ValidationMode",
    
    # Supporting components
    "CandidateExtractor",
    "RelationshipPatternExtractor",
    "EntityLinker",
    "TotEntityValidator",
    "TotBatchResult",
    "TotValidationResult",
    "RelationshipValidator",
    "RelationshipValidationResult",
    "RelationshipBatchResult",
    "LightweightREPL",
    
    # Data classes
    "EntityCandidate",
    "RelationshipCandidate",
    
    # Adaptive Learning Registries
    "LearnedTypeRegistry",
    "LearnedRelationshipTypeRegistry",
    "LearnedNormalizationPatterns",
    "get_learned_type_registry",
    "get_learned_relationship_type_registry",
    "get_learned_normalization_patterns",
    "record_learned_type",
    "record_learned_relationship_type",
    
    # Utility functions
    "merge_entities",
    "extract_implicit_relationships",
    "extract_candidates_from_text",
    "extract_relationship_candidates",
]
