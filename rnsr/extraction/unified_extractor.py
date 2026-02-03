"""
RNSR Unified Grounded Extractor

Extracts BOTH entities AND relationships using the grounded pattern:

1. Pattern extraction (CODE FIRST)
   - Entity candidates from regex patterns
   - Relationship candidates from patterns + co-occurrence
   
2. ToT Validation (LLM SECOND)
   - Entity validation with probabilities
   - Relationship validation with entity context
   
3. Cross-validation
   - Relationships inform entity confidence
   - Entity types inform relationship types

This unified approach prevents hallucination for both entities AND relationships.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import structlog

from rnsr.extraction.models import (
    Entity,
    EntityType,
    ExtractionResult,
    Relationship,
)
from rnsr.extraction.candidate_extractor import CandidateExtractor
from rnsr.extraction.relationship_patterns import RelationshipPatternExtractor
from rnsr.extraction.tot_validator import TotEntityValidator
from rnsr.extraction.relationship_validator import RelationshipValidator
from rnsr.extraction.learned_types import get_learned_type_registry
from rnsr.llm import get_llm

if TYPE_CHECKING:
    from rnsr.models import DocumentTree

logger = structlog.get_logger(__name__)


@dataclass
class UnifiedExtractionResult:
    """Result of unified entity + relationship extraction."""
    
    node_id: str
    doc_id: str
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    
    # Statistics
    entity_candidates: int = 0
    relationship_candidates: int = 0
    validated_entities: int = 0
    validated_relationships: int = 0
    processing_time_ms: float = 0.0
    
    # Metadata
    extraction_method: str = "unified_grounded"
    warnings: list[str] = field(default_factory=list)


class UnifiedGroundedExtractor:
    """
    Unified extractor for entities AND relationships.
    
    Uses the same grounded + ToT pattern for both:
    1. Pattern extraction (grounded candidates)
    2. ToT validation (probabilities + reasoning)
    3. Cross-validation between entities and relationships
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        min_content_length: int = 50,
        entity_selection_threshold: float = 0.6,
        relationship_selection_threshold: float = 0.6,
        enable_cross_validation: bool = True,
        enable_type_learning: bool = True,
    ):
        """
        Initialize the unified extractor.
        
        Args:
            llm: LLM instance.
            min_content_length: Minimum content length to process.
            entity_selection_threshold: ToT threshold for entities.
            relationship_selection_threshold: ToT threshold for relationships.
            enable_cross_validation: Cross-validate entities and relationships.
            enable_type_learning: Learn new entity types.
        """
        self.llm = llm
        self.min_content_length = min_content_length
        self.entity_selection_threshold = entity_selection_threshold
        self.relationship_selection_threshold = relationship_selection_threshold
        self.enable_cross_validation = enable_cross_validation
        self.enable_type_learning = enable_type_learning
        
        # Pattern extractors
        self.entity_candidate_extractor = CandidateExtractor()
        self.relationship_pattern_extractor = RelationshipPatternExtractor()
        
        # ToT validators (lazy init)
        self._entity_validator: TotEntityValidator | None = None
        self._relationship_validator: RelationshipValidator | None = None
        
        # Type learning
        self._type_registry = get_learned_type_registry() if enable_type_learning else None
        
        # LLM
        self._llm_initialized = False
        
        # Cache
        self._cache: dict[str, UnifiedExtractionResult] = {}
    
    def _get_llm(self) -> Any:
        """Get or initialize LLM."""
        if self.llm is None and not self._llm_initialized:
            self.llm = get_llm()
            self._llm_initialized = True
        return self.llm
    
    def _get_entity_validator(self) -> TotEntityValidator:
        """Get or initialize entity validator."""
        if self._entity_validator is None:
            self._entity_validator = TotEntityValidator(
                llm=self._get_llm(),
                selection_threshold=self.entity_selection_threshold,
            )
        return self._entity_validator
    
    def _get_relationship_validator(self) -> RelationshipValidator:
        """Get or initialize relationship validator."""
        if self._relationship_validator is None:
            self._relationship_validator = RelationshipValidator(
                llm=self._get_llm(),
                selection_threshold=self.relationship_selection_threshold,
            )
        return self._relationship_validator
    
    def extract(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        page_num: int | None = None,
        document_tree: "DocumentTree | None" = None,
    ) -> UnifiedExtractionResult:
        """
        Extract entities and relationships using unified grounded approach.
        
        Flow:
        1. Extract entity candidates (patterns)
        2. Extract relationship candidates (patterns + co-occurrence)
        3. Validate entities with ToT
        4. Validate relationships with ToT (using entity context)
        5. Cross-validate (optional)
        
        Args:
            node_id: Section node ID.
            doc_id: Document ID.
            header: Section header.
            content: Section content.
            page_num: Page number.
            document_tree: Optional tree for navigation.
            
        Returns:
            UnifiedExtractionResult with entities and relationships.
        """
        start_time = time.time()
        
        result = UnifiedExtractionResult(
            node_id=node_id,
            doc_id=doc_id,
        )
        
        # Skip short content
        if len(content.strip()) < self.min_content_length:
            return result
        
        # Check cache
        cache_key = f"{doc_id}:{node_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # STEP 1: Extract entity candidates (CODE FIRST)
        entity_candidates = self.entity_candidate_extractor.extract_candidates(content)
        result.entity_candidates = len(entity_candidates)
        
        # STEP 2: Validate entities with ToT
        entities = []
        if entity_candidates:
            entity_validator = self._get_entity_validator()
            entity_validation = entity_validator.validate_candidates(
                candidates=entity_candidates,
                section_header=header,
                section_content=content,
                document_tree=document_tree,
                node_id=node_id,
            )
            entities = entity_validator.candidates_to_entities(
                candidates=entity_candidates,
                validation_result=entity_validation,
                node_id=node_id,
                doc_id=doc_id,
                page_num=page_num,
            )
        
        result.entities = entities
        result.validated_entities = len(entities)
        
        # Learn from OTHER types
        if self._type_registry:
            for entity in entities:
                if entity.type == EntityType.OTHER:
                    self._type_registry.record_type(
                        type_name=entity.metadata.get("original_type", "unknown"),
                        context=entity.mentions[0].context if entity.mentions else "",
                        entity_name=entity.canonical_name,
                    )
        
        # STEP 3: Extract relationship candidates (patterns + co-occurrence)
        relationship_candidates = self.relationship_pattern_extractor.extract_candidates(
            text=content,
            entities=entities,  # Use validated entities for co-occurrence
        )
        result.relationship_candidates = len(relationship_candidates)
        
        # STEP 4: Validate relationships with ToT
        relationships = []
        if relationship_candidates:
            relationship_validator = self._get_relationship_validator()
            relationship_validation = relationship_validator.validate_candidates(
                candidates=relationship_candidates,
                entities=entities,
                section_header=header,
                section_content=content,
            )
            relationships = relationship_validator.candidates_to_relationships(
                candidates=relationship_candidates,
                validation_result=relationship_validation,
                node_id=node_id,
                doc_id=doc_id,
            )
        
        result.relationships = relationships
        result.validated_relationships = len(relationships)
        
        # STEP 5: Cross-validation (optional)
        if self.enable_cross_validation and entities and relationships:
            entities, relationships = self._cross_validate(entities, relationships)
            result.entities = entities
            result.relationships = relationships
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Cache
        self._cache[cache_key] = result
        
        logger.info(
            "unified_extraction_complete",
            node_id=node_id,
            entity_candidates=result.entity_candidates,
            validated_entities=result.validated_entities,
            relationship_candidates=result.relationship_candidates,
            validated_relationships=result.validated_relationships,
            time_ms=result.processing_time_ms,
        )
        
        return result
    
    def _cross_validate(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Cross-validate entities and relationships.
        
        - Entities mentioned in relationships get confidence boost
        - Relationships between validated entities get confidence boost
        """
        entity_ids = {e.id for e in entities}
        
        # Boost entity confidence if mentioned in relationships
        entity_mentions = set()
        for rel in relationships:
            if rel.source_type == "entity":
                entity_mentions.add(rel.source_id)
            if rel.target_type == "entity":
                entity_mentions.add(rel.target_id)
        
        for entity in entities:
            if entity.id in entity_mentions:
                # Boost confidence
                if entity.mentions:
                    entity.mentions[0].confidence = min(
                        entity.mentions[0].confidence * 1.1, 1.0
                    )
                entity.metadata["cross_validated"] = True
        
        # Filter relationships with invalid entity references
        valid_relationships = []
        for rel in relationships:
            source_valid = rel.source_type != "entity" or rel.source_id in entity_ids
            target_valid = rel.target_type != "entity" or rel.target_id in entity_ids
            
            if source_valid and target_valid:
                # Both entities validated - boost confidence
                if rel.source_type == "entity" and rel.target_type == "entity":
                    rel.confidence = min(rel.confidence * 1.1, 1.0)
                    rel.metadata["cross_validated"] = True
                valid_relationships.append(rel)
        
        return entities, valid_relationships
    
    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
    
    def to_extraction_result(
        self,
        unified_result: UnifiedExtractionResult,
    ) -> ExtractionResult:
        """Convert to standard ExtractionResult format."""
        return ExtractionResult(
            node_id=unified_result.node_id,
            doc_id=unified_result.doc_id,
            entities=unified_result.entities,
            relationships=unified_result.relationships,
            processing_time_ms=unified_result.processing_time_ms,
            extraction_method=unified_result.extraction_method,
            warnings=unified_result.warnings,
        )
