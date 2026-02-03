"""
RNSR Relationship Pattern Extractor

Pre-extracts relationship candidates using patterns, similar to entity extraction.
This provides GROUNDED relationship candidates that are validated by LLM/ToT.

Patterns detect:
1. Entity proximity (co-occurrence signals relationships)
2. Explicit relationship markers (verbs, prepositions)
3. Reference patterns (citations, exhibits)
4. Temporal markers (before, after, during)
5. Causal markers (caused, led to, resulted in)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from rnsr.extraction.candidate_extractor import EntityCandidate
from rnsr.extraction.models import Entity, RelationType

logger = structlog.get_logger(__name__)


@dataclass
class RelationshipCandidate:
    """
    A candidate relationship extracted from text before LLM validation.
    
    Grounded in actual text - tied to specific spans and patterns.
    """
    
    source_text: str              # Source entity text
    target_text: str              # Target entity text
    relationship_type: str        # Suggested relationship type
    evidence: str                 # The text that indicates the relationship
    span_start: int               # Start of relationship evidence
    span_end: int                 # End of relationship evidence
    confidence: float = 0.5       # Pattern match confidence
    pattern_name: str = ""        # Which pattern matched
    source_entity_id: str | None = None
    target_entity_id: str | None = None
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Relationship Pattern Definitions
# =============================================================================

# Affiliation patterns: "X of Y", "X at Y", "X, [title] of Y"
AFFILIATION_PATTERNS = [
    # "John Smith, CEO of Acme Corp"
    (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s+(?:CEO|President|Director|Manager|Partner|Attorney|Counsel|Agent|Representative)\s+(?:of|at|for)\s+([A-Z][A-Za-z\s&]+(?:Inc\.|LLC|Corp\.?|Company)?)', "title_of"),
    
    # "employed by", "works for"
    (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:is\s+)?(?:employed|hired|engaged)\s+by\s+([A-Z][A-Za-z\s&]+)', "employed_by"),
    
    # "X, an employee of Y"
    (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s+(?:an?\s+)?(?:employee|officer|director|member)\s+of\s+([A-Z][A-Za-z\s&]+)', "member_of"),
]

# Party-to patterns: parties to agreements, cases
PARTY_TO_PATTERNS = [
    # "X entered into [agreement] with Y"
    (r'([A-Z][A-Za-z\s]+?)\s+(?:entered\s+into|executed|signed)\s+(?:the\s+)?(?:Agreement|Contract|Lease|License)\s+with\s+([A-Z][A-Za-z\s]+)', "entered_into"),
    
    # "between X and Y"
    (r'between\s+([A-Z][A-Za-z\s,]+?)\s+and\s+([A-Z][A-Za-z\s,]+?)(?:,|\.|;)', "between_parties"),
    
    # "X v. Y" (legal case)
    (r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', "versus"),
    
    # "Plaintiff X" / "Defendant Y"
    (r'(?:Plaintiff|Petitioner|Appellant)\s+([A-Z][A-Za-z\s]+?)(?:,|and|;|\.|filed)', "plaintiff"),
    (r'(?:Defendant|Respondent|Appellee)\s+([A-Z][A-Za-z\s]+?)(?:,|and|;|\.)', "defendant"),
]

# Temporal patterns: before, after, during
TEMPORAL_PATTERNS = [
    # "X before Y"
    (r'([A-Z][A-Za-z\s]+?|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\s+(?:before|prior\s+to|preceding)\s+([A-Z][A-Za-z\s]+?|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})', "temporal_before"),
    
    # "X after Y"
    (r'([A-Z][A-Za-z\s]+?|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\s+(?:after|following|subsequent\s+to)\s+([A-Z][A-Za-z\s]+?)', "temporal_after"),
    
    # "from X to Y"
    (r'from\s+([A-Z][A-Za-z\s,\d]+?)\s+(?:to|until|through)\s+([A-Z][A-Za-z\s,\d]+?)(?:,|\.)', "temporal_range"),
]

# Causal patterns: caused, led to, resulted in
CAUSAL_PATTERNS = [
    # "X caused Y"
    (r'([A-Z][A-Za-z\s]+?)\s+(?:caused|led\s+to|resulted\s+in|gave\s+rise\s+to)\s+([A-Z][A-Za-z\s]+?)(?:,|\.)', "caused"),
    
    # "X as a result of Y"
    (r'([A-Z][A-Za-z\s]+?)\s+(?:as\s+a\s+result\s+of|due\s+to|because\s+of|arising\s+from)\s+([A-Z][A-Za-z\s]+?)(?:,|\.)', "result_of"),
    
    # "X breach ... damages"
    (r'([A-Z][A-Za-z\s]+?)\s+(?:breach(?:ed)?|violat(?:ed|ion))\s+.{0,100}(damages|injury|harm|loss)', "breach_damages"),
]

# Reference patterns: citations, exhibits
REFERENCE_PATTERNS = [
    # "See Exhibit A"
    (r'(?:See|see|per|Per|As\s+(?:shown|stated|set\s+forth)\s+in)\s+(Exhibit\s+[A-Z0-9]+)', "see_exhibit"),
    
    # "pursuant to Section 3.2"
    (r'(?:pursuant\s+to|under|per|in\s+accordance\s+with)\s+(Section\s+[\d\.]+|Article\s+[IVX\d]+)', "pursuant_to"),
    
    # Legal citations "123 F.3d 456"
    (r'([A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+),?\s+(\d+\s+[A-Z]\.\s*\d*[a-z]*\s+\d+)', "case_citation"),
]

# Support/Contradict patterns
SUPPORT_CONTRADICT_PATTERNS = [
    # "consistent with", "in accordance with"
    (r'(?:consistent\s+with|in\s+accordance\s+with|supports|confirms)\s+([A-Z][A-Za-z\s]+?)(?:,|\.)', "supports"),
    
    # "contrary to", "inconsistent with"
    (r'(?:contrary\s+to|inconsistent\s+with|contradicts|conflicts\s+with)\s+([A-Z][A-Za-z\s]+?)(?:,|\.)', "contradicts"),
]

# Supersedes/Amends patterns
AMENDMENT_PATTERNS = [
    # "supersedes"
    (r'([A-Z][A-Za-z\s]+?)\s+(?:supersedes|replaces|terminates)\s+([A-Z][A-Za-z\s]+?)(?:,|\.)', "supersedes"),
    
    # "amends"
    (r'([A-Z][A-Za-z\s]+?)\s+(?:amends|modifies|supplements)\s+([A-Z][A-Za-z\s]+?)(?:,|\.)', "amends"),
]


# Compile all patterns with relationship type mapping
COMPILED_RELATIONSHIP_PATTERNS: dict[str, list[tuple[re.Pattern, str, str]]] = {
    "AFFILIATED_WITH": [
        (re.compile(p, re.IGNORECASE), n, "AFFILIATED_WITH") 
        for p, n in AFFILIATION_PATTERNS
    ],
    "PARTY_TO": [
        (re.compile(p, re.IGNORECASE), n, "PARTY_TO") 
        for p, n in PARTY_TO_PATTERNS
    ],
    "TEMPORAL": [
        (re.compile(p, re.IGNORECASE), n, 
         "TEMPORAL_BEFORE" if "before" in n else "TEMPORAL_AFTER")
        for p, n in TEMPORAL_PATTERNS
    ],
    "CAUSAL": [
        (re.compile(p, re.IGNORECASE), n, "CAUSAL") 
        for p, n in CAUSAL_PATTERNS
    ],
    "REFERENCES": [
        (re.compile(p, re.IGNORECASE), n, "REFERENCES") 
        for p, n in REFERENCE_PATTERNS
    ],
    "SUPPORT_CONTRADICT": [
        (re.compile(p, re.IGNORECASE), n, 
         "SUPPORTS" if "support" in n else "CONTRADICTS")
        for p, n in SUPPORT_CONTRADICT_PATTERNS
    ],
    "AMENDMENT": [
        (re.compile(p, re.IGNORECASE), n, 
         "SUPERSEDES" if "supersedes" in n else "AMENDS")
        for p, n in AMENDMENT_PATTERNS
    ],
}


# =============================================================================
# Relationship Pattern Extractor
# =============================================================================


class RelationshipPatternExtractor:
    """
    Extracts relationship candidates from text using patterns.
    
    Provides GROUNDED candidates - every relationship is tied to
    actual text evidence, preventing hallucination.
    """
    
    def __init__(
        self,
        context_window: int = 150,
        min_confidence: float = 0.4,
    ):
        """
        Initialize the relationship pattern extractor.
        
        Args:
            context_window: Characters of context around matches.
            min_confidence: Minimum confidence to include.
        """
        self.context_window = context_window
        self.min_confidence = min_confidence
    
    def extract_candidates(
        self,
        text: str,
        entities: list[Entity] | None = None,
        relationship_types: list[str] | None = None,
    ) -> list[RelationshipCandidate]:
        """
        Extract relationship candidates from text.
        
        Args:
            text: Text to extract from.
            entities: Optional list of known entities for matching.
            relationship_types: Optional filter for relationship types.
            
        Returns:
            List of RelationshipCandidate objects.
        """
        if not text:
            return []
        
        candidates = []
        types_to_check = relationship_types or list(COMPILED_RELATIONSHIP_PATTERNS.keys())
        
        for rel_category in types_to_check:
            patterns = COMPILED_RELATIONSHIP_PATTERNS.get(rel_category, [])
            
            for pattern, pattern_name, rel_type in patterns:
                for match in pattern.finditer(text):
                    candidate = self._create_candidate_from_match(
                        match=match,
                        pattern_name=pattern_name,
                        relationship_type=rel_type,
                        text=text,
                        entities=entities,
                    )
                    
                    if candidate and candidate.confidence >= self.min_confidence:
                        candidates.append(candidate)
        
        # Also extract co-occurrence relationships
        if entities:
            cooccurrence_candidates = self._extract_cooccurrence_candidates(
                text=text,
                entities=entities,
            )
            candidates.extend(cooccurrence_candidates)
        
        logger.debug(
            "relationship_candidates_extracted",
            total=len(candidates),
            by_type={t: sum(1 for c in candidates if c.relationship_type == t)
                     for t in set(c.relationship_type for c in candidates)},
        )
        
        return candidates
    
    def _create_candidate_from_match(
        self,
        match: re.Match,
        pattern_name: str,
        relationship_type: str,
        text: str,
        entities: list[Entity] | None,
    ) -> RelationshipCandidate | None:
        """Create a relationship candidate from a regex match."""
        groups = match.groups()
        
        if len(groups) < 1:
            return None
        
        # For single-group patterns (like "supports X"), the target is the match
        if len(groups) == 1:
            source_text = "this_section"  # Will be resolved to node_id
            target_text = groups[0].strip()
        else:
            source_text = groups[0].strip()
            target_text = groups[1].strip() if len(groups) > 1 else ""
        
        if not source_text or not target_text:
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(match, pattern_name)
        
        # Get evidence context
        start = max(0, match.start() - self.context_window)
        end = min(len(text), match.end() + self.context_window)
        evidence = text[start:end]
        
        # Try to match to known entities
        source_entity_id = None
        target_entity_id = None
        
        if entities:
            source_entity_id = self._match_to_entity(source_text, entities)
            target_entity_id = self._match_to_entity(target_text, entities)
        
        return RelationshipCandidate(
            source_text=source_text,
            target_text=target_text,
            relationship_type=relationship_type,
            evidence=match.group(),
            span_start=match.start(),
            span_end=match.end(),
            confidence=confidence,
            pattern_name=pattern_name,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            metadata={
                "full_context": evidence,
                "pattern_groups": groups,
            },
        )
    
    def _calculate_confidence(
        self,
        match: re.Match,
        pattern_name: str,
    ) -> float:
        """Calculate confidence for a pattern match."""
        base_confidence = 0.6
        
        # High confidence patterns
        high_confidence_patterns = {
            "versus": 0.95,      # X v. Y is very explicit
            "see_exhibit": 0.9,
            "case_citation": 0.9,
            "entered_into": 0.85,
            "caused": 0.8,
            "supersedes": 0.85,
        }
        
        if pattern_name in high_confidence_patterns:
            return high_confidence_patterns[pattern_name]
        
        # Boost for longer, more specific matches
        match_length = len(match.group())
        if match_length > 50:
            base_confidence += 0.15
        elif match_length > 25:
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    def _match_to_entity(
        self,
        text: str,
        entities: list[Entity],
    ) -> str | None:
        """Try to match extracted text to a known entity."""
        text_lower = text.lower().strip()
        
        for entity in entities:
            # Check canonical name
            if entity.canonical_name.lower() == text_lower:
                return entity.id
            
            # Check aliases
            for alias in entity.aliases:
                if alias.lower() == text_lower:
                    return entity.id
            
            # Fuzzy match (one contains the other)
            if text_lower in entity.canonical_name.lower() or \
               entity.canonical_name.lower() in text_lower:
                return entity.id
        
        return None
    
    def _extract_cooccurrence_candidates(
        self,
        text: str,
        entities: list[Entity],
        window_size: int = 100,
    ) -> list[RelationshipCandidate]:
        """
        Extract relationship candidates based on entity co-occurrence.
        
        Entities mentioned close together often have relationships.
        """
        candidates = []
        
        # Find all entity mentions in text
        entity_positions = []
        for entity in entities:
            # Search for canonical name
            for match in re.finditer(re.escape(entity.canonical_name), text, re.IGNORECASE):
                entity_positions.append({
                    "entity": entity,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                })
            
            # Search for aliases
            for alias in entity.aliases:
                for match in re.finditer(re.escape(alias), text, re.IGNORECASE):
                    entity_positions.append({
                        "entity": entity,
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                    })
        
        # Sort by position
        entity_positions.sort(key=lambda x: x["start"])
        
        # Find co-occurring pairs within window
        for i, pos1 in enumerate(entity_positions):
            for pos2 in entity_positions[i+1:]:
                # Skip if same entity
                if pos1["entity"].id == pos2["entity"].id:
                    continue
                
                # Check if within window
                distance = pos2["start"] - pos1["end"]
                if distance > window_size:
                    break  # Too far, no need to check further
                
                if distance < 0:
                    continue  # Overlapping, skip
                
                # Create co-occurrence candidate
                evidence_start = pos1["start"]
                evidence_end = pos2["end"]
                evidence = text[evidence_start:evidence_end]
                
                # Determine relationship type based on entity types
                rel_type = self._infer_cooccurrence_type(
                    pos1["entity"], pos2["entity"], evidence
                )
                
                # Lower confidence for co-occurrence (needs validation)
                confidence = 0.4 + (1 - distance / window_size) * 0.2
                
                candidates.append(RelationshipCandidate(
                    source_text=pos1["text"],
                    target_text=pos2["text"],
                    relationship_type=rel_type,
                    evidence=evidence,
                    span_start=evidence_start,
                    span_end=evidence_end,
                    confidence=confidence,
                    pattern_name="co_occurrence",
                    source_entity_id=pos1["entity"].id,
                    target_entity_id=pos2["entity"].id,
                    metadata={
                        "distance": distance,
                        "source_type": pos1["entity"].type.value,
                        "target_type": pos2["entity"].type.value,
                    },
                ))
        
        return candidates
    
    def _infer_cooccurrence_type(
        self,
        entity1: Entity,
        entity2: Entity,
        evidence: str,
    ) -> str:
        """Infer relationship type from co-occurring entities."""
        from rnsr.extraction.models import EntityType
        
        type1 = entity1.type
        type2 = entity2.type
        evidence_lower = evidence.lower()
        
        # Person + Organization → likely AFFILIATED_WITH
        if (type1 == EntityType.PERSON and type2 == EntityType.ORGANIZATION) or \
           (type1 == EntityType.ORGANIZATION and type2 == EntityType.PERSON):
            return "AFFILIATED_WITH"
        
        # Date + Event → likely TEMPORAL
        if type1 == EntityType.DATE or type2 == EntityType.DATE:
            return "TEMPORAL_BEFORE"  # Will be refined by validator
        
        # Event + Event → could be CAUSAL
        if type1 == EntityType.EVENT and type2 == EntityType.EVENT:
            if any(word in evidence_lower for word in ["caused", "led", "resulted"]):
                return "CAUSAL"
            return "TEMPORAL_BEFORE"
        
        # Reference patterns
        if type1 == EntityType.REFERENCE or type2 == EntityType.REFERENCE:
            return "REFERENCES"
        
        # Document + anything → likely MENTIONS
        if type1 == EntityType.DOCUMENT or type2 == EntityType.DOCUMENT:
            return "MENTIONS"
        
        # Default
        return "MENTIONS"


def extract_relationship_candidates(
    text: str,
    entities: list[Entity] | None = None,
) -> list[RelationshipCandidate]:
    """
    Convenience function to extract relationship candidates.
    
    Args:
        text: Text to extract from.
        entities: Optional known entities for matching.
        
    Returns:
        List of RelationshipCandidate objects.
    """
    extractor = RelationshipPatternExtractor()
    return extractor.extract_candidates(text, entities)
