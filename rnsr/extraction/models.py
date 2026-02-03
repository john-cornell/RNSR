"""
RNSR Extraction Data Models

Pydantic models for entity extraction, relationships, and ontological linking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Entity Types
# =============================================================================


class EntityType(str, Enum):
    """
    Types of entities that can be extracted from documents.
    
    Note: The OTHER type is used as a fallback for any entity types
    the LLM identifies that don't match a predefined category.
    The original type string is preserved in entity.metadata["original_type"].
    """
    
    PERSON = "person"           # Names, roles, parties (plaintiff, defendant, witness)
    ORGANIZATION = "organization"  # Companies, agencies, courts
    LEGAL_CONCEPT = "legal_concept"  # Claims, breaches, obligations, remedies
    DATE = "date"               # Key dates, events, deadlines
    EVENT = "event"             # Significant occurrences
    LOCATION = "location"       # Places, addresses, jurisdictions
    REFERENCE = "reference"     # Section references, document citations
    MONETARY = "monetary"       # Dollar amounts, financial figures
    DOCUMENT = "document"       # Referenced documents (exhibits, contracts)
    
    # Fallback for any type not in the predefined list
    OTHER = "other"             # Catch-all for novel/custom entity types


class RelationType(str, Enum):
    """
    Types of relationships between entities and sections.
    
    Note: The OTHER type is used as a fallback for any relationship types
    the LLM identifies that don't match a predefined category.
    The original type string is preserved in relationship.metadata["original_type"].
    """
    
    # Entity-to-Section relationships
    MENTIONS = "mentions"           # Entity X is mentioned in Section Y
    DEFINED_IN = "defined_in"       # Entity X is defined/introduced in Section Y
    
    # Entity-to-Entity relationships
    TEMPORAL_BEFORE = "temporal_before"  # Event X occurred before Event Y
    TEMPORAL_AFTER = "temporal_after"    # Event X occurred after Event Y
    CAUSAL = "causal"                    # Action X caused/led to Outcome Y
    AFFILIATED_WITH = "affiliated_with"  # Person X is affiliated with Org Y
    PARTY_TO = "party_to"                # Entity X is party to Document/Event Y
    
    # Section-to-Section relationships
    SUPPORTS = "supports"           # Section X supports claim in Section Y
    CONTRADICTS = "contradicts"     # Section X contradicts Section Y
    REFERENCES = "references"       # Section X references Document/Section Y
    SUPERSEDES = "supersedes"       # Section X supersedes/overrides Section Y
    AMENDS = "amends"              # Section X amends Section Y
    
    # Fallback for any relationship type not in the predefined list
    OTHER = "other"                # Catch-all for novel/custom relationship types


# =============================================================================
# Mention Model
# =============================================================================


class Mention(BaseModel):
    """A specific occurrence of an entity in a document section."""
    
    id: str = Field(default_factory=lambda: f"mention_{str(uuid4())[:8]}")
    node_id: str                    # Which skeleton node contains this mention
    doc_id: str                     # Which document (for multi-doc bundles)
    span_start: int | None = None   # Character offset start (optional)
    span_end: int | None = None     # Character offset end (optional)
    context: str = ""               # Surrounding text snippet for grounding
    page_num: int | None = None     # Page number if available
    confidence: float = 1.0         # Extraction confidence (0.0-1.0)
    
    class Config:
        frozen = False


# =============================================================================
# Entity Model
# =============================================================================


class Entity(BaseModel):
    """
    An extracted entity from a document.
    
    Entities represent named concepts (people, organizations, dates, etc.)
    that can be tracked across document sections and linked across documents.
    """
    
    id: str = Field(default_factory=lambda: f"ent_{str(uuid4())[:8]}")
    type: EntityType
    canonical_name: str             # Normalized/canonical name
    aliases: list[str] = Field(default_factory=list)  # Alternative names/spellings
    mentions: list[Mention] = Field(default_factory=list)  # Where this entity appears
    metadata: dict[str, Any] = Field(default_factory=dict)  # Type-specific metadata
    
    # Tracking
    source_doc_id: str | None = None  # Original document where first extracted
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        frozen = False
    
    def add_mention(self, mention: Mention) -> None:
        """Add a new mention of this entity."""
        self.mentions.append(mention)
    
    def add_alias(self, alias: str) -> None:
        """Add an alternative name for this entity."""
        normalized = alias.strip()
        if normalized and normalized not in self.aliases and normalized != self.canonical_name:
            self.aliases.append(normalized)
    
    @property
    def all_names(self) -> list[str]:
        """Get all names (canonical + aliases)."""
        return [self.canonical_name] + self.aliases
    
    @property
    def document_ids(self) -> set[str]:
        """Get all document IDs where this entity appears."""
        return {m.doc_id for m in self.mentions}
    
    @property
    def node_ids(self) -> set[str]:
        """Get all node IDs where this entity appears."""
        return {m.node_id for m in self.mentions}


# =============================================================================
# Relationship Model
# =============================================================================


class Relationship(BaseModel):
    """
    A relationship between entities or between sections.
    
    Relationships capture semantic connections that enable cross-document
    understanding and complex query resolution.
    """
    
    id: str = Field(default_factory=lambda: f"rel_{str(uuid4())[:8]}")
    type: RelationType
    source_id: str              # Entity ID or Node ID
    target_id: str              # Entity ID or Node ID
    source_type: str = "entity"  # "entity" or "node"
    target_type: str = "entity"  # "entity" or "node"
    
    doc_id: str | None = None   # Source document
    confidence: float = 1.0     # Extraction confidence (0.0-1.0)
    evidence: str = ""          # Supporting text that establishes the relationship
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        frozen = False


# =============================================================================
# Entity Link Model (for cross-document linking)
# =============================================================================


class EntityLink(BaseModel):
    """
    A link between two entities that represent the same real-world entity.
    
    Used for cross-document entity resolution.
    """
    
    entity_id_1: str            # First entity ID
    entity_id_2: str            # Second entity ID
    confidence: float = 1.0     # Link confidence (0.0-1.0)
    link_method: str = "exact"  # How the link was established (exact, fuzzy, llm)
    evidence: str = ""          # Why these entities are considered the same
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        frozen = False
    
    @property
    def entity_ids(self) -> tuple[str, str]:
        """Get both entity IDs as a sorted tuple for consistent ordering."""
        return tuple(sorted([self.entity_id_1, self.entity_id_2]))


# =============================================================================
# Extraction Result Model
# =============================================================================


class ExtractionResult(BaseModel):
    """
    Result of entity/relationship extraction from a document section.
    """
    
    node_id: str                # Which node was processed
    doc_id: str                 # Which document
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    
    # Processing metadata
    extraction_method: str = "llm"  # Method used (llm, rule-based, hybrid)
    processing_time_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    
    class Config:
        frozen = False
