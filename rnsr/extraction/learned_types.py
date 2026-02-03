"""
RNSR Learned Entity Types

Adaptive learning system for entity types. When the LLM discovers entity types
that don't match the predefined EntityType enum, they are stored in a flat file.
Over time, this builds a domain-specific vocabulary of entity types.

The learned types are:
1. Stored in a JSON file (configurable location)
2. Loaded at startup and used in extraction prompts
3. Updated with frequency counts when new types are discovered
4. Can be promoted to "suggested" types for the LLM

Usage:
    from rnsr.extraction.learned_types import LearnedTypeRegistry
    
    registry = LearnedTypeRegistry()
    
    # Record a new type
    registry.record_type("witness", context="John Doe, the witness, testified...")
    
    # Get learned types for prompts
    learned = registry.get_learned_types(min_count=3)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default locations for the learned types files
DEFAULT_LEARNED_TYPES_PATH = Path.home() / ".rnsr" / "learned_entity_types.json"
DEFAULT_LEARNED_RELATIONSHIP_TYPES_PATH = Path.home() / ".rnsr" / "learned_relationship_types.json"


class LearnedTypeRegistry:
    """
    Registry for learning and storing custom entity types discovered during extraction.
    
    The registry maintains:
    - Type name and frequency count
    - Example contexts where the type was found
    - First/last seen timestamps
    - Optional mapping to existing EntityType (for future promotion)
    
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
        max_examples_per_type: int = 5,
    ):
        """
        Initialize the learned type registry.
        
        Args:
            storage_path: Path to the JSON file for persistence.
                         Defaults to ~/.rnsr/learned_entity_types.json
            auto_save: Whether to save after each new type is recorded.
            max_examples_per_type: Maximum example contexts to store per type.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_LEARNED_TYPES_PATH
        self.auto_save = auto_save
        self.max_examples_per_type = max_examples_per_type
        
        self._lock = Lock()
        self._types: dict[str, dict[str, Any]] = {}
        self._dirty = False
        
        # Load existing types
        self._load()
    
    def _load(self) -> None:
        """Load learned types from storage."""
        if not self.storage_path.exists():
            logger.debug("no_learned_types_file", path=str(self.storage_path))
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._types = data.get("types", {})
            
            logger.info(
                "learned_types_loaded",
                path=str(self.storage_path),
                count=len(self._types),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_learned_types", error=str(e))
    
    def _save(self) -> None:
        """Save learned types to storage."""
        if not self._dirty:
            return
        
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "types": self._types,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
            logger.debug(
                "learned_types_saved",
                path=str(self.storage_path),
                count=len(self._types),
            )
            
        except Exception as e:
            logger.warning("failed_to_save_learned_types", error=str(e))
    
    def record_type(
        self,
        type_name: str,
        context: str = "",
        entity_name: str = "",
    ) -> None:
        """
        Record a discovered entity type.
        
        Args:
            type_name: The entity type name (e.g., "witness", "clause").
            context: Example context where this type was found.
            entity_name: Name of the entity with this type.
        """
        type_name = type_name.lower().strip()
        
        if not type_name:
            return
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if type_name not in self._types:
                # New type
                self._types[type_name] = {
                    "count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "examples": [],
                    "suggested_mapping": None,
                }
                
                logger.info("new_entity_type_discovered", type=type_name)
            
            # Update existing type
            type_data = self._types[type_name]
            type_data["count"] += 1
            type_data["last_seen"] = now
            
            # Add example if we have context
            if context and len(type_data["examples"]) < self.max_examples_per_type:
                example = {
                    "entity": entity_name,
                    "context": context[:200],  # Truncate long contexts
                    "timestamp": now,
                }
                type_data["examples"].append(example)
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def get_type(self, type_name: str) -> dict[str, Any] | None:
        """
        Get information about a learned type.
        
        Args:
            type_name: The type name to look up.
            
        Returns:
            Type data dict or None if not found.
        """
        return self._types.get(type_name.lower().strip())
    
    def get_learned_types(
        self,
        min_count: int = 1,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get learned types sorted by frequency.
        
        Args:
            min_count: Minimum occurrence count to include.
            limit: Maximum number of types to return.
            
        Returns:
            List of type dicts with name and count.
        """
        with self._lock:
            filtered = [
                {"name": name, **data}
                for name, data in self._types.items()
                if data["count"] >= min_count
            ]
        
        # Sort by count descending
        filtered.sort(key=lambda x: -x["count"])
        
        return filtered[:limit]
    
    def get_types_for_prompt(
        self,
        min_count: int = 2,
        limit: int = 20,
    ) -> list[str]:
        """
        Get type names suitable for including in extraction prompts.
        
        Only returns types that have been seen multiple times,
        indicating they are likely relevant for this workload.
        
        Args:
            min_count: Minimum occurrences to be considered "learned".
            limit: Maximum types to include.
            
        Returns:
            List of type name strings.
        """
        learned = self.get_learned_types(min_count=min_count, limit=limit)
        return [t["name"] for t in learned]
    
    def suggest_mapping(
        self,
        type_name: str,
        map_to: str,
    ) -> None:
        """
        Suggest a mapping from a learned type to a standard EntityType.
        
        This allows users to map frequently occurring custom types
        to one of the predefined EntityType values.
        
        Args:
            type_name: The learned type name.
            map_to: The EntityType value to map to.
        """
        type_name = type_name.lower().strip()
        
        with self._lock:
            if type_name in self._types:
                self._types[type_name]["suggested_mapping"] = map_to
                self._dirty = True
                
                if self.auto_save:
                    self._save()
    
    def get_mappings(self) -> dict[str, str]:
        """
        Get all suggested type mappings.
        
        Returns:
            Dict mapping learned type names to EntityType values.
        """
        with self._lock:
            return {
                name: data["suggested_mapping"]
                for name, data in self._types.items()
                if data.get("suggested_mapping")
            }
    
    def clear(self) -> None:
        """Clear all learned types."""
        with self._lock:
            self._types.clear()
            self._dirty = True
            self._save()
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about learned types."""
        with self._lock:
            total_types = len(self._types)
            total_occurrences = sum(t["count"] for t in self._types.values())
            
            if self._types:
                most_common = max(self._types.items(), key=lambda x: x[1]["count"])
                most_common_name = most_common[0]
                most_common_count = most_common[1]["count"]
            else:
                most_common_name = None
                most_common_count = 0
        
        return {
            "total_types": total_types,
            "total_occurrences": total_occurrences,
            "most_common_type": most_common_name,
            "most_common_count": most_common_count,
            "storage_path": str(self.storage_path),
        }
    
    def force_save(self) -> None:
        """Force save to disk."""
        self._dirty = True
        self._save()


# Global registry instance (lazily initialized)
_global_registry: LearnedTypeRegistry | None = None


def get_learned_type_registry() -> LearnedTypeRegistry:
    """
    Get the global learned type registry.
    
    Returns:
        The singleton LearnedTypeRegistry instance.
    """
    global _global_registry
    
    if _global_registry is None:
        # Check for custom path in environment
        custom_path = os.getenv("RNSR_LEARNED_TYPES_PATH")
        _global_registry = LearnedTypeRegistry(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_registry


def record_learned_type(
    type_name: str,
    context: str = "",
    entity_name: str = "",
) -> None:
    """
    Convenience function to record a learned type using the global registry.
    
    Args:
        type_name: The entity type name.
        context: Example context.
        entity_name: Entity name.
    """
    registry = get_learned_type_registry()
    registry.record_type(type_name, context, entity_name)


# =============================================================================
# Learned Relationship Types Registry
# =============================================================================


class LearnedRelationshipTypeRegistry:
    """
    Registry for learning and storing custom relationship types discovered during extraction.
    
    Same pattern as LearnedTypeRegistry but for relationships.
    Learns types like "testified_against", "represented_by", "prescribed_by".
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
        max_examples_per_type: int = 5,
    ):
        """
        Initialize the learned relationship type registry.
        
        Args:
            storage_path: Path to the JSON file for persistence.
            auto_save: Whether to save after each new type is recorded.
            max_examples_per_type: Maximum example contexts to store per type.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_LEARNED_RELATIONSHIP_TYPES_PATH
        self.auto_save = auto_save
        self.max_examples_per_type = max_examples_per_type
        
        self._lock = Lock()
        self._types: dict[str, dict[str, Any]] = {}
        self._dirty = False
        
        self._load()
    
    def _load(self) -> None:
        """Load learned types from storage."""
        if not self.storage_path.exists():
            logger.debug("no_learned_relationship_types_file", path=str(self.storage_path))
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._types = data.get("types", {})
            
            logger.info(
                "learned_relationship_types_loaded",
                path=str(self.storage_path),
                count=len(self._types),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_learned_relationship_types", error=str(e))
    
    def _save(self) -> None:
        """Save learned types to storage."""
        if not self._dirty:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "types": self._types,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
            logger.debug(
                "learned_relationship_types_saved",
                path=str(self.storage_path),
                count=len(self._types),
            )
            
        except Exception as e:
            logger.warning("failed_to_save_learned_relationship_types", error=str(e))
    
    def record_type(
        self,
        type_name: str,
        context: str = "",
        relationship_description: str = "",
    ) -> None:
        """
        Record a discovered relationship type.
        
        Args:
            type_name: The relationship type name (e.g., "testified_against").
            context: Example evidence text.
            relationship_description: Description of the relationship (source -> target).
        """
        type_name = type_name.lower().strip()
        
        if not type_name:
            return
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if type_name not in self._types:
                self._types[type_name] = {
                    "count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "examples": [],
                    "suggested_mapping": None,
                }
                
                logger.info("new_relationship_type_discovered", type=type_name)
            
            type_data = self._types[type_name]
            type_data["count"] += 1
            type_data["last_seen"] = now
            
            if context and len(type_data["examples"]) < self.max_examples_per_type:
                example = {
                    "description": relationship_description,
                    "context": context[:200],
                    "timestamp": now,
                }
                type_data["examples"].append(example)
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def get_types_for_prompt(
        self,
        min_count: int = 2,
        limit: int = 20,
    ) -> list[str]:
        """
        Get type names suitable for including in extraction prompts.
        """
        with self._lock:
            filtered = [
                (name, data["count"])
                for name, data in self._types.items()
                if data["count"] >= min_count
            ]
        
        filtered.sort(key=lambda x: -x[1])
        return [name for name, _ in filtered[:limit]]
    
    def get_mappings(self) -> dict[str, str]:
        """Get all suggested type mappings."""
        with self._lock:
            return {
                name: data["suggested_mapping"]
                for name, data in self._types.items()
                if data.get("suggested_mapping")
            }
    
    def suggest_mapping(self, type_name: str, map_to: str) -> None:
        """Suggest a mapping from a learned type to a standard RelationType."""
        type_name = type_name.lower().strip()
        
        with self._lock:
            if type_name in self._types:
                self._types[type_name]["suggested_mapping"] = map_to
                self._dirty = True
                
                if self.auto_save:
                    self._save()
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about learned relationship types."""
        with self._lock:
            total_types = len(self._types)
            total_occurrences = sum(t["count"] for t in self._types.values())
            
            if self._types:
                most_common = max(self._types.items(), key=lambda x: x[1]["count"])
                most_common_name = most_common[0]
                most_common_count = most_common[1]["count"]
            else:
                most_common_name = None
                most_common_count = 0
        
        return {
            "total_types": total_types,
            "total_occurrences": total_occurrences,
            "most_common_type": most_common_name,
            "most_common_count": most_common_count,
            "storage_path": str(self.storage_path),
        }
    
    def force_save(self) -> None:
        """Force save to disk."""
        self._dirty = True
        self._save()


# Global relationship registry instance
_global_relationship_registry: LearnedRelationshipTypeRegistry | None = None


def get_learned_relationship_type_registry() -> LearnedRelationshipTypeRegistry:
    """
    Get the global learned relationship type registry.
    
    Returns:
        The singleton LearnedRelationshipTypeRegistry instance.
    """
    global _global_relationship_registry
    
    if _global_relationship_registry is None:
        custom_path = os.getenv("RNSR_LEARNED_RELATIONSHIP_TYPES_PATH")
        _global_relationship_registry = LearnedRelationshipTypeRegistry(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_relationship_registry


def record_learned_relationship_type(
    type_name: str,
    context: str = "",
    relationship_description: str = "",
) -> None:
    """
    Convenience function to record a learned relationship type.
    
    Args:
        type_name: The relationship type name.
        context: Example evidence.
        relationship_description: Source -> target description.
    """
    registry = get_learned_relationship_type_registry()
    registry.record_type(type_name, context, relationship_description)
