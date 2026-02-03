"""
RNSR Reasoning Chain Memory

Long-term memory of successful reasoning chains for similar queries.
Implements "Chain of Thought Distillation" - learning reusable patterns
from successful query-answer interactions.

Features:
- Stores successful query + reasoning chain + answer
- Pattern matching to find similar past queries
- Adapts stored chains to new queries
- Dramatically speeds up common query types

Storage: ~/.rnsr/reasoning_chains.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# Default storage location
DEFAULT_REASONING_MEMORY_PATH = Path.home() / ".rnsr" / "reasoning_chains.json"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    
    action: str  # e.g., "navigate", "read", "synthesize"
    description: str
    node_id: str | None = None
    content_summary: str = ""
    outcome: str = ""
    confidence: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "description": self.description,
            "node_id": self.node_id,
            "content_summary": self.content_summary,
            "outcome": self.outcome,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningStep":
        """Create from dictionary."""
        return cls(
            action=data.get("action", ""),
            description=data.get("description", ""),
            node_id=data.get("node_id"),
            content_summary=data.get("content_summary", ""),
            outcome=data.get("outcome", ""),
            confidence=data.get("confidence", 0.5),
        )


@dataclass
class ReasoningChain:
    """A complete reasoning chain for a query."""
    
    id: str = ""
    
    # The query and answer
    query: str = ""
    query_pattern: str = ""  # Abstracted pattern (e.g., "What is [X] in [DOC]?")
    answer: str = ""
    
    # Reasoning steps
    steps: list[ReasoningStep] = field(default_factory=list)
    
    # Quality metrics
    confidence: float = 0.5
    success: bool = True
    user_feedback: str | None = None  # Optional user feedback
    
    # Usage stats
    created_at: str = ""
    last_used_at: str = ""
    use_count: int = 0
    successful_reuses: int = 0
    
    # Metadata
    doc_type: str = ""  # Type of document (legal, financial, etc.)
    query_type: str = ""  # Type of query (lookup, comparison, etc.)
    entities_involved: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "query_pattern": self.query_pattern,
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "confidence": self.confidence,
            "success": self.success,
            "user_feedback": self.user_feedback,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
            "successful_reuses": self.successful_reuses,
            "doc_type": self.doc_type,
            "query_type": self.query_type,
            "entities_involved": self.entities_involved,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningChain":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            query=data.get("query", ""),
            query_pattern=data.get("query_pattern", ""),
            answer=data.get("answer", ""),
            steps=[ReasoningStep.from_dict(s) for s in data.get("steps", [])],
            confidence=data.get("confidence", 0.5),
            success=data.get("success", True),
            user_feedback=data.get("user_feedback"),
            created_at=data.get("created_at", ""),
            last_used_at=data.get("last_used_at", ""),
            use_count=data.get("use_count", 0),
            successful_reuses=data.get("successful_reuses", 0),
            doc_type=data.get("doc_type", ""),
            query_type=data.get("query_type", ""),
            entities_involved=data.get("entities_involved", []),
        )


@dataclass
class ChainMatch:
    """A matched reasoning chain with similarity score."""
    
    chain: ReasoningChain
    similarity: float
    adaptations_needed: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain.id,
            "similarity": self.similarity,
            "adaptations_needed": self.adaptations_needed,
        }


# =============================================================================
# Reasoning Chain Memory
# =============================================================================


class ReasoningChainMemory:
    """
    Long-term memory for successful reasoning chains.
    
    Key capabilities:
    1. Store successful query-chain-answer triplets
    2. Find similar past queries using pattern matching
    3. Adapt stored chains to new queries
    4. Track success rates for chain reuse
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
        max_chains: int = 1000,
        min_similarity_threshold: float = 0.6,
    ):
        """
        Initialize reasoning chain memory.
        
        Args:
            storage_path: Path to JSON storage file.
            auto_save: Whether to save after changes.
            max_chains: Maximum chains to store.
            min_similarity_threshold: Minimum similarity for matches.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_REASONING_MEMORY_PATH
        self.auto_save = auto_save
        self.max_chains = max_chains
        self.min_similarity_threshold = min_similarity_threshold
        
        self._lock = Lock()
        self._chains: dict[str, ReasoningChain] = {}
        self._pattern_index: dict[str, list[str]] = {}  # pattern -> chain IDs
        self._dirty = False
        
        self._load()
    
    def _load(self) -> None:
        """Load chains from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            for chain_data in data.get("chains", []):
                chain = ReasoningChain.from_dict(chain_data)
                self._chains[chain.id] = chain
                
                # Index by pattern
                if chain.query_pattern:
                    if chain.query_pattern not in self._pattern_index:
                        self._pattern_index[chain.query_pattern] = []
                    self._pattern_index[chain.query_pattern].append(chain.id)
            
            logger.info(
                "reasoning_chains_loaded",
                count=len(self._chains),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_reasoning_chains", error=str(e))
    
    def _save(self) -> None:
        """Save chains to storage."""
        if not self._dirty:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "chains": [c.to_dict() for c in self._chains.values()],
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
        except Exception as e:
            logger.warning("failed_to_save_reasoning_chains", error=str(e))
    
    def store_chain(
        self,
        query: str,
        answer: str,
        steps: list[ReasoningStep],
        confidence: float = 0.5,
        success: bool = True,
        doc_type: str = "",
        query_type: str = "",
        entities_involved: list[str] | None = None,
    ) -> str:
        """
        Store a reasoning chain.
        
        Args:
            query: The original query.
            answer: The final answer.
            steps: List of reasoning steps.
            confidence: Answer confidence.
            success: Whether the answer was successful.
            doc_type: Type of document.
            query_type: Type of query.
            entities_involved: Entity types involved.
            
        Returns:
            Chain ID.
        """
        # Generate ID
        chain_id = f"chain_{hashlib.md5(query.encode()).hexdigest()[:12]}"
        
        # Extract query pattern
        query_pattern = self._extract_pattern(query)
        
        now = datetime.utcnow().isoformat()
        
        chain = ReasoningChain(
            id=chain_id,
            query=query,
            query_pattern=query_pattern,
            answer=answer,
            steps=steps,
            confidence=confidence,
            success=success,
            created_at=now,
            last_used_at=now,
            use_count=1,
            doc_type=doc_type,
            query_type=query_type,
            entities_involved=entities_involved or [],
        )
        
        with self._lock:
            # Evict if at capacity
            if len(self._chains) >= self.max_chains:
                self._evict_oldest()
            
            self._chains[chain_id] = chain
            
            # Index by pattern
            if query_pattern:
                if query_pattern not in self._pattern_index:
                    self._pattern_index[query_pattern] = []
                if chain_id not in self._pattern_index[query_pattern]:
                    self._pattern_index[query_pattern].append(chain_id)
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
        
        logger.info(
            "reasoning_chain_stored",
            chain_id=chain_id,
            pattern=query_pattern[:50] if query_pattern else None,
        )
        
        return chain_id
    
    def _extract_pattern(self, query: str) -> str:
        """
        Extract an abstract pattern from a query.
        
        Replaces specific entities with placeholders.
        E.g., "What is the liability clause in Contract.pdf?" -> "What is [CONCEPT] in [DOC]?"
        """
        pattern = query
        
        # Replace quoted strings
        pattern = re.sub(r'"[^"]*"', '[QUOTED]', pattern)
        
        # Replace file references
        pattern = re.sub(r'\b\w+\.(pdf|doc|docx|txt)\b', '[DOC]', pattern, flags=re.IGNORECASE)
        
        # Replace dates
        pattern = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', pattern)
        pattern = re.sub(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b', '[DATE]', pattern, flags=re.IGNORECASE)
        
        # Replace money amounts
        pattern = re.sub(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion))?', '[AMOUNT]', pattern)
        
        # Replace numbers
        pattern = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', '[NUMBER]', pattern)
        
        # Replace proper nouns (capitalized words not at start of sentence)
        words = pattern.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word.isalpha():
                words[i] = '[ENTITY]'
        pattern = ' '.join(words)
        
        return pattern
    
    def find_similar(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[ChainMatch]:
        """
        Find similar reasoning chains.
        
        Args:
            query: The query to match.
            max_results: Maximum results to return.
            
        Returns:
            List of ChainMatch objects.
        """
        query_pattern = self._extract_pattern(query)
        query_words = set(query.lower().split())
        
        matches = []
        
        with self._lock:
            for chain in self._chains.values():
                # Only consider successful chains
                if not chain.success:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(
                    query, query_pattern, query_words, chain
                )
                
                if similarity >= self.min_similarity_threshold:
                    # Determine adaptations needed
                    adaptations = self._identify_adaptations(query, chain)
                    
                    matches.append(ChainMatch(
                        chain=chain,
                        similarity=similarity,
                        adaptations_needed=adaptations,
                    ))
        
        # Sort by similarity
        matches.sort(key=lambda m: -m.similarity)
        
        return matches[:max_results]
    
    def _calculate_similarity(
        self,
        query: str,
        query_pattern: str,
        query_words: set[str],
        chain: ReasoningChain,
    ) -> float:
        """Calculate similarity between query and stored chain."""
        # Pattern match (highest weight)
        pattern_score = 0.0
        if chain.query_pattern and query_pattern:
            pattern_words_1 = set(query_pattern.lower().split())
            pattern_words_2 = set(chain.query_pattern.lower().split())
            if pattern_words_1 and pattern_words_2:
                intersection = len(pattern_words_1 & pattern_words_2)
                union = len(pattern_words_1 | pattern_words_2)
                pattern_score = intersection / union if union > 0 else 0
        
        # Word overlap
        chain_words = set(chain.query.lower().split())
        word_overlap = len(query_words & chain_words) / max(len(query_words), 1)
        
        # Query type match
        type_bonus = 0.1 if chain.query_type else 0
        
        # Combine scores
        similarity = (pattern_score * 0.5) + (word_overlap * 0.4) + type_bonus
        
        # Boost by success rate
        if chain.use_count > 1:
            success_rate = chain.successful_reuses / chain.use_count
            similarity *= (0.8 + 0.2 * success_rate)
        
        return min(similarity, 1.0)
    
    def _identify_adaptations(
        self,
        query: str,
        chain: ReasoningChain,
    ) -> list[str]:
        """Identify what adaptations are needed to reuse a chain."""
        adaptations = []
        
        # Check for different entities
        query_lower = query.lower()
        chain_lower = chain.query.lower()
        
        # Extract quoted terms
        query_quotes = set(re.findall(r'"([^"]*)"', query))
        chain_quotes = set(re.findall(r'"([^"]*)"', chain.query))
        
        if query_quotes != chain_quotes:
            adaptations.append("Replace entity references")
        
        # Check for different document references
        query_docs = set(re.findall(r'\b\w+\.(pdf|doc|docx|txt)\b', query, re.IGNORECASE))
        chain_docs = set(re.findall(r'\b\w+\.(pdf|doc|docx|txt)\b', chain.query, re.IGNORECASE))
        
        if query_docs != chain_docs:
            adaptations.append("Adjust document references")
        
        # Check for different numbers
        query_nums = set(re.findall(r'\d+', query))
        chain_nums = set(re.findall(r'\d+', chain.query))
        
        if query_nums != chain_nums:
            adaptations.append("Update numerical values")
        
        return adaptations
    
    def record_reuse(
        self,
        chain_id: str,
        success: bool,
    ) -> None:
        """
        Record that a chain was reused.
        
        Args:
            chain_id: ID of the reused chain.
            success: Whether the reuse was successful.
        """
        with self._lock:
            if chain_id in self._chains:
                chain = self._chains[chain_id]
                chain.use_count += 1
                chain.last_used_at = datetime.utcnow().isoformat()
                
                if success:
                    chain.successful_reuses += 1
                
                self._dirty = True
                
                if self.auto_save:
                    self._save()
    
    def add_feedback(
        self,
        chain_id: str,
        feedback: str,
    ) -> None:
        """Add user feedback to a chain."""
        with self._lock:
            if chain_id in self._chains:
                self._chains[chain_id].user_feedback = feedback
                self._dirty = True
                
                if self.auto_save:
                    self._save()
    
    def _evict_oldest(self) -> None:
        """Evict oldest/least used chains."""
        if not self._chains:
            return
        
        # Score chains by recency and usage
        scored = []
        for chain_id, chain in self._chains.items():
            # Lower score = evict first
            score = chain.use_count * 0.5 + chain.successful_reuses
            scored.append((chain_id, score))
        
        scored.sort(key=lambda x: x[1])
        
        # Evict bottom 10%
        evict_count = max(1, len(scored) // 10)
        for i in range(evict_count):
            chain_id = scored[i][0]
            chain = self._chains[chain_id]
            
            # Remove from pattern index
            if chain.query_pattern in self._pattern_index:
                if chain_id in self._pattern_index[chain.query_pattern]:
                    self._pattern_index[chain.query_pattern].remove(chain_id)
            
            del self._chains[chain_id]
        
        logger.debug("chains_evicted", count=evict_count)
    
    def get_chain(self, chain_id: str) -> ReasoningChain | None:
        """Get a chain by ID."""
        return self._chains.get(chain_id)
    
    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            total_uses = sum(c.use_count for c in self._chains.values())
            total_successes = sum(c.successful_reuses for c in self._chains.values())
            
            return {
                "total_chains": len(self._chains),
                "unique_patterns": len(self._pattern_index),
                "total_uses": total_uses,
                "total_successful_reuses": total_successes,
                "reuse_success_rate": total_successes / total_uses if total_uses > 0 else 0,
            }
    
    def get_top_patterns(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most used patterns."""
        pattern_stats = []
        
        with self._lock:
            for pattern, chain_ids in self._pattern_index.items():
                total_uses = sum(
                    self._chains[cid].use_count
                    for cid in chain_ids
                    if cid in self._chains
                )
                pattern_stats.append({
                    "pattern": pattern,
                    "chain_count": len(chain_ids),
                    "total_uses": total_uses,
                })
        
        pattern_stats.sort(key=lambda x: -x["total_uses"])
        return pattern_stats[:limit]


# =============================================================================
# Chain Adapter
# =============================================================================


class ChainAdapter:
    """
    Adapts a stored reasoning chain to a new query.
    
    Takes a similar chain and modifies it to work for
    a different but related query.
    """
    
    def __init__(self, llm_fn: Any | None = None):
        """
        Initialize the chain adapter.
        
        Args:
            llm_fn: LLM function for adaptation.
        """
        self.llm_fn = llm_fn
    
    def adapt_chain(
        self,
        chain: ReasoningChain,
        new_query: str,
        adaptations_needed: list[str],
    ) -> list[ReasoningStep]:
        """
        Adapt a chain's steps to a new query.
        
        Args:
            chain: The source chain.
            new_query: The new query to adapt to.
            adaptations_needed: List of required adaptations.
            
        Returns:
            Adapted reasoning steps.
        """
        adapted_steps = []
        
        for step in chain.steps:
            adapted_step = ReasoningStep(
                action=step.action,
                description=self._adapt_description(
                    step.description,
                    chain.query,
                    new_query,
                ),
                node_id=None,  # Will be determined during execution
                content_summary=step.content_summary,
                outcome="",  # To be filled during execution
                confidence=step.confidence,
            )
            adapted_steps.append(adapted_step)
        
        return adapted_steps
    
    def _adapt_description(
        self,
        description: str,
        old_query: str,
        new_query: str,
    ) -> str:
        """Adapt a step description to the new query context."""
        # Simple substitution-based adaptation
        # Extract entities from old and new queries
        
        old_entities = re.findall(r'"([^"]*)"', old_query)
        new_entities = re.findall(r'"([^"]*)"', new_query)
        
        adapted = description
        
        # Replace old entities with new ones
        for old, new in zip(old_entities, new_entities):
            adapted = adapted.replace(old, new)
        
        return adapted


# =============================================================================
# Global Memory Instance
# =============================================================================

_global_memory: ReasoningChainMemory | None = None


def get_reasoning_memory() -> ReasoningChainMemory:
    """Get the global reasoning chain memory."""
    global _global_memory
    
    if _global_memory is None:
        custom_path = os.getenv("RNSR_REASONING_MEMORY_PATH")
        _global_memory = ReasoningChainMemory(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_memory


def store_reasoning_chain(
    query: str,
    answer: str,
    trace: list[dict],
    confidence: float = 0.5,
    success: bool = True,
) -> str:
    """
    Store a reasoning chain from a navigation trace.
    
    Convenience function for storing chains.
    """
    memory = get_reasoning_memory()
    
    # Convert trace to reasoning steps
    steps = []
    for entry in trace:
        step = ReasoningStep(
            action=entry.get("action", "navigate"),
            description=entry.get("description", ""),
            node_id=entry.get("node_id"),
            content_summary=entry.get("summary", "")[:200],
            outcome=entry.get("outcome", ""),
            confidence=entry.get("confidence", 0.5),
        )
        steps.append(step)
    
    return memory.store_chain(
        query=query,
        answer=answer,
        steps=steps,
        confidence=confidence,
        success=success,
    )


def find_similar_chains(query: str, max_results: int = 3) -> list[ChainMatch]:
    """Find similar reasoning chains for a query."""
    memory = get_reasoning_memory()
    return memory.find_similar(query, max_results)
