"""
RNSR Query Clarification System

Detects ambiguous queries and generates clarifying questions.
Improves answer quality by resolving ambiguity BEFORE retrieval.

Features:
- Detects ambiguous queries (multiple interpretations, missing context)
- Generates targeted clarifying questions
- Learns which query types need clarification
- Optional: proceeds with "best guess" interpretation

Integration: Call before navigation to refine queries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Query Analysis Prompts
# =============================================================================

AMBIGUITY_DETECTION_PROMPT = """Analyze this query for ambiguity.

QUERY: {query}

DOCUMENT CONTEXT: {doc_context}

Identify if the query is ambiguous. Consider:
1. REFERENCE AMBIGUITY: Does "it", "they", "this" refer to something unclear?
2. SCOPE AMBIGUITY: Is the scope unclear (which section, which document)?
3. TEMPORAL AMBIGUITY: Is the time period unclear (current vs historical)?
4. TERM AMBIGUITY: Are there terms with multiple meanings in this context?
5. COMPARISON AMBIGUITY: Is it unclear what is being compared to what?

Respond in JSON:
{{
    "is_ambiguous": true/false,
    "ambiguity_type": "reference|scope|temporal|term|comparison|none",
    "ambiguity_description": "...",
    "possible_interpretations": ["interpretation 1", "interpretation 2"],
    "confidence": 0.0-1.0,
    "suggested_clarification": "A question to ask the user"
}}"""


CLARIFICATION_GENERATION_PROMPT = """Generate a clarifying question for this ambiguous query.

QUERY: {query}

AMBIGUITY: {ambiguity_description}

POSSIBLE INTERPRETATIONS:
{interpretations}

Generate a SHORT, SPECIFIC question that would resolve this ambiguity.
The question should:
1. Be easy to answer (yes/no or short answer)
2. Distinguish between the possible interpretations
3. Be polite and professional

Respond with ONLY the clarifying question, no other text."""


QUERY_REFINEMENT_PROMPT = """Refine this query based on the user's clarification.

ORIGINAL QUERY: {original_query}

CLARIFYING QUESTION: {clarification_question}

USER'S ANSWER: {user_answer}

Generate a refined, unambiguous query that incorporates the user's clarification.

Respond with ONLY the refined query, no other text."""


# =============================================================================
# Data Models
# =============================================================================


class AmbiguityType(str, Enum):
    """Types of query ambiguity."""
    
    REFERENCE = "reference"      # Unclear pronoun/reference
    SCOPE = "scope"              # Unclear scope
    TEMPORAL = "temporal"        # Unclear time period
    TERM = "term"                # Ambiguous term
    COMPARISON = "comparison"    # Unclear comparison
    ENTITY = "entity"            # Multiple possible entities
    NONE = "none"                # No ambiguity


@dataclass
class AmbiguityAnalysis:
    """Result of ambiguity analysis."""
    
    is_ambiguous: bool = False
    ambiguity_type: AmbiguityType = AmbiguityType.NONE
    description: str = ""
    possible_interpretations: list[str] = field(default_factory=list)
    confidence: float = 0.5
    suggested_clarification: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_type": self.ambiguity_type.value,
            "description": self.description,
            "possible_interpretations": self.possible_interpretations,
            "confidence": self.confidence,
            "suggested_clarification": self.suggested_clarification,
        }


@dataclass
class ClarificationRequest:
    """A request for clarification from the user."""
    
    id: str = ""
    original_query: str = ""
    question: str = ""
    options: list[str] = field(default_factory=list)  # Optional multiple choice
    ambiguity_analysis: AmbiguityAnalysis | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "original_query": self.original_query,
            "question": self.question,
            "options": self.options,
            "ambiguity_analysis": self.ambiguity_analysis.to_dict() if self.ambiguity_analysis else None,
        }


@dataclass
class ClarificationResult:
    """Result of the clarification process."""
    
    original_query: str = ""
    refined_query: str = ""
    clarifications_asked: list[ClarificationRequest] = field(default_factory=list)
    user_responses: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    proceeded_with_best_guess: bool = False
    best_guess_interpretation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "refined_query": self.refined_query,
            "clarifications_asked": [c.to_dict() for c in self.clarifications_asked],
            "user_responses": self.user_responses,
            "needs_clarification": self.needs_clarification,
            "proceeded_with_best_guess": self.proceeded_with_best_guess,
            "best_guess_interpretation": self.best_guess_interpretation,
        }


# =============================================================================
# Query Clarifier
# =============================================================================


class QueryClarifier:
    """
    Detects ambiguous queries and manages clarification.
    
    Can be used in two modes:
    1. Interactive: Ask user for clarification
    2. Best-guess: Proceed with most likely interpretation
    """
    
    # Heuristic patterns for ambiguity detection
    AMBIGUOUS_PATTERNS = [
        # Pronoun references without clear antecedent
        (r'\b(it|they|this|that|these|those)\b', AmbiguityType.REFERENCE),
        # Vague scope
        (r'\b(the section|the clause|the document)\b(?! (on|about|titled|named))', AmbiguityType.SCOPE),
        # Temporal ambiguity
        (r'\b(recently|previously|before|after|current)\b(?! \d)', AmbiguityType.TEMPORAL),
        # Comparison without clear target
        (r'\b(compare|versus|vs|differ|difference)\b(?!.*\bto\b)', AmbiguityType.COMPARISON),
    ]
    
    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        use_llm_detection: bool = True,
        ambiguity_threshold: float = 0.6,
        max_clarifications: int = 2,
    ):
        """
        Initialize the query clarifier.
        
        Args:
            llm_fn: LLM function for advanced detection.
            use_llm_detection: Whether to use LLM for detection.
            ambiguity_threshold: Confidence threshold for ambiguity.
            max_clarifications: Maximum clarification rounds.
        """
        self.llm_fn = llm_fn
        self.use_llm_detection = use_llm_detection
        self.ambiguity_threshold = ambiguity_threshold
        self.max_clarifications = max_clarifications
        
        # Track clarification patterns
        self._clarification_stats: dict[str, int] = {}
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self.llm_fn = llm_fn
    
    def analyze_query(
        self,
        query: str,
        doc_context: str = "",
    ) -> AmbiguityAnalysis:
        """
        Analyze a query for ambiguity.
        
        Args:
            query: The query to analyze.
            doc_context: Optional document context.
            
        Returns:
            AmbiguityAnalysis result.
        """
        # First, try heuristic detection
        heuristic_result = self._heuristic_detection(query)
        
        if heuristic_result.is_ambiguous and heuristic_result.confidence >= self.ambiguity_threshold:
            return heuristic_result
        
        # If LLM enabled, do deeper analysis
        if self.use_llm_detection and self.llm_fn:
            return self._llm_detection(query, doc_context)
        
        return heuristic_result
    
    def _heuristic_detection(self, query: str) -> AmbiguityAnalysis:
        """Detect ambiguity using heuristic patterns."""
        query_lower = query.lower()
        
        for pattern, ambiguity_type in self.AMBIGUOUS_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return AmbiguityAnalysis(
                    is_ambiguous=True,
                    ambiguity_type=ambiguity_type,
                    description=f"Detected {ambiguity_type.value} ambiguity: '{match.group()}'",
                    confidence=0.6,
                    suggested_clarification=self._generate_heuristic_clarification(
                        ambiguity_type, match.group()
                    ),
                )
        
        return AmbiguityAnalysis(
            is_ambiguous=False,
            ambiguity_type=AmbiguityType.NONE,
            confidence=0.8,
        )
    
    def _generate_heuristic_clarification(
        self,
        ambiguity_type: AmbiguityType,
        matched_text: str,
    ) -> str:
        """Generate clarification question based on heuristic match."""
        if ambiguity_type == AmbiguityType.REFERENCE:
            return f"What does '{matched_text}' refer to in your question?"
        elif ambiguity_type == AmbiguityType.SCOPE:
            return "Which specific section or part of the document are you asking about?"
        elif ambiguity_type == AmbiguityType.TEMPORAL:
            return "What time period are you asking about (e.g., current, as of a specific date)?"
        elif ambiguity_type == AmbiguityType.COMPARISON:
            return "What would you like me to compare against?"
        else:
            return "Could you please clarify your question?"
    
    def _llm_detection(
        self,
        query: str,
        doc_context: str,
    ) -> AmbiguityAnalysis:
        """Detect ambiguity using LLM."""
        prompt = AMBIGUITY_DETECTION_PROMPT.format(
            query=query,
            doc_context=doc_context[:1000] if doc_context else "No specific document context provided.",
        )
        
        try:
            response = self.llm_fn(prompt)
            return self._parse_ambiguity_response(response)
            
        except Exception as e:
            logger.warning("llm_ambiguity_detection_failed", error=str(e))
            return AmbiguityAnalysis(
                is_ambiguous=False,
                confidence=0.5,
            )
    
    def _parse_ambiguity_response(self, response: str) -> AmbiguityAnalysis:
        """Parse LLM ambiguity detection response."""
        result = AmbiguityAnalysis()
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return result
            
            data = json.loads(json_match.group())
            
            result.is_ambiguous = data.get("is_ambiguous", False)
            result.confidence = data.get("confidence", 0.5)
            result.description = data.get("ambiguity_description", "")
            result.possible_interpretations = data.get("possible_interpretations", [])
            result.suggested_clarification = data.get("suggested_clarification", "")
            
            try:
                result.ambiguity_type = AmbiguityType(data.get("ambiguity_type", "none"))
            except ValueError:
                result.ambiguity_type = AmbiguityType.NONE
            
        except json.JSONDecodeError:
            pass
        
        return result
    
    def generate_clarification(
        self,
        query: str,
        analysis: AmbiguityAnalysis,
    ) -> ClarificationRequest:
        """
        Generate a clarification request.
        
        Args:
            query: The original query.
            analysis: Ambiguity analysis.
            
        Returns:
            ClarificationRequest to present to user.
        """
        # Use LLM if available for better questions
        if self.llm_fn and analysis.possible_interpretations:
            question = self._llm_generate_clarification(query, analysis)
        else:
            question = analysis.suggested_clarification or \
                      "Could you please clarify your question?"
        
        # Generate options if we have interpretations
        options = analysis.possible_interpretations[:4] if analysis.possible_interpretations else []
        
        # Track statistics
        self._clarification_stats[analysis.ambiguity_type.value] = \
            self._clarification_stats.get(analysis.ambiguity_type.value, 0) + 1
        
        return ClarificationRequest(
            id=f"clarify_{hash(query) % 10000:04d}",
            original_query=query,
            question=question,
            options=options,
            ambiguity_analysis=analysis,
        )
    
    def _llm_generate_clarification(
        self,
        query: str,
        analysis: AmbiguityAnalysis,
    ) -> str:
        """Generate clarification question using LLM."""
        interpretations_text = "\n".join([
            f"- {interp}" for interp in analysis.possible_interpretations
        ])
        
        prompt = CLARIFICATION_GENERATION_PROMPT.format(
            query=query,
            ambiguity_description=analysis.description,
            interpretations=interpretations_text,
        )
        
        try:
            response = self.llm_fn(prompt)
            return response.strip()
        except Exception as e:
            logger.warning("clarification_generation_failed", error=str(e))
            return analysis.suggested_clarification or "Could you please clarify?"
    
    def refine_query(
        self,
        original_query: str,
        clarification: ClarificationRequest,
        user_response: str,
    ) -> str:
        """
        Refine query based on user's clarification response.
        
        Args:
            original_query: The original ambiguous query.
            clarification: The clarification that was asked.
            user_response: The user's response.
            
        Returns:
            Refined query.
        """
        if self.llm_fn:
            return self._llm_refine_query(original_query, clarification, user_response)
        else:
            return self._heuristic_refine_query(original_query, clarification, user_response)
    
    def _llm_refine_query(
        self,
        original_query: str,
        clarification: ClarificationRequest,
        user_response: str,
    ) -> str:
        """Refine query using LLM."""
        prompt = QUERY_REFINEMENT_PROMPT.format(
            original_query=original_query,
            clarification_question=clarification.question,
            user_answer=user_response,
        )
        
        try:
            response = self.llm_fn(prompt)
            return response.strip()
        except Exception as e:
            logger.warning("query_refinement_failed", error=str(e))
            return f"{original_query} (regarding: {user_response})"
    
    def _heuristic_refine_query(
        self,
        original_query: str,
        clarification: ClarificationRequest,
        user_response: str,
    ) -> str:
        """Simple heuristic query refinement."""
        # Append the clarification to the query
        return f"{original_query} (specifically: {user_response})"
    
    def get_best_guess(
        self,
        query: str,
        analysis: AmbiguityAnalysis,
    ) -> str:
        """
        Get best-guess interpretation without asking user.
        
        Args:
            query: The ambiguous query.
            analysis: Ambiguity analysis.
            
        Returns:
            Best-guess refined query.
        """
        if analysis.possible_interpretations:
            # Use first interpretation as best guess
            best_guess = analysis.possible_interpretations[0]
            return f"{query} (assuming: {best_guess})"
        else:
            return query
    
    def clarify(
        self,
        query: str,
        doc_context: str = "",
        get_user_response: Callable[[ClarificationRequest], str] | None = None,
        allow_best_guess: bool = True,
    ) -> ClarificationResult:
        """
        Full clarification flow.
        
        Args:
            query: The query to clarify.
            doc_context: Document context.
            get_user_response: Callback to get user response.
            allow_best_guess: Allow proceeding without user response.
            
        Returns:
            ClarificationResult with refined query.
        """
        result = ClarificationResult(original_query=query)
        
        current_query = query
        
        for _ in range(self.max_clarifications):
            # Analyze for ambiguity
            analysis = self.analyze_query(current_query, doc_context)
            
            if not analysis.is_ambiguous or analysis.confidence < self.ambiguity_threshold:
                # Query is clear
                result.refined_query = current_query
                return result
            
            result.needs_clarification = True
            
            # Generate clarification request
            clarification = self.generate_clarification(current_query, analysis)
            result.clarifications_asked.append(clarification)
            
            # Get user response if callback provided
            if get_user_response:
                try:
                    user_response = get_user_response(clarification)
                    result.user_responses.append(user_response)
                    
                    # Refine query
                    current_query = self.refine_query(
                        current_query, clarification, user_response
                    )
                except Exception as e:
                    logger.warning("user_response_failed", error=str(e))
                    if allow_best_guess:
                        result.proceeded_with_best_guess = True
                        result.best_guess_interpretation = self.get_best_guess(
                            current_query, analysis
                        )
                        result.refined_query = result.best_guess_interpretation
                        return result
                    break
            elif allow_best_guess:
                # No callback, use best guess
                result.proceeded_with_best_guess = True
                result.best_guess_interpretation = self.get_best_guess(
                    current_query, analysis
                )
                result.refined_query = result.best_guess_interpretation
                return result
            else:
                # Can't proceed without user
                break
        
        result.refined_query = current_query
        return result
    
    def get_stats(self) -> dict[str, Any]:
        """Get clarification statistics."""
        return dict(self._clarification_stats)


# =============================================================================
# Convenience Functions
# =============================================================================


def needs_clarification(
    query: str,
    doc_context: str = "",
    llm_fn: Callable[[str], str] | None = None,
) -> tuple[bool, AmbiguityAnalysis]:
    """
    Quick check if a query needs clarification.
    
    Args:
        query: The query to check.
        doc_context: Optional document context.
        llm_fn: Optional LLM function.
        
    Returns:
        Tuple of (needs_clarification, analysis).
    """
    clarifier = QueryClarifier(
        llm_fn=llm_fn,
        use_llm_detection=llm_fn is not None,
    )
    
    analysis = clarifier.analyze_query(query, doc_context)
    return analysis.is_ambiguous, analysis


def clarify_query(
    query: str,
    doc_context: str = "",
    llm_fn: Callable[[str], str] | None = None,
    get_user_response: Callable[[ClarificationRequest], str] | None = None,
) -> str:
    """
    Clarify a query and return refined version.
    
    Args:
        query: The query to clarify.
        doc_context: Optional document context.
        llm_fn: Optional LLM function.
        get_user_response: Callback for user interaction.
        
    Returns:
        Refined query.
    """
    clarifier = QueryClarifier(
        llm_fn=llm_fn,
        use_llm_detection=llm_fn is not None,
    )
    
    result = clarifier.clarify(
        query=query,
        doc_context=doc_context,
        get_user_response=get_user_response,
        allow_best_guess=True,
    )
    
    return result.refined_query
