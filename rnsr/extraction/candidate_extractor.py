"""
RNSR Candidate Extractor

Pre-extraction of entity candidates using regex and pattern matching.
This provides GROUNDED candidates that the LLM then classifies,
rather than asking the LLM to hallucinate entities from scratch.

The flow is:
1. Extract candidates using regex/patterns (grounded in actual text)
2. LLM classifies and validates candidates (not inventing, labeling)
3. Merge and deduplicate

This approach prevents hallucination because:
- Every entity is tied to an exact text span in the document
- LLM's job is classification, not generation
- Candidates come from deterministic pattern matching
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EntityCandidate:
    """
    A candidate entity extracted from text before LLM classification.
    
    This is GROUNDED - it points to exact text in the document.
    """
    
    text: str                      # Exact text as it appears
    start: int                     # Character offset start
    end: int                       # Character offset end
    candidate_type: str            # Suggested type (from pattern)
    confidence: float = 0.5        # Pattern match confidence
    context: str = ""              # Surrounding text
    pattern_name: str = ""         # Which pattern matched
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Pattern Definitions
# =============================================================================

# Person patterns (names)
PERSON_PATTERNS = [
    # Titles + Names: "Mr. John Smith", "Dr. Jane Doe"
    (r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Hon\.|Rev\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', "title_name"),
    
    # Full names (First Last): "John Smith", "Mary Jane Watson"
    (r'\b[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', "full_name"),
    
    # Names with suffix: "John Smith Jr.", "Robert Johnson III"
    (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Jr\.|Sr\.|II|III|IV|Esq\.)\b', "name_suffix"),
    
    # Role-based: "Plaintiff John Smith", "Defendant ABC Corp"
    (r'\b(?:Plaintiff|Defendant|Petitioner|Respondent|Appellant|Appellee)\s+[A-Z][A-Za-z\s,\.]+?(?=,|\.|;|and\b|\n)', "legal_party"),
]

# Organization patterns
ORGANIZATION_PATTERNS = [
    # Company suffixes: "Acme Inc.", "BigCorp LLC"
    (r'\b[A-Z][A-Za-z\s&]+?(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation|Company|Co\.|L\.P\.|LLP|PLC|GmbH|Pty Ltd)\.?\b', "company_suffix"),
    
    # "The X Company/Organization/Association"
    (r'\bThe\s+[A-Z][A-Za-z\s]+(?:Company|Corporation|Organization|Association|Foundation|Institute|Agency|Department|Board|Commission|Committee)\b', "org_name"),
    
    # Courts: "Supreme Court", "District Court of..."
    (r'\b(?:Supreme|District|Circuit|Appeals?|Bankruptcy|Federal|State|County|Municipal)\s+Court(?:\s+of\s+[A-Za-z\s]+)?\b', "court"),
    
    # Government agencies
    (r'\b(?:Department|Bureau|Office|Agency|Administration)\s+of\s+[A-Z][A-Za-z\s]+\b', "gov_agency"),
]

# Date patterns
DATE_PATTERNS = [
    # Full dates: "January 15, 2024", "15 January 2024"
    (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', "full_date"),
    (r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', "full_date"),
    
    # Numeric dates: "01/15/2024", "2024-01-15"
    (r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', "numeric_date"),
    (r'\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b', "iso_date"),
    
    # Relative dates: "on or about January 2024"
    (r'\b(?:on\s+or\s+about|approximately|around)\s+[A-Z][a-z]+\s+\d{4}\b', "approx_date"),
]

# Monetary patterns
MONETARY_PATTERNS = [
    # Dollar amounts: "$1,234.56", "$1.5 million"
    (r'\$[\d,]+(?:\.\d{2})?\s*(?:million|billion|thousand|M|B|K)?\b', "dollar_amount"),
    
    # Written amounts: "One Million Dollars"
    (r'\b(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+)\s+(?:Hundred|Thousand|Million|Billion)\s+(?:Dollars|dollars|USD)\b', "written_amount"),
    
    # Currency codes: "USD 1,234", "EUR 500"
    (r'\b(?:USD|EUR|GBP|CAD|AUD)\s*[\d,]+(?:\.\d{2})?\b', "currency_code"),
]

# Location patterns  
LOCATION_PATTERNS = [
    # US States
    (r'\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming)\b', "us_state"),
    
    # City, State format
    (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b', "city_state"),
    
    # Addresses
    (r'\b\d+\s+[A-Z][A-Za-z\s]+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Drive|Dr\.|Lane|Ln\.|Way|Place|Pl\.)\b', "street_address"),
]

# Reference patterns (legal citations, exhibits)
REFERENCE_PATTERNS = [
    # Section references: "Section 3.2", "§ 12"
    (r'\b(?:Section|§)\s*\d+(?:\.\d+)*\b', "section_ref"),
    
    # Exhibit references: "Exhibit A", "Attachment 1"
    (r'\b(?:Exhibit|Attachment|Appendix|Schedule|Annex)\s+[A-Z0-9]+\b', "exhibit_ref"),
    
    # Legal citations: "123 F.3d 456"
    (r'\b\d+\s+[A-Z]\.\s*(?:\d+[a-z]*)?\s+\d+\b', "legal_citation"),
    
    # Case citations: "Smith v. Jones"
    (r'\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+\b', "case_citation"),
]

# Document patterns
DOCUMENT_PATTERNS = [
    # Agreement types
    (r'\b(?:the\s+)?[A-Z][A-Za-z\s]*(?:Agreement|Contract|Lease|License|Deed|Will|Trust|Policy|Amendment|Addendum)\b', "agreement_type"),
    
    # Legal documents
    (r'\b(?:Complaint|Motion|Order|Judgment|Verdict|Subpoena|Affidavit|Declaration|Stipulation|Brief)\b', "legal_doc"),
]


# Compile all patterns
COMPILED_PATTERNS: dict[str, list[tuple[re.Pattern, str]]] = {
    "person": [(re.compile(p, re.IGNORECASE), n) for p, n in PERSON_PATTERNS],
    "organization": [(re.compile(p, re.IGNORECASE), n) for p, n in ORGANIZATION_PATTERNS],
    "date": [(re.compile(p, re.IGNORECASE), n) for p, n in DATE_PATTERNS],
    "monetary": [(re.compile(p, re.IGNORECASE), n) for p, n in MONETARY_PATTERNS],
    "location": [(re.compile(p, re.IGNORECASE), n) for p, n in LOCATION_PATTERNS],
    "reference": [(re.compile(p, re.IGNORECASE), n) for p, n in REFERENCE_PATTERNS],
    "document": [(re.compile(p, re.IGNORECASE), n) for p, n in DOCUMENT_PATTERNS],
}


# =============================================================================
# Candidate Extractor
# =============================================================================


class CandidateExtractor:
    """
    Extracts entity candidates from text using regex patterns.
    
    This provides GROUNDED candidates - every candidate points to
    exact text in the document, preventing LLM hallucination.
    """
    
    def __init__(
        self,
        context_window: int = 100,
        min_confidence: float = 0.3,
        dedupe_overlap_threshold: float = 0.5,
    ):
        """
        Initialize the candidate extractor.
        
        Args:
            context_window: Characters of context to capture around matches.
            min_confidence: Minimum confidence to include a candidate.
            dedupe_overlap_threshold: Overlap ratio to consider duplicates.
        """
        self.context_window = context_window
        self.min_confidence = min_confidence
        self.dedupe_overlap_threshold = dedupe_overlap_threshold
    
    def extract_candidates(
        self,
        text: str,
        entity_types: list[str] | None = None,
    ) -> list[EntityCandidate]:
        """
        Extract all entity candidates from text.
        
        Args:
            text: The text to extract from.
            entity_types: Optional list of types to extract (default: all).
            
        Returns:
            List of EntityCandidate objects, sorted by position.
        """
        if not text:
            return []
        
        candidates = []
        types_to_check = entity_types or list(COMPILED_PATTERNS.keys())
        
        for entity_type in types_to_check:
            patterns = COMPILED_PATTERNS.get(entity_type, [])
            
            for pattern, pattern_name in patterns:
                for match in pattern.finditer(text):
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(match, pattern_name)
                    
                    if confidence < self.min_confidence:
                        continue
                    
                    # Extract context
                    start = max(0, match.start() - self.context_window)
                    end = min(len(text), match.end() + self.context_window)
                    context = text[start:end]
                    
                    candidate = EntityCandidate(
                        text=match.group().strip(),
                        start=match.start(),
                        end=match.end(),
                        candidate_type=entity_type,
                        confidence=confidence,
                        context=context,
                        pattern_name=pattern_name,
                    )
                    candidates.append(candidate)
        
        # Deduplicate overlapping candidates
        candidates = self._deduplicate(candidates)
        
        # Sort by position
        candidates.sort(key=lambda c: c.start)
        
        logger.debug(
            "candidates_extracted",
            total=len(candidates),
            by_type={t: sum(1 for c in candidates if c.candidate_type == t) 
                     for t in set(c.candidate_type for c in candidates)},
        )
        
        return candidates
    
    def _calculate_confidence(
        self,
        match: re.Match,
        pattern_name: str,
    ) -> float:
        """
        Calculate confidence score for a pattern match.
        
        More specific patterns get higher confidence.
        """
        base_confidence = 0.5
        
        # Boost for specific pattern types
        high_confidence_patterns = {
            "title_name": 0.9,
            "company_suffix": 0.85,
            "full_date": 0.9,
            "dollar_amount": 0.95,
            "legal_citation": 0.9,
            "case_citation": 0.95,
            "iso_date": 0.9,
            "court": 0.85,
            "exhibit_ref": 0.9,
            "section_ref": 0.85,
        }
        
        if pattern_name in high_confidence_patterns:
            return high_confidence_patterns[pattern_name]
        
        # Boost for longer matches (more specific)
        match_length = len(match.group())
        if match_length > 30:
            base_confidence += 0.2
        elif match_length > 15:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _deduplicate(
        self,
        candidates: list[EntityCandidate],
    ) -> list[EntityCandidate]:
        """
        Remove overlapping candidates, keeping the higher confidence one.
        """
        if not candidates:
            return []
        
        # Sort by confidence descending, then by span length descending
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (-c.confidence, -(c.end - c.start)),
        )
        
        kept = []
        for candidate in sorted_candidates:
            # Check if this overlaps with any kept candidate
            overlaps = False
            for kept_candidate in kept:
                overlap = self._calculate_overlap(candidate, kept_candidate)
                if overlap > self.dedupe_overlap_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(candidate)
        
        return kept
    
    def _calculate_overlap(
        self,
        c1: EntityCandidate,
        c2: EntityCandidate,
    ) -> float:
        """Calculate overlap ratio between two candidates."""
        start = max(c1.start, c2.start)
        end = min(c1.end, c2.end)
        
        if start >= end:
            return 0.0
        
        overlap = end - start
        min_length = min(c1.end - c1.start, c2.end - c2.start)
        
        return overlap / min_length if min_length > 0 else 0.0
    
    def extract_by_type(
        self,
        text: str,
        entity_type: str,
    ) -> list[EntityCandidate]:
        """Extract candidates of a specific type."""
        return self.extract_candidates(text, entity_types=[entity_type])


def extract_candidates_from_text(text: str) -> list[EntityCandidate]:
    """
    Convenience function to extract all candidates from text.
    
    Args:
        text: Text to extract from.
        
    Returns:
        List of EntityCandidate objects.
    """
    extractor = CandidateExtractor()
    return extractor.extract_candidates(text)
