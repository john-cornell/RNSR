"""
RNSR Unified RLM Extractor

The single, comprehensive extractor for BOTH entities AND relationships.
Always uses the most accurate approach:

1. LLM analyzes document and writes extraction code
2. Code executes on DOC_VAR (grounded in actual text)
3. ToT validation with probabilities
4. Cross-validation between entities and relationships
5. Adaptive learning for new types

This is the RECOMMENDED extractor - it consolidates all the best
practices from the RLM paper into a single, unified interface.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import structlog

from rnsr.extraction.models import (
    Entity,
    EntityType,
    ExtractionResult,
    Mention,
    Relationship,
    RelationType,
)
from rnsr.extraction.learned_types import (
    get_learned_type_registry,
    get_learned_relationship_type_registry,
)
from rnsr.llm import get_llm

if TYPE_CHECKING:
    from rnsr.agent.repl_env import REPLEnvironment
    from rnsr.models import DocumentTree

logger = structlog.get_logger(__name__)


# =============================================================================
# Unified RLM Prompts
# =============================================================================

RLM_UNIFIED_SYSTEM_PROMPT = """You are an RLM (Recursive Language Model) extracting entities AND relationships from a document.

CRITICAL: You do NOT have the full document in context. It is stored in DOC_VAR.
You must write Python code to extract both entities and relationships.

## Available Variables:
- DOC_VAR: The document text (string)
- SECTION_CONTENT: Current section content (string)
- KNOWN_ENTITY_TYPES: Entity types the system has learned (list)
- KNOWN_RELATIONSHIP_TYPES: Relationship types the system has learned (list)

## Available Functions:
- search_text(pattern): Search for regex pattern, returns list of (start, end, match)
- re.findall(pattern, text): Standard regex
- re.finditer(pattern, text): Iterate matches with positions
- store_variable(name, content): Store findings

## Your Task:
Write Python code that extracts:
1. ENTITIES: People, organizations, dates, money, locations, legal concepts, etc.
2. RELATIONSHIPS: **Meaningful semantic connections between entities.**

## CRITICAL: Relationship Extraction Rules

Relationships MUST connect real-world entities to each other in ways that a human would
describe.  DO NOT create relationships between a table header and a cell value — instead
figure out **which entity** the row is about and link it.

Good examples:
- Passport PA1234567 → BELONGS_TO → GeoV William Sorenssen  (person owns the document)
- GeoV William Sorenssen → EMPLOYED_AT → Downer Engineering  (person works at org)
- GeoV William Sorenssen → BORN_IN → Brisbane               (person born in location)
- GeoV William Sorenssen → SPOUSE_OF → Lisa Marie Krar      (person-to-person)
- GeoV William Sorenssen → HAS_QUALIFICATION → Bachelor of Engineering
- Lisa Marie Krar → CHILD_OF → Margaret Anne Sorenssen       (family link)
- Passport PA1234567 → ISSUED_BY → Australia                 (document issuer)
- SRM Migration → HAS_CONTACT → 0410 031 651                (org has phone)

Bad examples (DO NOT produce these):
- Passport → MENTIONS → PA1234567           (just echoing table cells)
- Passport → HAS_DATE → 2020-01-10          (a date alone without the person)
- Residential Address → REFERENCES → 122A…  (the label is not an entity)

When processing a table row, identify the **subject entity** the row describes (usually a
person, document, or organisation named earlier) and make that subject the source or
target of the relationship.

## Output Format:
```python
entities = []
relationships = []

# Extract entities with exact text positions
for match in re.finditer(r'pattern', SECTION_CONTENT):
    entities.append({{
        "text": match.group(),
        "canonical_name": "Normalized Name",
        "type": "ENTITY_TYPE",
        "start": match.start(),
        "end": match.end(),
        "confidence": 0.9
    }})

# Extract SEMANTIC relationships between entities
# Think: "Who/what is this about?" → "What is the connection?" → "To whom/what?"
relationships.append({{
    "source_text": "GeoV William Sorenssen",  # the real-world entity
    "target_text": "Downer Engineering",       # the related entity
    "type": "EMPLOYED_AT",                     # meaningful relationship type
    "evidence": "GeoV William Sorenssen works at Downer Engineering as Project Manager",
    "confidence": 0.9
}})

store_variable("ENTITIES", entities)
store_variable("RELATIONSHIPS", relationships)
```

## Entity Types:
PERSON, ORGANIZATION, DATE, MONETARY, LOCATION, REFERENCE, DOCUMENT, EVENT, LEGAL_CONCEPT
{learned_entity_types}

## Relationship Types (use the most specific type that fits):
BELONGS_TO      — Document/asset belongs to a person or org
EMPLOYED_AT     — Person works/worked at an organisation
BORN_IN         — Person was born in a location
SPOUSE_OF       — Person is the spouse of another person
CHILD_OF        — Person is the child of another person
SIBLING_OF      — Person is the sibling of another person
ISSUED_BY       — Document/credential issued by org or country
HAS_QUALIFICATION — Person holds a degree, certification, or skill
LOCATED_IN      — Entity is located in a place
HAS_CONTACT     — Person or org has a phone, email, or address
HAS_DATE        — Entity has an associated date (use sparingly; prefer more specific types)
AFFILIATED_WITH — Person/org is affiliated with another org
PARTY_TO        — Entity is party to a document, event, or agreement
TEMPORAL_BEFORE — Event X occurred before Event Y
TEMPORAL_AFTER  — Event X occurred after Event Y
CAUSAL          — Action X caused Outcome Y
REFERENCES      — Section/entity references another entity
MENTIONS        — Entity mentioned in a section (section-level only)
{learned_relationship_types}

Write code appropriate for this specific document."""


RLM_UNIFIED_EXTRACTION_PROMPT = """Document section to extract from:

Section Header: {header}

{ancestor_context}

Section Content (first 3000 chars):
---
{content_preview}
---

Total section length: {content_length} characters

Write Python code to extract ALL entities and SEMANTIC relationships.

Step-by-step:
1. **Use the Document Path and Subject Context above** to identify the SUBJECT of this
   section.  The subject is usually a person, organisation, or document described by the
   parent section.  For example, if the path is
   "Form 80 > PRIMARY APPLICANT DETAILS > Identity Documents" and the subject context
   shows "Family Name: Sorenssen, Given Names: GeoV William", then the subject is
   **GeoV William Sorenssen** and every entity here relates to him.

2. Extract all entities (people, orgs, dates, locations, qualifications, documents, etc.).
   **Include the subject as an entity** even if their name doesn't appear literally in
   this section — you know it from the parent context.

3. For each entity, ask: "How does this relate to the **subject** or to other entities?"
   Create relationships that connect entities to each other — NOT to table headers/labels.

IMPORTANT — For tables, each row should produce a relationship FROM the subject TO the value:
  Subject: "GeoV William Sorenssen" (from parent context)
  Row: "| Passport | PA1234567 | Australia | 10 January 2020 | 10 January 2030 |"
  → Passport PA1234567 → BELONGS_TO → GeoV William Sorenssen
  → Passport PA1234567 → ISSUED_BY → Australia
  → GeoV William Sorenssen → HAS_DATE → 10 January 2020 (passport issue date)

  Row: "| BHP Billiton | Project Engineer | Jan 2015 | Feb 2018 | Australia |"
  → GeoV William Sorenssen → EMPLOYED_AT → BHP Billiton
  → GeoV William Sorenssen → HAS_QUALIFICATION → Project Engineer

End with:
store_variable("ENTITIES", entities)
store_variable("RELATIONSHIPS", relationships)"""


RLM_TOT_VALIDATION_PROMPT = """You are validating extracted entities and relationships using Tree of Thoughts reasoning.

## Extracted Entities:
{entities_json}

## Extracted Relationships:
{relationships_json}

## Section Content (for verification):
{content_preview}

VALIDATION TASK:
For each entity and relationship, estimate probability (0.0-1.0) that it is valid.

ENTITY VALIDATION:
- Is this a real, specific entity (not a generic term)?
- Is the type correct?
- Is the canonical_name properly normalized?

RELATIONSHIP VALIDATION:
- Is there actual evidence for this relationship in the text?
- Is the relationship type correct?
- Are source and target correctly identified?

OUTPUT FORMAT (JSON):
{{
    "entity_validations": [
        {{"id": 0, "valid": true, "probability": 0.9, "type": "PERSON", "canonical_name": "John Smith", "reasoning": "Clear person name with title"}},
        {{"id": 1, "valid": false, "probability": 0.2, "reasoning": "Generic term, not specific entity"}}
    ],
    "relationship_validations": [
        {{"id": 0, "valid": true, "probability": 0.85, "type": "AFFILIATED_WITH", "reasoning": "Evidence shows employment relationship"}},
        {{"id": 1, "valid": false, "probability": 0.3, "reasoning": "Co-occurrence but no explicit relationship"}}
    ],
    "cross_validation": {{
        "entities_in_relationships": [0, 2],
        "orphan_relationships": [],
        "confidence_adjustments": []
    }}
}}

Respond ONLY with JSON."""


# =============================================================================
# Unified REPL for Extraction
# =============================================================================

class UnifiedREPL:
    """
    REPL for unified entity + relationship extraction.
    """
    
    def __init__(
        self,
        document_text: str,
        section_content: str = "",
        known_entity_types: list[str] | None = None,
        known_relationship_types: list[str] | None = None,
    ):
        """Initialize with document and learned types."""
        self.document_text = document_text
        self.section_content = section_content or document_text
        self.known_entity_types = known_entity_types or []
        self.known_relationship_types = known_relationship_types or []
        self.variables: dict[str, Any] = {}
        
        self._namespace = self._build_namespace()
    
    def _build_namespace(self) -> dict[str, Any]:
        """Build Python namespace for code execution."""
        return {
            # Core variables
            "DOC_VAR": self.document_text,
            "SECTION_CONTENT": self.section_content,
            "KNOWN_ENTITY_TYPES": self.known_entity_types,
            "KNOWN_RELATIONSHIP_TYPES": self.known_relationship_types,
            "VARIABLES": self.variables,
            
            # Built-ins
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "sorted": sorted,
            "min": min,
            "max": max,
            "any": any,
            "all": all,
            "re": re,
            
            # Functions
            "search_text": self._search_text,
            "store_variable": self._store_variable,
            "get_variable": self._get_variable,
        }
    
    def _search_text(self, pattern: str) -> list[tuple[int, int, str]]:
        """Search document for regex pattern."""
        results = []
        try:
            for match in re.finditer(pattern, self.section_content, re.IGNORECASE):
                results.append((match.start(), match.end(), match.group()))
        except re.error as e:
            logger.warning("regex_error", pattern=pattern, error=str(e))
        return results
    
    def _store_variable(self, name: str, content: Any) -> str:
        """Store a variable."""
        self.variables[name] = content
        return f"Stored ${name}"
    
    def _get_variable(self, name: str) -> Any:
        """Retrieve a variable."""
        return self.variables.get(name)
    
    def execute(self, code: str) -> dict[str, Any]:
        """Execute Python code."""
        result = {
            "success": False,
            "entities": [],
            "relationships": [],
            "error": None,
            "variables": list(self.variables.keys()),
        }
        
        # Clean code
        code = self._clean_code(code)
        
        try:
            # Compile and execute
            compiled = compile(code, "<rlm_unified_extraction>", "exec")
            exec(compiled, self._namespace)
            
            result["success"] = True
            result["variables"] = list(self.variables.keys())
            result["entities"] = self.variables.get("ENTITIES", [])
            result["relationships"] = self.variables.get("RELATIONSHIPS", [])
            
        except Exception as e:
            result["error"] = str(e)
            logger.warning("rlm_execution_error", error=str(e), code=code[:200])
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown code blocks."""
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        return code.strip()


# =============================================================================
# Unified Extraction Result
# =============================================================================

@dataclass
class RLMUnifiedResult:
    """Result of unified RLM extraction."""
    
    node_id: str = ""
    doc_id: str = ""
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    
    # Code generation
    code_generated: str = ""
    code_executed: bool = False
    
    # Raw candidates (before validation)
    raw_entities: list[dict] = field(default_factory=list)
    raw_relationships: list[dict] = field(default_factory=list)
    
    # Validation
    tot_validated: bool = False
    cross_validated: bool = False
    
    # Stats
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# RLM Unified Extractor
# =============================================================================

class RLMUnifiedExtractor:
    """
    Unified RLM Extractor for entities AND relationships.
    
    This is the RECOMMENDED extractor. It uses:
    1. RLM code generation (LLM writes extraction code)
    2. ToT validation (probabilities + reasoning)
    3. Cross-validation between entities and relationships
    4. Adaptive learning for new types
    
    Always grounded - all extractions tied to exact text spans.
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        max_code_attempts: int = 3,
        tot_selection_threshold: float = 0.6,
        enable_type_learning: bool = True,
        enable_tot_validation: bool = True,
        enable_cross_validation: bool = True,
    ):
        """
        Initialize the unified extractor.
        
        Args:
            llm: LLM instance.
            max_code_attempts: Max attempts if code fails.
            tot_selection_threshold: Threshold for ToT validation.
            enable_type_learning: Learn new entity/relationship types.
            enable_tot_validation: Use ToT for validation.
            enable_cross_validation: Cross-validate entities and relationships.
        """
        self.llm = llm
        self.max_code_attempts = max_code_attempts
        self.tot_selection_threshold = tot_selection_threshold
        self.enable_type_learning = enable_type_learning
        self.enable_tot_validation = enable_tot_validation
        self.enable_cross_validation = enable_cross_validation
        
        self._llm_initialized = False
        
        # Learned type registries
        self._entity_type_registry = None
        self._relationship_type_registry = None
        
        if enable_type_learning:
            self._entity_type_registry = get_learned_type_registry()
            try:
                self._relationship_type_registry = get_learned_relationship_type_registry()
            except Exception:
                # Registry may not exist yet
                self._relationship_type_registry = None
    
    def _get_llm(self) -> Any:
        """Get or initialize LLM.
        
        Checks the ``RNSR_EXTRACTION_MODEL`` environment variable first.
        This lets you pin a small, fast model (e.g. ``gemini-2.5-flash``,
        ``gpt-5-mini``) for entity extraction while keeping a larger model
        for navigation / answer synthesis.
        """
        if self.llm is None and not self._llm_initialized:
            import os
            from rnsr.llm import LLMProvider

            extraction_model = os.getenv("RNSR_EXTRACTION_MODEL")
            extraction_provider_str = os.getenv("RNSR_EXTRACTION_PROVIDER")

            kwargs: dict[str, Any] = {}
            if extraction_model:
                kwargs["model"] = extraction_model
            if extraction_provider_str:
                try:
                    kwargs["provider"] = LLMProvider(extraction_provider_str.lower())
                except ValueError:
                    logger.warning(
                        "invalid_extraction_provider",
                        value=extraction_provider_str,
                        valid=list(LLMProvider),
                    )

            if kwargs:
                logger.info(
                    "extraction_llm_override",
                    **kwargs,
                )
                self.llm = get_llm(**kwargs)
            else:
                self.llm = get_llm()
            self._llm_initialized = True
        return self.llm
    
    def extract(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        page_num: int | None = None,
        document_text: str | None = None,
        ancestor_context: str | None = None,
    ) -> RLMUnifiedResult:
        """
        Extract entities AND relationships using unified RLM approach.
        
        Flow:
        1. LLM generates extraction code based on document
        2. Code executes on DOC_VAR (grounded)
        3. ToT validates candidates with probabilities
        4. Cross-validation boosts/filters
        5. Learn new types
        
        Args:
            node_id: Section node ID.
            doc_id: Document ID.
            header: Section header.
            content: Section content.
            page_num: Page number.
            document_text: Full document text for DOC_VAR.
            ancestor_context: Hierarchical breadcrumb + subject hint
                so the extractor knows the broader context (e.g. whose
                passport details these are).
            
        Returns:
            RLMUnifiedResult with entities and relationships.
        """
        start_time = time.time()
        
        result = RLMUnifiedResult(
            node_id=node_id,
            doc_id=doc_id,
        )
        
        if len(content.strip()) < 50:
            return result
        
        llm = self._get_llm()
        if llm is None:
            result.warnings.append("No LLM available")
            return result
        
        # Get learned types for prompt
        learned_entity_types = self._get_learned_entity_types()
        learned_relationship_types = self._get_learned_relationship_types()
        
        # STEP 1: Generate and execute extraction code
        exec_result = self._generate_and_execute_code(
            header=header,
            content=content,
            document_text=document_text or content,
            learned_entity_types=learned_entity_types,
            learned_relationship_types=learned_relationship_types,
            ancestor_context=ancestor_context or "",
        )
        
        result.code_generated = exec_result.get("code", "")
        result.code_executed = exec_result.get("success", False)
        result.raw_entities = exec_result.get("entities", [])
        result.raw_relationships = exec_result.get("relationships", [])
        
        if not result.code_executed:
            result.warnings.append(f"Code execution failed: {exec_result.get('error', 'Unknown')}")
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        # STEP 2: ToT Validation
        if self.enable_tot_validation and (result.raw_entities or result.raw_relationships):
            validated = self._tot_validate(
                entities=result.raw_entities,
                relationships=result.raw_relationships,
                content=content,
            )
            entities = validated.get("entities", result.raw_entities)
            relationships = validated.get("relationships", result.raw_relationships)
            result.tot_validated = True
        else:
            entities = result.raw_entities
            relationships = result.raw_relationships
        
        # STEP 3: Convert to model objects
        result.entities = self._candidates_to_entities(
            candidates=entities,
            node_id=node_id,
            doc_id=doc_id,
            content=content,
            page_num=page_num,
        )
        
        result.relationships = self._candidates_to_relationships(
            candidates=relationships,
            entities=result.entities,
            node_id=node_id,
            doc_id=doc_id,
        )
        
        # STEP 4: Cross-validation
        if self.enable_cross_validation and result.entities and result.relationships:
            result.entities, result.relationships = self._cross_validate(
                result.entities, result.relationships
            )
            result.cross_validated = True
        
        # STEP 5: Learn new types
        if self.enable_type_learning:
            self._learn_new_types(result.entities, result.relationships)
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "rlm_unified_extraction_complete",
            node_id=node_id,
            entities=len(result.entities),
            relationships=len(result.relationships),
            time_ms=result.processing_time_ms,
        )
        
        return result
    
    def _get_learned_entity_types(self) -> str:
        """Get learned entity types for prompt."""
        if not self._entity_type_registry:
            return ""
        
        types = self._entity_type_registry.get_types_for_prompt()
        if not types:
            return ""
        
        return f"\nAlso consider these learned types: {', '.join(types)}"
    
    def _get_learned_relationship_types(self) -> str:
        """Get learned relationship types for prompt."""
        if not self._relationship_type_registry:
            return ""
        
        try:
            types = self._relationship_type_registry.get_types_for_prompt()
            if not types:
                return ""
            
            logger.info(
                "injecting_learned_relationship_types",
                count=len(types),
                types=types[:10],
            )
            return f"\nAlso consider these learned relationship types: {', '.join(types)}"
        except Exception:
            return ""
    
    def _generate_and_execute_code(
        self,
        header: str,
        content: str,
        document_text: str,
        learned_entity_types: str,
        learned_relationship_types: str,
        ancestor_context: str = "",
    ) -> dict[str, Any]:
        """Generate extraction code and execute it."""
        llm = self._get_llm()
        
        # Get learned types for REPL
        entity_types = []
        relationship_types = []
        
        if self._entity_type_registry:
            entity_types = self._entity_type_registry.get_types_for_prompt()
        if self._relationship_type_registry:
            try:
                relationship_types = self._relationship_type_registry.get_types_for_prompt()
            except Exception:
                pass
        
        # Create REPL
        repl = UnifiedREPL(
            document_text=document_text,
            section_content=content,
            known_entity_types=entity_types,
            known_relationship_types=relationship_types,
        )
        
        # Build prompt
        system_prompt = RLM_UNIFIED_SYSTEM_PROMPT.format(
            learned_entity_types=learned_entity_types,
            learned_relationship_types=learned_relationship_types,
        )
        
        extraction_prompt = RLM_UNIFIED_EXTRACTION_PROMPT.format(
            header=header,
            content_preview=content[:3000],
            content_length=len(content),
            ancestor_context=ancestor_context,
        )
        
        prompt = f"{system_prompt}\n\n{extraction_prompt}"
        
        for attempt in range(self.max_code_attempts):
            try:
                # LLM generates code
                response = llm.complete(prompt)
                code = str(response) if not isinstance(response, str) else response
                
                # Validate we got actual code
                if not code or len(code.strip()) < 20:
                    logger.warning("empty_or_short_code_response", attempt=attempt, length=len(code) if code else 0)
                    continue
                
                # Check if response looks like code (not just JSON or text)
                if "store_variable" not in code and "entities" not in code.lower():
                    logger.warning("response_not_code", attempt=attempt, preview=code[:100])
                    prompt += "\n\nPlease respond ONLY with Python code that extracts entities and relationships."
                    continue
                
                # Execute
                exec_result = repl.execute(code)
                
                if exec_result["success"]:
                    entities = exec_result.get("entities", [])
                    relationships = exec_result.get("relationships", [])
                    
                    # Validate entities are properly structured
                    valid_entities = []
                    for e in entities:
                        if isinstance(e, dict) and e.get("text"):
                            valid_entities.append(e)
                    
                    valid_relationships = []
                    for r in relationships:
                        if isinstance(r, dict) and (r.get("source_text") or r.get("type")):
                            valid_relationships.append(r)
                    
                    return {
                        "success": True,
                        "code": code,
                        "entities": valid_entities,
                        "relationships": valid_relationships,
                    }
                else:
                    # Retry with error feedback
                    prompt += f"\n\nPrevious code had error: {exec_result['error']}\nPlease fix."
                    
            except Exception as e:
                logger.warning("code_generation_failed", attempt=attempt, error=str(e))
        
        return {"success": False, "error": "Max attempts exceeded"}
    
    def _tot_validate(
        self,
        entities: list[dict],
        relationships: list[dict],
        content: str,
    ) -> dict[str, list[dict]]:
        """Validate with Tree of Thoughts."""
        llm = self._get_llm()
        
        entities_json = json.dumps([
            {"id": i, "text": e.get("text", ""), "type": e.get("type", ""), "confidence": e.get("confidence", 0.5)}
            for i, e in enumerate(entities[:20])
        ], indent=2)
        
        relationships_json = json.dumps([
            {"id": i, "source": r.get("source_text", ""), "target": r.get("target_text", ""), 
             "type": r.get("type", ""), "evidence": r.get("evidence", "")[:100]}
            for i, r in enumerate(relationships[:15])
        ], indent=2)
        
        prompt = RLM_TOT_VALIDATION_PROMPT.format(
            entities_json=entities_json,
            relationships_json=relationships_json,
            content_preview=content[:2000],
        )
        
        try:
            response = llm.complete(prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            # Clean response - remove markdown code blocks if present
            response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
            response_text = response_text.strip()
            
            # Parse JSON - try multiple strategies
            data = None
            
            # Strategy 1: Direct parse
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Extract JSON object
            if data is None:
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Try to fix common issues
            if data is None:
                # Try fixing trailing commas, missing quotes, etc.
                fixed = re.sub(r',(\s*[}\]])', r'\1', response_text)
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    pass
            
            if data is None:
                logger.debug("tot_json_parse_failed", response_preview=response_text[:200])
                return {"entities": entities, "relationships": relationships}
            
            # Apply validations
            entity_validations = {v["id"]: v for v in data.get("entity_validations", [])}
            relationship_validations = {v["id"]: v for v in data.get("relationship_validations", [])}
            
            # Filter and update entities
            validated_entities = []
            for i, entity in enumerate(entities):
                validation = entity_validations.get(i, {})
                if validation.get("valid", True) and validation.get("probability", 0.5) >= self.tot_selection_threshold:
                    entity["type"] = validation.get("type", entity.get("type", "OTHER"))
                    entity["canonical_name"] = validation.get("canonical_name", entity.get("canonical_name", entity.get("text", "")))
                    entity["confidence"] = validation.get("probability", entity.get("confidence", 0.5))
                    entity["tot_reasoning"] = validation.get("reasoning", "")
                    validated_entities.append(entity)
            
            # Filter and update relationships
            validated_relationships = []
            for i, rel in enumerate(relationships):
                validation = relationship_validations.get(i, {})
                if validation.get("valid", True) and validation.get("probability", 0.5) >= self.tot_selection_threshold:
                    rel["type"] = validation.get("type", rel.get("type", "MENTIONS"))
                    rel["confidence"] = validation.get("probability", rel.get("confidence", 0.5))
                    rel["tot_reasoning"] = validation.get("reasoning", "")
                    validated_relationships.append(rel)
            
            return {"entities": validated_entities, "relationships": validated_relationships}
            
        except Exception as e:
            logger.warning("tot_validation_failed", error=str(e))
            return {"entities": entities, "relationships": relationships}
    
    def _candidates_to_entities(
        self,
        candidates: list[dict],
        node_id: str,
        doc_id: str,
        content: str,
        page_num: int | None,
    ) -> list[Entity]:
        """Convert candidates to Entity objects."""
        entities = []
        
        for candidate in candidates:
            if not candidate.get("text"):
                continue
            
            entity_type = self._map_entity_type(candidate.get("type", "OTHER"))
            
            mention = Mention(
                node_id=node_id,
                doc_id=doc_id,
                span_start=candidate.get("start"),
                span_end=candidate.get("end"),
                context=content[
                    max(0, (candidate.get("start") or 0) - 50):
                    (candidate.get("end") or 0) + 50
                ] if candidate.get("start") is not None else "",
                page_num=page_num,
                confidence=candidate.get("confidence", 0.5),
            )
            
            metadata = {
                "rlm_extracted": True,
                "grounded": candidate.get("start") is not None,
                "tot_validated": "tot_reasoning" in candidate,
            }
            
            if candidate.get("tot_reasoning"):
                metadata["tot_reasoning"] = candidate["tot_reasoning"]
            
            if entity_type == EntityType.OTHER:
                metadata["original_type"] = candidate.get("type", "").lower()
            
            entity = Entity(
                type=entity_type,
                canonical_name=candidate.get("canonical_name", candidate.get("text", "")),
                aliases=[candidate.get("text")] if candidate.get("canonical_name") != candidate.get("text") else [],
                mentions=[mention],
                metadata=metadata,
                source_doc_id=doc_id,
            )
            entities.append(entity)
        
        return entities
    
    def _candidates_to_relationships(
        self,
        candidates: list[dict],
        entities: list[Entity],
        node_id: str,
        doc_id: str,
    ) -> list[Relationship]:
        """Convert candidates to Relationship objects."""
        relationships = []
        
        # Build entity lookup
        entity_by_text = {}
        for entity in entities:
            entity_by_text[entity.canonical_name.lower()] = entity.id
            for alias in entity.aliases:
                entity_by_text[alias.lower()] = entity.id
        
        for candidate in candidates:
            rel_type = self._map_relationship_type(candidate.get("type", "MENTIONS"))
            
            # Try to match source/target to entities
            source_text = candidate.get("source_text", "")
            target_text = candidate.get("target_text", "")
            
            source_id = entity_by_text.get(source_text.lower(), f"text:{source_text}")
            target_id = entity_by_text.get(target_text.lower(), f"text:{target_text}")
            
            source_type = "entity" if source_id in [e.id for e in entities] else "text"
            target_type = "entity" if target_id in [e.id for e in entities] else "text"
            
            metadata = {
                "rlm_extracted": True,
                "grounded": candidate.get("start") is not None,
                "tot_validated": "tot_reasoning" in candidate,
            }
            
            if candidate.get("tot_reasoning"):
                metadata["tot_reasoning"] = candidate["tot_reasoning"]
            
            if rel_type == RelationType.OTHER:
                metadata["original_type"] = candidate.get("type", "").lower()
            
            relationship = Relationship(
                type=rel_type,
                source_id=source_id,
                source_type=source_type,
                target_id=target_id,
                target_type=target_type,
                confidence=candidate.get("confidence", 0.5),
                evidence=candidate.get("evidence", ""),
                doc_id=doc_id,
                node_id=node_id,
                metadata=metadata,
            )
            relationships.append(relationship)
        
        return relationships
    
    def _cross_validate(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship]]:
        """Cross-validate entities and relationships."""
        entity_ids = {e.id for e in entities}
        
        # Find entities referenced in relationships
        entities_in_rels = set()
        for rel in relationships:
            if rel.source_type == "entity" and rel.source_id in entity_ids:
                entities_in_rels.add(rel.source_id)
            if rel.target_type == "entity" and rel.target_id in entity_ids:
                entities_in_rels.add(rel.target_id)
        
        # Boost confidence for entities in relationships
        for entity in entities:
            if entity.id in entities_in_rels:
                if entity.mentions:
                    entity.mentions[0].confidence = min(entity.mentions[0].confidence * 1.1, 1.0)
                entity.metadata["cross_validated"] = True
        
        # Boost confidence for relationships with validated entities
        for rel in relationships:
            both_valid = (
                (rel.source_type == "entity" and rel.source_id in entity_ids) and
                (rel.target_type == "entity" and rel.target_id in entity_ids)
            )
            if both_valid:
                rel.confidence = min(rel.confidence * 1.1, 1.0)
                rel.metadata["cross_validated"] = True
        
        return entities, relationships
    
    # Fuzzy-match table for auto-suggesting mappings of learned relationship types
    _LEARNED_REL_KEYWORDS: dict[str, str] = {
        "employ": "employed_at",
        "work": "employed_at",
        "hire": "employed_at",
        "job": "employed_at",
        "born": "born_in",
        "birth": "born_in",
        "spouse": "spouse_of",
        "married": "spouse_of",
        "husband": "spouse_of",
        "wife": "spouse_of",
        "child": "child_of",
        "parent": "child_of",
        "son": "child_of",
        "daughter": "child_of",
        "sibling": "sibling_of",
        "brother": "sibling_of",
        "sister": "sibling_of",
        "belong": "belongs_to",
        "own": "belongs_to",
        "holder": "belongs_to",
        "issue": "issued_by",
        "grant": "issued_by",
        "certif": "issued_by",
        "locat": "located_in",
        "address": "located_in",
        "resid": "located_in",
        "live": "located_in",
        "contact": "has_contact",
        "phone": "has_contact",
        "email": "has_contact",
        "mobile": "has_contact",
        "qualif": "has_qualification",
        "degree": "has_qualification",
        "educat": "has_qualification",
        "stud": "has_qualification",
        "affil": "affiliated_with",
        "citizen": "affiliated_with",
        "national": "affiliated_with",
        "member": "affiliated_with",
        "date": "has_date",
        "expir": "has_date",
        "refer": "references",
        "mention": "mentions",
        "cause": "causal",
        "before": "temporal_before",
        "after": "temporal_after",
        "party": "party_to",
    }

    def _learn_new_types(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Learn new entity and relationship types.
        
        For relationship types that fall through to OTHER, we also try to
        auto-suggest a canonical mapping based on keyword matching so that
        future documents benefit without waiting for manual curation.
        """
        # Learn entity types
        if self._entity_type_registry:
            for entity in entities:
                if entity.type == EntityType.OTHER:
                    original_type = entity.metadata.get("original_type", "unknown")
                    context = entity.mentions[0].context if entity.mentions else ""
                    self._entity_type_registry.record_type(
                        type_name=original_type,
                        context=context,
                        entity_name=entity.canonical_name,
                    )
        
        # Learn relationship types
        if self._relationship_type_registry:
            for rel in relationships:
                if rel.type == RelationType.OTHER:
                    original_type = rel.metadata.get("original_type", "unknown")
                    try:
                        self._relationship_type_registry.record_type(
                            type_name=original_type,
                            context=rel.evidence,
                            relationship_description=f"{rel.source_id} -> {rel.target_id}",
                        )
                        
                        # Auto-suggest a canonical mapping via keyword matching
                        self._auto_suggest_relationship_mapping(original_type)
                    except Exception:
                        pass

    def _auto_suggest_relationship_mapping(self, type_name: str) -> None:
        """Try to auto-map a learned relationship type to a canonical RelationType."""
        if not self._relationship_type_registry:
            return
        
        # Skip if already has a mapping
        existing = self._relationship_type_registry.get_mappings()
        if type_name.lower() in existing:
            return
        
        # Check if any keyword in the type_name matches a canonical type
        lower = type_name.lower().replace("_", " ")
        for keyword, canonical in self._LEARNED_REL_KEYWORDS.items():
            if keyword in lower:
                self._relationship_type_registry.suggest_mapping(type_name, canonical)
                logger.info(
                    "auto_mapped_learned_relationship_type",
                    learned_type=type_name,
                    mapped_to=canonical,
                    matched_keyword=keyword,
                )
                return
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map type string to EntityType enum."""
        type_str = type_str.upper()
        
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "ORG": EntityType.ORGANIZATION,
            "COMPANY": EntityType.ORGANIZATION,
            "DATE": EntityType.DATE,
            "MONETARY": EntityType.MONETARY,
            "MONEY": EntityType.MONETARY,
            "LOCATION": EntityType.LOCATION,
            "REFERENCE": EntityType.REFERENCE,
            "DOCUMENT": EntityType.DOCUMENT,
            "EVENT": EntityType.EVENT,
            "LEGAL_CONCEPT": EntityType.LEGAL_CONCEPT,
            "LEGAL": EntityType.LEGAL_CONCEPT,
        }
        
        try:
            return EntityType(type_str.lower())
        except ValueError:
            return mapping.get(type_str, EntityType.OTHER)
    
    def _map_relationship_type(self, type_str: str) -> RelationType:
        """Map type string to RelationType enum, including common aliases."""
        type_str = type_str.upper().strip()
        
        # Canonical mapping (exact match on upper-cased key)
        mapping = {
            # Core types
            "MENTIONS": RelationType.MENTIONS,
            "DEFINED_IN": RelationType.DEFINED_IN,
            "TEMPORAL_BEFORE": RelationType.TEMPORAL_BEFORE,
            "TEMPORAL_AFTER": RelationType.TEMPORAL_AFTER,
            "CAUSAL": RelationType.CAUSAL,
            "SUPPORTS": RelationType.SUPPORTS,
            "CONTRADICTS": RelationType.CONTRADICTS,
            "AFFILIATED_WITH": RelationType.AFFILIATED_WITH,
            "PARTY_TO": RelationType.PARTY_TO,
            "REFERENCES": RelationType.REFERENCES,
            "SUPERSEDES": RelationType.SUPERSEDES,
            "AMENDS": RelationType.AMENDS,
            # New semantic types
            "BELONGS_TO": RelationType.BELONGS_TO,
            "EMPLOYED_AT": RelationType.EMPLOYED_AT,
            "BORN_IN": RelationType.BORN_IN,
            "SPOUSE_OF": RelationType.SPOUSE_OF,
            "CHILD_OF": RelationType.CHILD_OF,
            "SIBLING_OF": RelationType.SIBLING_OF,
            "ISSUED_BY": RelationType.ISSUED_BY,
            "HAS_QUALIFICATION": RelationType.HAS_QUALIFICATION,
            "LOCATED_IN": RelationType.LOCATED_IN,
            "HAS_CONTACT": RelationType.HAS_CONTACT,
            "HAS_DATE": RelationType.HAS_DATE,
            # Common aliases the LLM may produce
            "WORKS_AT": RelationType.EMPLOYED_AT,
            "EMPLOYED_BY": RelationType.EMPLOYED_AT,
            "WORKS_FOR": RelationType.EMPLOYED_AT,
            "EMPLOYEE_OF": RelationType.EMPLOYED_AT,
            "WORKED_AT": RelationType.EMPLOYED_AT,
            "BORN_AT": RelationType.BORN_IN,
            "PLACE_OF_BIRTH": RelationType.BORN_IN,
            "MARRIED_TO": RelationType.SPOUSE_OF,
            "SPOUSE": RelationType.SPOUSE_OF,
            "PARENT_OF": RelationType.CHILD_OF,    # reversed at call site if needed
            "SON_OF": RelationType.CHILD_OF,
            "DAUGHTER_OF": RelationType.CHILD_OF,
            "BROTHER_OF": RelationType.SIBLING_OF,
            "SISTER_OF": RelationType.SIBLING_OF,
            "OWNS": RelationType.BELONGS_TO,
            "OWNED_BY": RelationType.BELONGS_TO,
            "HOLDER_OF": RelationType.BELONGS_TO,
            "ISSUED_TO": RelationType.BELONGS_TO,
            "HAS_DOCUMENT": RelationType.BELONGS_TO,
            "LOCATED_AT": RelationType.LOCATED_IN,
            "LIVES_IN": RelationType.LOCATED_IN,
            "RESIDES_IN": RelationType.LOCATED_IN,
            "ADDRESS": RelationType.LOCATED_IN,
            "HAS_ADDRESS": RelationType.LOCATED_IN,
            "HAS_EMAIL": RelationType.HAS_CONTACT,
            "HAS_PHONE": RelationType.HAS_CONTACT,
            "CONTACT": RelationType.HAS_CONTACT,
            "HAS_CONTACT_DETAIL": RelationType.HAS_CONTACT,
            "EDUCATED_AT": RelationType.HAS_QUALIFICATION,
            "STUDIED_AT": RelationType.HAS_QUALIFICATION,
            "DEGREE": RelationType.HAS_QUALIFICATION,
            "HAS_DEGREE": RelationType.HAS_QUALIFICATION,
            "QUALIFICATION": RelationType.HAS_QUALIFICATION,
            "CERTIFIED_BY": RelationType.ISSUED_BY,
            "GRANTED_BY": RelationType.ISSUED_BY,
            "CITIZEN_OF": RelationType.AFFILIATED_WITH,
            "CITIZENSHIP": RelationType.AFFILIATED_WITH,
            "NATIONALITY": RelationType.AFFILIATED_WITH,
            "MEMBER_OF": RelationType.AFFILIATED_WITH,
            "HAS_DATE_OF_BIRTH": RelationType.BORN_IN,
            "DATE_OF_BIRTH": RelationType.HAS_DATE,
            "EXPIRY_DATE": RelationType.HAS_DATE,
            "ISSUE_DATE": RelationType.HAS_DATE,
            "START_DATE": RelationType.HAS_DATE,
            "END_DATE": RelationType.HAS_DATE,
        }
        
        try:
            return RelationType(type_str.lower())
        except ValueError:
            pass
        
        if type_str in mapping:
            return mapping[type_str]
        
        # Check learned relationship type mappings from the registry
        if self._relationship_type_registry:
            learned_mappings = self._relationship_type_registry.get_mappings()
            mapped_value = learned_mappings.get(type_str.lower())
            if mapped_value:
                try:
                    return RelationType(mapped_value)
                except ValueError:
                    pass
        
        return RelationType.OTHER
    
    def to_extraction_result(self, unified_result: RLMUnifiedResult) -> ExtractionResult:
        """Convert to standard ExtractionResult format."""
        return ExtractionResult(
            node_id=unified_result.node_id,
            doc_id=unified_result.doc_id,
            entities=unified_result.entities,
            relationships=unified_result.relationships,
            processing_time_ms=unified_result.processing_time_ms,
            extraction_method="rlm_unified",
            warnings=unified_result.warnings,
        )


# Convenience function
def extract_entities_and_relationships(
    node_id: str,
    doc_id: str,
    header: str,
    content: str,
    page_num: int | None = None,
    ancestor_context: str | None = None,
) -> RLMUnifiedResult:
    """
    Extract entities and relationships using the unified RLM approach.
    
    This is the recommended way to extract - always uses the most
    accurate, grounded approach with ToT validation.
    
    Args:
        ancestor_context: Hierarchical breadcrumb and subject hint so
            the LLM knows *who/what* the section is about.
    """
    extractor = RLMUnifiedExtractor()
    return extractor.extract(
        node_id=node_id,
        doc_id=doc_id,
        header=header,
        content=content,
        page_num=page_num,
        ancestor_context=ancestor_context,
    )
