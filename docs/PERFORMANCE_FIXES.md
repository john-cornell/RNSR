# RNSR Performance Fixes - Applied January 26, 2026

## Problem Summary

RNSR achieved 0% exact match vs 55% baseline on QuALITY benchmark. Analysis revealed:

1. **Navigation was working** - ToT found correct sections (e.g., Section 3 for "haggle" question)
2. **Answer synthesis was broken** - System had right content but said "Cannot determine"
3. **Node summaries were insufficient** - Generic truncation lost key facts needed for ToT

## Root Causes

### Issue 1: Over-Conservative Synthesis Prompt
**Evidence:** Question 5 had correct content (Section 3, 1,957 chars) but returned "Cannot determine"

**Original Prompt:**
```
"If the answer cannot be determined from the context, say 'Cannot determine from available context.'"
```

**Problem:** LLM defaulted to "cannot determine" when any uncertainty existed, even with sufficient evidence.

### Issue 2: Poor Node Summaries
**Evidence:** Summaries were generic 75-word truncations, losing:
- Named entities (Blake, Eldoria)
- Specific facts (prices, actions)
- Context needed for ToT evaluation

**Original Approach:**
```python
# Simple truncation with ellipsis
return " ".join(words[:75]) + "..."
```

**Problem:** 
- Lost key details after word 75
- No prioritization of important facts
- ToT couldn't distinguish relevant vs irrelevant nodes

### Issue 3: ToT Not Selective Enough
**Evidence:** ToT selected 4 nodes when 1 was needed, adding 60s latency

**Original:** Selected top-k=3 with low threshold (0.1)

## Fixes Applied

### Fix 1: More Confident Synthesis Prompt ✅

**File:** `rnsr/agent/graph.py` (lines 1032-1046)

**Changes:**
```python
prompt = """Based on the following context, answer the question concisely.

IMPORTANT INSTRUCTIONS:
- If the context contains information that DIRECTLY answers the question, provide that answer
- If the context contains information that allows you to INFER the answer, provide your best inference
- Use evidence from the context to support your answer
- Only say "Cannot determine from available context" if the context is completely unrelated or missing critical information
- It's better to provide a reasonable answer based on available evidence than to say "cannot determine"

Question: {question}

Context:
{context_text}

Answer (be concise, direct, and confident):"""
```

**Expected Impact:**
- Reduce "Cannot determine" rate from 25% to <10%
- Increase willingness to infer from context
- Better utilization of the content ToT already found

### Fix 2: Extractive Node Summaries ✅

**File:** `rnsr/indexing/skeleton_index.py` (lines 40-65, 82-96)

**Changes:**

1. **Increased max_words from 75 to 100**
   - More room for key facts
   - Better preservation of context

2. **Improved documentation**
```python
def generate_summary(content: str, max_words: int = 100) -> str:
    """
    EXTRACTIVE approach: Take first portion to preserve key facts,
    entities, and concrete details that ToT needs for evaluation.
    """
```

3. **Better LLM summary prompt**
```python
prompt = """Summarize the following text in {max_words} words or less.

IMPORTANT: Use an EXTRACTIVE approach - preserve:
- Key facts, entities, names, and concrete details (who, what, when, where)
- Specific actions, events, and outcomes
- Numbers, dates, and measurements
- The main subject and what happens to/with it

Avoid:
- Vague generalizations ("discusses various topics")
- Meta-commentary ("this section explains...")
- Abstractions without specifics
"""
```

**Expected Impact:**
- Summaries contain "Blake", "Eldoria", "3000 quandoes" instead of generic text
- ToT can better distinguish relevant sections
- Reduced node selection errors

### Fix 3: More Selective ToT Evaluation ✅

**File:** `rnsr/agent/graph.py` (lines 318-344)

**Changes:**
```python
INSTRUCTIONS:
1. Evaluate: For each child node, analyze its summary and estimate relevance probability.
2. Be SELECTIVE: Only select nodes with probability > 0.6 (strong evidence of relevance)
3. Look for SPECIFIC matches: Prefer nodes with concrete facts/entities mentioned in query
4. Aim for PRECISION over RECALL: Better to select 1 highly relevant node than 5 moderately relevant
5. Plan: Select top-{top_k} most promising nodes (or fewer if only a few are truly relevant)
6. Reasoning: Explain what SPECIFIC content in summary makes it relevant
7. Backtrack Signal: If NO child seems relevant (all probabilities < 0.3), report "DEAD_END"
```

**Key Changes:**
- Raised selection threshold from 0.1 to 0.6
- Emphasized PRECISION over RECALL
- Required SPECIFIC reasoning (not vague)
- Allow selecting fewer than top_k if appropriate

**Expected Impact:**
- Select 1-2 highly relevant nodes instead of 4+ moderate ones
- Reduce latency (fewer nodes visited)
- Less noise in synthesis context
- Faster convergence to answer

## Expected Results

### Performance Predictions

| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| Answer EM | 0.0% | 30-40% | +30-40% |
| Answer F1 | 14.1% | 40-50% | +26-36% |
| "Cannot Determine" Rate | 25% | <10% | -15% |
| Avg Time per Question | 53s | 35-40s | -25-35% |
| Nodes Visited (avg) | 7.2 | 4-5 | -30% |

### Why These Improvements

**1. Synthesis Confidence → +30% EM**
- Questions where ToT found right content (75% of cases) will now answer
- Only truly unanswerable questions will say "cannot determine"

**2. Better Summaries → +15% accuracy**
- ToT can distinguish relevant from irrelevant nodes
- Fewer wrong selections
- Less wasted exploration

**3. Selective ToT → -30% latency**
- Visit 4-5 nodes instead of 7
- Each node saves ~10s processing
- Still maintain accuracy (precision > recall)

## Testing Plan

### Phase 1: Verify Fixes (2 minutes)
```bash
python rnsr/benchmarks/evaluation_suite.py --datasets quality --samples 5 --llm-provider gemini
```

**Check:**
- Synthesis no longer always says "Cannot determine"
- ToT selects fewer nodes (1-3 instead of 4+)
- Answers are more specific

### Phase 2: Full Benchmark (5 minutes)
```bash
python rnsr/benchmarks/evaluation_suite.py --datasets quality --samples 20 --llm-provider gemini
```

**Expected:**
- Answer EM: 30-40% (vs 0% before)
- Avg time: 35-40s (vs 53s before)
- Still slower than baseline but with better contextual understanding

### Phase 3: Error Analysis
If results are still poor:
1. Check synthesis prompt output - is it still too conservative?
2. Examine node summaries - do they contain key facts?
3. Review ToT selections - is it still too broad?

## Remaining Issues (Future Work)

### Not Addressed Yet

1. **Context window optimization**
   - Still storing 2000 chars when answer is 1 sentence
   - Could extract key sentences when storing variables
   - Would improve synthesis focus

2. **Adaptive top_k**
   - Currently fixed at 3
   - Could reduce to 1-2 for simple questions
   - Increase to 5+ for complex multi-hop

3. **Hybrid search fallback**
   - If ToT finds nothing, could fall back to semantic search
   - Would reduce "cannot determine" rate further

4. **Summary quality validation**
   - Need to verify summaries actually contain key facts
   - Could add unit tests comparing summary to full content

## Code Changes Summary

**Modified Files:**
1. `rnsr/agent/graph.py`
   - Lines 1032-1046: Synthesis prompt (more confident)
   - Lines 318-344: ToT prompt (more selective)

2. `rnsr/indexing/skeleton_index.py`
   - Lines 40-65: Extractive summary function
   - Lines 82-96: Better LLM summary prompt

**No Breaking Changes:**
- All APIs unchanged
- Backward compatible
- Can revert easily if needed

## Success Metrics

**Minimum Acceptable:**
- Answer EM > 20% (vs 0% before)
- Answer F1 > 30% (vs 14% before)
- Avg time < 45s (vs 53s before)

**Target:**
- Answer EM > 35%
- Answer F1 > 45%
- Avg time < 40s

**Stretch Goal:**
- Match baseline accuracy (55% EM)
- With better contextual understanding (ToT path traceability)
- In comparable time (40-50s acceptable for "deep research")
