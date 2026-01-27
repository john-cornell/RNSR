# RNSR Benchmark Failure Analysis
## Date: January 26, 2026

## Executive Summary

RNSR achieved **0% exact match** compared to naive baseline's **55% exact match** on the QuALITY dataset (20 questions). The system is **5.7x slower** (53s vs 9s per question).

## Critical Findings

### 1. **RNSR FINDS the right content but FAILS to answer** ✅❌

**Key Insight:** In Question 5 ("Why doesn't Blake haggle with Eldoria?"):
- ✅ ToT correctly selected `seg_f20e93f8` (Section 3) - **the section containing the answer**
- ✅ Variable stitching stored 1,957 chars from that section
- ✅ Synthesis used this variable with confidence 1.0
- ❌ **Still answered: "Cannot determine from available context"**

**This proves:** The navigation is working reasonably well, but **answer generation is broken**.

### 2. **Root Causes Identified**

#### A. Over-Conservative Synthesis Prompt
The final synthesis step is refusing to answer even when it has the relevant content. The prompt likely:
- Sets too high a bar for "certainty"
- Doesn't recognize implicit/inferential answers
- Defaults to "Cannot determine" when any ambiguity exists

#### B. Variable Stitching Loses Critical Details
When content is stored as `$SECTION_3`, the specific sentence about Blake not haggling gets buried in 1,957 characters of context. The synthesis LLM may be:
- Overwhelmed by too much context
- Missing the specific detail that answers the question
- Unable to connect the question to the relevant portion of the variable

#### C. ToT Evaluation is Not Precise Enough
While ToT selected the right section, it also selected 3 other sections:
- `seg_20dc79e1` (Section 11) - about Deirdre and Blake Past
- `seg_004f769c` (Section 16) - unrelated content  
- `seg_26467941` (Section 4) - about Deirdre's education

**Problem:** ToT is selecting 4 nodes when only 1 is needed. This:
- Adds 60s of processing time (visiting + storing 3 irrelevant sections)
- Dilutes the relevant content with noise
- Confuses the synthesis step

### 3. **Performance Metrics**

| Metric | RNSR | Baseline | Delta |
|--------|------|----------|-------|
| Answer EM | 0.0% | 55.0% | **-55.0%** |
| Answer F1 | 14.1% | 57.9% | **-43.8%** |
| Avg Time | 53.3s | 9.3s | **+470%** |
| Nodes Visited (avg) | 7.2 | N/A | - |

### 4. **Failure Pattern Analysis**

Out of 20 questions:
- **5 (25%)** - RNSR said "Cannot determine" (complete failures)
- **15 (75%)** - RNSR attempted an answer but with 0% exact match

Even the 15 "answered" questions have **extremely low quality**:
- F1 score of ~14% suggests answers are vague or wrong
- Compare to baseline's 58% F1 score

## What's Working vs. What's Broken

### ✅ What's Working
1. **Tree-based indexing** - Documents are being parsed and indexed into hierarchical structure
2. **Variable stitching** - Content is being stored and retrieved correctly  
3. **ToT navigation** - Selecting relevant nodes (though not optimally)
4. **RLM decomposition** - Questions are being broken down into sub-questions

### ❌ What's Broken
1. **Answer synthesis** - The final step refuses to commit to answers
2. **ToT precision** - Selecting too many nodes, adding latency and noise
3. **Content extraction** - The right section is found but the specific answer detail is lost
4. **Prompt engineering** - The synthesis prompt is too conservative

## Recommended Fixes (Priority Order)

### 🔴 Priority 1: Fix Answer Synthesis (CRITICAL)
**Problem:** System has the right content but won't answer.

**Solutions:**
1. **Rewrite synthesis prompt** to be more willing to infer/deduce answers
2. **Add confidence threshold tuning** - currently defaulting to "Cannot determine" too easily
3. **Test synthesis in isolation** - use the actual variables from failed questions
4. **Add explicit instruction:** "If the context contains information that allows inference, provide your best answer rather than saying 'Cannot determine'"

### 🔴 Priority 2: Switch to Semantic Search for Node Selection
**Problem:** ToT evaluation is slow (multiple LLM calls) and imprecise (selects 4 nodes when 1 is needed).

**Solution:** Replace ToT with semantic search (already implemented in `semantic_search.py`):
```python
# Current: ToT makes 2-3 expensive LLM calls to evaluate all children
# New: Semantic search uses vector embeddings (one-time cost)
searcher = SemanticSearcher(skeleton_nodes, kv_store)
relevant_nodes = searcher.search(query, top_k=3)  # Much faster
```

**Benefits:**
- ⚡ **10-20x faster** (no LLM calls for node evaluation)
- 🎯 **More accurate** (semantic similarity vs. LLM interpretation)
- 💰 **Cheaper** (eliminate 2-3 LLM calls per question)

### 🟡 Priority 3: Improve Variable Context Window
**Problem:** 1,957 chars of context contains the answer, but synthesis can't find it.

**Solutions:**
1. **Extract key sentences** when storing variables - not just full section
2. **Add section summary** alongside full content in variables
3. **Use smaller chunks** - maybe 512 chars max per variable

### 🟡 Priority 4: Add Semantic Search Fallback
**Problem:** 25% of questions get "Cannot determine".

**Solution:**
```python
# If top-k semantic search doesn't yield answer:
if answer == "Cannot determine":
    # Expand to top-20 nodes
    more_nodes = searcher.search(query, top_k=20)
    # Try again with broader context
```

## Comparison: Why Baseline Wins

The naive chunk baseline (55% EM) succeeds because:
1. **Simple retrieval** - No complex navigation, just semantic search over 512-char chunks
2. **Direct answers** - Doesn't overthink, just returns what's in top chunks
3. **Fast** - 9s per question vs RNSR's 53s

**Irony:** RNSR's sophisticated tree navigation and RLM decomposition are **hurting** performance because:
- ToT evaluation is slow and imprecise
- Variable stitching loses details
- Synthesis is over-conservative

## Next Steps

1. ✅ **Immediate:** Fix synthesis prompt (can be done in 30 min)
2. ✅ **Short-term:** Switch to semantic search for navigation (2-3 hours)
3. ⏳ **Medium-term:** Optimize variable context extraction (1 day)
4. ⏳ **Long-term:** Fine-tune ToT vs semantic search hybrid approach

## Key Takeaway

**The navigation is working - the problem is answer generation.**

RNSR is finding the right sections but:
1. Storing too much context (2000 chars when answer is 1 sentence)
2. Being too conservative in synthesis ("Cannot determine" when uncertain)
3. Taking 5-7x longer than necessary due to ToT evaluation overhead

**Quick Win:** Replace ToT with semantic search → expect 3-5x speedup and likely better accuracy.
