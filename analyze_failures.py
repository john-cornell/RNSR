"""
Analyze RNSR benchmark failures to understand navigation issues.

This script:
1. Loads the benchmark results
2. Compares RNSR vs baseline answers
3. Identifies navigation problems
4. Provides actionable insights
"""

import json
from pathlib import Path
from typing import Any

# Load the latest benchmark results
results_file = Path("benchmark_results/eval_report_20260126_121422.json")

with open(results_file) as f:
    data = json.load(f)

dataset_results = data["dataset_results"]["quality"]
predictions = dataset_results["predictions"]

print("=" * 80)
print("RNSR BENCHMARK FAILURE ANALYSIS")
print("=" * 80)
print()

# Analyze each prediction
failures = []
for i, pred in enumerate(predictions):
    question = pred["question"]
    rnsr_answer = pred["answer"]
    nodes_visited = pred["nodes_visited"]
    time_taken = pred["time_s"]
    
    # Check if RNSR failed
    is_failure = rnsr_answer in [
        "Cannot determine from available context.",
        "Cannot answer from available context.",
        "Unable to determine from context.",
        "",
    ]
    
    if is_failure:
        failures.append({
            "id": i,
            "question": question,
            "answer": rnsr_answer,
            "nodes_visited": len(nodes_visited),
            "time_s": time_taken,
            "trace": pred.get("trace", []),
        })

print(f"Total Questions: {len(predictions)}")
print(f"RNSR Failures: {len(failures)} ({len(failures)/len(predictions)*100:.1f}%)")
print(f"RNSR answer_em: {dataset_results['rnsr_metrics']['answer_em']:.3f}")
print(f"RNSR answer_f1: {dataset_results['rnsr_metrics']['answer_f1']:.3f}")
print(f"Baseline answer_em: {dataset_results['baseline_metrics']['naive_chunk_512']['answer_em']:.3f}")
print(f"Baseline answer_f1: {dataset_results['baseline_metrics']['naive_chunk_512']['answer_f1']:.3f}")
print()

# Analyze navigation patterns
print("=" * 80)
print("NAVIGATION PATTERN ANALYSIS")
print("=" * 80)
print()

total_nodes_visited = sum(len(p["nodes_visited"]) for p in predictions)
avg_nodes = total_nodes_visited / len(predictions)

print(f"Average nodes visited per question: {avg_nodes:.1f}")
print(f"Average time per question: {dataset_results['rnsr_metrics']['mean_time_s']:.1f}s")
print()

# Check for ToT evaluation issues
tot_issues = 0
json_repair_issues = 0
dead_end_issues = 0

for pred in predictions:
    trace = pred.get("trace", [])
    
    # Count ToT evaluations
    tot_evals = [t for t in trace if "tot_evaluation_complete" in str(t.get("action", ""))]
    if len(tot_evals) > 3:  # More than 3 ToT rounds seems excessive
        tot_issues += 1
    
    # Check for JSON repair attempts (indicates LLM returning bad format)
    for t in trace:
        if "json_repair" in str(t.get("action", "")).lower():
            json_repair_issues += 1
            break
    
    # Check for dead ends
    for t in trace:
        details = t.get("details", {})
        if details.get("is_dead_end"):
            dead_end_issues += 1
            break

print(f"Questions with excessive ToT evaluations (>3): {tot_issues}")
print(f"Questions with JSON repair attempts: {json_repair_issues}")
print(f"Questions hitting dead ends: {dead_end_issues}")
print()

# Show sample failures
print("=" * 80)
print("SAMPLE FAILURE CASES")
print("=" * 80)
print()

for i, failure in enumerate(failures[:5]):  # Show first 5
    print(f"\n--- Failure {i+1} ---")
    print(f"Question: {failure['question'][:80]}...")
    print(f"RNSR Answer: {failure['answer']}")
    print(f"Nodes visited: {failure['nodes_visited']}")
    print(f"Time taken: {failure['time_s']:.1f}s")
    
    # Check what was explored
    trace = failure.get("trace", [])
    nav_events = [t for t in trace if t.get("node_type") == "navigation"]
    
    if nav_events:
        print(f"Navigation events: {len(nav_events)}")
        
        # Find ToT evaluation results
        tot_evals = [t for t in trace if "Queued" in str(t.get("action", ""))]
        if tot_evals:
            print(f"ToT selections made: {len(tot_evals)}")
            # Show first ToT selection
            first_tot = tot_evals[0]
            nodes_selected = first_tot.get("details", {}).get("nodes", [])
            print(f"First ToT selected {len(nodes_selected)} nodes")

print()
print("=" * 80)
print("KEY ISSUES IDENTIFIED")
print("=" * 80)
print()

issues = []

if len(failures) > len(predictions) * 0.5:
    issues.append("❌ CRITICAL: Over 50% of questions return 'Cannot determine from context'")
    issues.append("   This suggests ToT is not finding relevant content effectively")

if avg_nodes > 15:
    issues.append("❌ CRITICAL: Visiting too many nodes per question (avg {:.1f})".format(avg_nodes))
    issues.append("   ToT should be selective, but it's exploring too broadly")

if json_repair_issues > 5:
    issues.append(f"⚠️  WARNING: {json_repair_issues} questions had JSON repair attempts")
    issues.append("   LLM is returning malformed/irrelevant responses")

if tot_issues > 10:
    issues.append(f"⚠️  WARNING: {tot_issues} questions had excessive ToT rounds (>3)")
    issues.append("   ToT is running multiple evaluations without converging")

if dataset_results['rnsr_metrics']['answer_em'] == 0:
    issues.append("❌ CRITICAL: 0% exact match - RNSR is not answering correctly")
    issues.append("   Even when content is found, answer generation is failing")

if issues:
    for issue in issues:
        print(issue)
else:
    print("✓ No critical issues detected")

print()
print("=" * 80)
print("RECOMMENDED FIXES")
print("=" * 80)
print()

print("1. SWITCH TO SEMANTIC SEARCH")
print("   - Use vector embeddings to find relevant nodes (O(log N))")
print("   - Replace expensive LLM-based ToT evaluation")
print("   - Will be faster and more accurate")
print()

print("2. FIX ANSWER GENERATION")
print("   - Even when content is found, answers are wrong")
print("   - Check if variable stitching is providing too much/little context")
print("   - Verify final synthesis prompt is effective")
print()

print("3. ADD FALLBACK STRATEGY")
print("   - If top-k semantic search doesn't find answer, expand search")
print("   - Stop returning 'Cannot determine' so frequently")
print()

print("4. IMPROVE ToT PROMPT")
print("   - Current ToT evaluations are returning irrelevant content")
print("   - Add better examples/constraints to the prompt")
print("   - Consider using smaller k in top-k selection")
print()

print("=" * 80)
