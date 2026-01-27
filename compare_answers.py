"""
Compare RNSR vs Baseline answers to see what's different.
"""

import json
from pathlib import Path

results_file = Path("benchmark_results/eval_report_20260126_121422.json")

with open(results_file) as f:
    data = json.load(f)

predictions = data["dataset_results"]["quality"]["predictions"]

# Look at a few examples where baseline succeeded but RNSR failed
print("=" * 100)
print("DETAILED ANSWER COMPARISON")
print("=" * 100)
print()

# Show questions where RNSR failed
failures = [p for p in predictions if p["answer"] in [
    "Cannot determine from available context.",
    "Cannot answer from available context.",
]]

# Show all RNSR answers vs what nodes were visited
print(f"Total Questions: {len(predictions)}")
print(f"RNSR gave 'Cannot determine': {len(failures)}")
print()

for i, pred in enumerate(predictions[:10]):  # First 10 questions
    print(f"\n{'='*100}")
    print(f"QUESTION {i}: {pred['question']}")
    print(f"{'='*100}")
    print(f"\nRNSR Answer: {pred['answer']}")
    print(f"\nNodes visited: {pred['nodes_visited']}")
    print(f"Time: {pred['time_s']:.1f}s")
    
    # Look at what sections were stored as variables
    trace = pred.get("trace", [])
    variables = [t for t in trace if t.get("node_type") == "variable_stitching"]
    
    if variables:
        print(f"\nVariables stored: {len(variables)}")
        for v in variables[:3]:  # Show first 3
            header = v.get("details", {}).get("header", "Unknown")
            chars = v.get("details", {}).get("chars", 0)
            print(f"  - {header}: {chars} chars")
    
    # Check synthesis
    synthesis = [t for t in trace if t.get("node_type") == "synthesis"]
    if synthesis:
        syn = synthesis[0]
        vars_used = syn.get("details", {}).get("variables_used", [])
        print(f"\nSynthesis used {len(vars_used)} variables")
    
    # Check if decomposition happened
    decomp = [t for t in trace if t.get("node_type") == "decomposition"]
    if decomp:
        d = decomp[0]
        sub_qs = d.get("details", {}).get("sub_questions", [])
        print(f"\nDecomposed into {len(sub_qs)} sub-questions:")
        for sq in sub_qs[:2]:
            print(f"  - {sq[:80]}...")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)
print()

# Count patterns
cannot_determine = len([p for p in predictions if "Cannot determine" in p["answer"]])
has_answer = len(predictions) - cannot_determine

print(f"Gave actual answer: {has_answer} ({has_answer/len(predictions)*100:.1f}%)")
print(f"Said 'Cannot determine': {cannot_determine} ({cannot_determine/len(predictions)*100:.1f}%)")
print()

# Check avg nodes visited for failures vs successes
failure_nodes = [len(p["nodes_visited"]) for p in predictions if "Cannot determine" in p["answer"]]
success_nodes = [len(p["nodes_visited"]) for p in predictions if "Cannot determine" not in p["answer"]]

if failure_nodes:
    print(f"Avg nodes for failures: {sum(failure_nodes)/len(failure_nodes):.1f}")
if success_nodes:
    print(f"Avg nodes for successes: {sum(success_nodes)/len(success_nodes):.1f}")
print()

print("KEY INSIGHT:")
print("Even when RNSR visits nodes and stores variables, it still says 'Cannot determine'")
print("This suggests:")
print("  1. Wrong nodes are being selected (ToT evaluation is off)")
print("  2. Right nodes are found but synthesis prompt is too conservative")
print("  3. Variable stitching is not providing enough context")
