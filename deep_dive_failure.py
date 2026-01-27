"""
Deep dive into specific failure case to understand why wrong nodes were selected.
"""

import json
from pathlib import Path

results_file = Path("benchmark_results/eval_report_20260126_121422.json")

with open(results_file) as f:
    data = json.load(f)

predictions = data["dataset_results"]["quality"]["predictions"]

# Look at Question 5: "Why doesn't Blake haggle with Eldoria about the price?"
# RNSR said "Cannot determine" but visited: seg_f20e93f8, seg_20dc79e1, seg_004f769c, seg_26467941

question_5 = predictions[5]

print("=" * 100)
print("DEEP DIVE: Question 5")
print("=" * 100)
print()
print(f"Question: {question_5['question']}")
print(f"RNSR Answer: {question_5['answer']}")
print()

# Get the trace
trace = question_5["trace"]

# Find ToT evaluation
tot_evals = [t for t in trace if "Queued" in str(t.get("action", ""))]
if tot_evals:
    print("ToT EVALUATION RESULTS:")
    print()
    first_tot = tot_evals[0]
    nodes_selected = first_tot.get("details", {}).get("nodes", [])
    print(f"ToT selected {len(nodes_selected)} nodes:")
    for node in nodes_selected:
        print(f"  - {node}")
    print()

# Check what was in the variables
print("VARIABLES STORED:")
print()
var_events = [t for t in trace if t.get("node_type") == "variable_stitching"]
for v in var_events:
    details = v.get("details", {})
    print(f"{details.get('header', 'Unknown')}: {details.get('chars', 0)} chars")
print()

# Check the synthesis
synthesis = [t for t in trace if t.get("node_type") == "synthesis"]
if synthesis:
    syn = synthesis[0]
    details = syn.get("details", {})
    print(f"SYNTHESIS:")
    print(f"  Variables used: {details.get('variables_used', [])}")
    print(f"  Confidence: {details.get('confidence', 'unknown')}")
print()

# Show navigation path
print("NAVIGATION PATH:")
nav_events = [t for t in trace if t.get("node_type") == "navigation" and "Visiting" in str(t.get("action", ""))]
for i, nav in enumerate(nav_events[:10]):
    action = nav.get("action", "")
    node = action.split(":")[-1].strip() if ":" in action else "unknown"
    print(f"  {i+1}. {node}")
print()

print("=" * 100)
print("ANALYSIS")
print("=" * 100)
print()

print("The question asks: 'Why doesn't Blake haggle with Eldoria about the price?'")
print()
print("This is about Blake's character motivation - he sees Eldoria as valuable/special")
print("and doesn't want to cheapen the transaction by haggling.")
print()
print("The answer should be in Section 3 where the price negotiation happens.")
print("RNSR DID visit seg_f20e93f8 (Section 3) - so it had the content!")
print()
print("But it still said 'Cannot determine' - why?")
print()
print("Possible reasons:")
print("1. The section summary didn't emphasize Blake's motivation")
print("2. The synthesis prompt is too conservative")
print("3. Variable stitching provides context but loses the specific detail")
print("4. ToT selected too many irrelevant nodes, diluting the relevant content")
