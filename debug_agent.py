"""Debug agent execution step by step."""

from rnsr.benchmarks.standard_benchmarks import BenchmarkLoader
from rnsr.ingestion.text_builder import build_tree_from_contexts
from rnsr.indexing import build_skeleton_index
from rnsr.agent import run_navigator

# Load first quality question
dataset = BenchmarkLoader.load_quality(max_samples=1)
question_obj = dataset.questions[0]

print(f"Question: {question_obj.question}")
print()

# Build tree
tree = build_tree_from_contexts(question_obj.context)
skeleton, kv = build_skeleton_index(tree)

print(f"Tree structure: {len(skeleton)} nodes, {len(skeleton['root'].child_ids)} root children")
print()

# Run navigator
result = run_navigator(
    question=question_obj.question,
    skeleton=skeleton,
    kv_store=kv,
    max_iterations=20,
)

print("=== RESULT ===")
print(f"Answer: {result['answer'][:200]}")
print(f"Confidence: {result['confidence']}")
print(f"Nodes visited: {result.get('nodes_visited', [])}")
print(f"Variables used: {result.get('variables_used', [])}")
print(f"Trace entries: {len(result.get('trace', []))}")
print()

# Show trace
print("=== TRACE ===")
for i, entry in enumerate(result.get('trace', [])[:10]):
    print(f"{i+1}. {entry.get('node_type')}: {entry.get('action')}")
    if entry.get('details'):
        print(f"   Details: {entry.get('details')}")
