"""Debug tree structure for QuALITY."""

from rnsr.benchmarks.standard_benchmarks import BenchmarkLoader
from rnsr.ingestion.text_builder import build_tree_from_contexts
from rnsr.indexing import build_skeleton_index

# Load first quality question
dataset = BenchmarkLoader.load_quality(max_samples=1)
question = dataset.questions[0]

print(f"Article length: {len(question.context[0])} chars")
print(f"Question: {question.question}")
print()

# Build tree like the benchmark does
tree = build_tree_from_contexts(question.context)
print(f"Tree root level: {tree.root.level}")
print(f"Root children: {len(tree.root.children)}")
print(f"Total nodes: {tree.total_nodes}")
print()

# Check skeleton
skeleton, kv = build_skeleton_index(tree)
print(f"Skeleton nodes: {len(skeleton)}")
print(f"Root node children: {len(skeleton['root'].child_ids)}")
print()

# List root children
print("Root children:")
for i, cid in enumerate(list(skeleton['root'].child_ids)[:10]):
    node = skeleton[cid]
    print(f"  {i+1}. {cid}: {node.header[:50]} (level={node.level}, children={len(node.child_ids)})")
