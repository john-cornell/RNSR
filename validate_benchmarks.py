#!/usr/bin/env python3
"""
Quick validation of benchmark loaders and DocVQA vision integration.

Tests each benchmark with 1 sample to confirm:
1. Dataset loading works
2. Questions are properly parsed
3. RNSR can answer from context
4. DocVQA image bytes are captured and stored in KV store
"""
from __future__ import annotations

import sys
import time
import traceback


def validate_loader(name: str, load_fn, max_samples: int = 1) -> bool:
    """Validate a benchmark loader produces usable questions."""
    print(f"\n{'='*60}")
    print(f"  Validating: {name}")
    print(f"{'='*60}")

    try:
        dataset = load_fn(max_samples=max_samples)
        print(f"  Loaded {len(dataset.questions)} question(s)")

        if not dataset.questions:
            print(f"  FAIL: No questions loaded!")
            return False

        q = dataset.questions[0]
        print(f"  Question: {q.question[:80]}...")
        print(f"  Answer:   {q.answer[:80]}...")
        print(f"  Context:  {len(q.context)} chunk(s)")
        print(f"  Metadata: {list(q.metadata.keys())}")

        if q.metadata.get("has_image"):
            img_bytes = q.metadata.get("image_bytes")
            print(f"  Image:    {'YES' if img_bytes else 'NO'} "
                  f"({len(img_bytes)} bytes)" if img_bytes else "")

        print(f"  OK")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def validate_rnsr_answer(name: str, dataset, use_vision: bool = False) -> bool:
    """Validate RNSR can answer a question from this dataset."""
    from rnsr.benchmarks.evaluation_suite import RNSRBenchmarkAdapter

    print(f"\n  Running RNSR on {name}...")

    adapter = RNSRBenchmarkAdapter()
    q = dataset.questions[0]

    try:
        start = time.perf_counter()
        result = adapter.answer_from_context(
            question=q.question,
            contexts=q.context,
            metadata=q.metadata,
        )
        elapsed = time.perf_counter() - start

        print(f"  RNSR Answer: {result.answer[:100]}...")
        print(f"  Expected:    {q.answer[:100]}...")
        print(f"  Time:        {elapsed:.1f}s")
        print(f"  Nodes:       {len(result.nodes_visited)}")

        # Check vision trace entries if this was a vision benchmark
        if use_vision:
            vision_entries = [
                t for t in result.trace
                if t.get("action_type") == "vision_analysis"
            ]
            print(f"  Vision:      {len(vision_entries)} analysis entries in trace")
            if vision_entries:
                print(f"  VISION INTEGRATION WORKING!")
            else:
                print(f"  WARNING: No vision trace entries (may still work if no leaf was expanded)")

        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def main():
    results: dict[str, bool] = {}

    # 1. MultiHiertt
    from rnsr.benchmarks.multihiertt_bench import MultiHierttLoader
    ok = validate_loader("MultiHiertt", MultiHierttLoader.load)
    results["multihiertt_load"] = ok
    if ok:
        ds = MultiHierttLoader.load(max_samples=1)
        results["multihiertt_answer"] = validate_rnsr_answer("MultiHiertt", ds)

    # 2. TAT-QA
    from rnsr.benchmarks.tatqa_bench import TATQALoader
    ok = validate_loader("TAT-QA", TATQALoader.load)
    results["tatqa_load"] = ok
    if ok:
        ds = TATQALoader.load(max_samples=1)
        results["tatqa_answer"] = validate_rnsr_answer("TAT-QA", ds)

    # 3. QASPER
    from rnsr.benchmarks.qasper_bench import QASPERLoader
    ok = validate_loader("QASPER", QASPERLoader.load)
    results["qasper_load"] = ok
    if ok:
        ds = QASPERLoader.load(max_samples=1)
        results["qasper_answer"] = validate_rnsr_answer("QASPER", ds)

    # 4. DocVQA (with vision)
    from rnsr.benchmarks.docvqa_bench import DocVQALoader
    ok = validate_loader("DocVQA", DocVQALoader.load)
    results["docvqa_load"] = ok
    if ok:
        ds = DocVQALoader.load(max_samples=1)
        # Check image bytes are captured
        q = ds.questions[0]
        img_bytes = q.metadata.get("image_bytes")
        results["docvqa_image_captured"] = img_bytes is not None
        if img_bytes:
            print(f"  Image bytes captured: {len(img_bytes)} bytes")
        else:
            print(f"  WARNING: No image bytes captured!")

        results["docvqa_answer"] = validate_rnsr_answer("DocVQA", ds, use_vision=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test:<30} {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")
    print(f"{'='*60}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
