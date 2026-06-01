"""Evaluate AI CTO retrieval quality (recall@k) and answer keywords."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import ask
from src.retrieval import retrieve_for_query, TOP_K
from src.query import rewrite_query
from src.logger import setup_logger

logger = setup_logger("evaluate_cto")


def load_eval_dataset(path: str = "evaluation/cto_eval.json") -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def source_hit(retrieved_sources: list[str], expected: list[str]) -> bool:
    for exp in expected:
        exp_norm = exp.replace("\\", "/")
        if any(exp_norm in src or src.endswith(exp_norm) for src in retrieved_sources):
            return True
    return False


def run_evaluation(dataset_path: str = "evaluation/cto_eval.json"):
    dataset = load_eval_dataset(dataset_path)
    if not dataset:
        print("Empty eval dataset.")
        return

    retrieval_hits = 0
    answer_hits = 0
    total = len(dataset)

    print(f"Running CTO eval on {total} questions (top_k={TOP_K})\n")

    for i, item in enumerate(dataset):
        question = item["question"]
        mode = item.get("mode", "ask")
        expected_sources = item.get("expected_sources", [])
        expected_keywords = item.get("expected_keywords", [])

        search_q = rewrite_query(question, mode)
        docs = retrieve_for_query(search_q, mode)
        retrieved = [d.metadata.get("source", "") for d in docs]

        r_hit = source_hit(retrieved, expected_sources) if expected_sources else None
        if r_hit:
            retrieval_hits += 1

        result = ask(question, mode=mode)
        answer = result["answer"].lower()
        if expected_keywords:
            a_hit = all(kw.lower() in answer for kw in expected_keywords)
            if a_hit:
                answer_hits += 1
        else:
            a_hit = None

        status_r = "PASS" if r_hit else ("FAIL" if r_hit is not None else "n/a")
        status_a = "PASS" if a_hit else ("FAIL" if a_hit is not None else "n/a")

        print(f"[{i + 1}/{total}] {status_r} retrieval | {status_a} answer")
        print(f"  Q: {question[:70]}...")
        print(f"  Retrieved: {retrieved[:3]}")
        if expected_sources:
            print(f"  Expected:  {expected_sources}")
        print(f"  A: {result['answer'][:120]}...")
        print()

    has_src = any(item.get("expected_sources") for item in dataset)
    has_kw = any(item.get("expected_keywords") for item in dataset)

    if has_src:
        recall = retrieval_hits / total * 100
        print(f"Retrieval recall@{TOP_K}: {retrieval_hits}/{total} ({recall:.1f}%)")
    if has_kw:
        kw_total = sum(1 for item in dataset if item.get("expected_keywords"))
        kw_acc = answer_hits / kw_total * 100 if kw_total else 0
        print(f"Answer keyword match: {answer_hits}/{kw_total} ({kw_acc:.1f}%)")


if __name__ == "__main__":
    run_evaluation()
