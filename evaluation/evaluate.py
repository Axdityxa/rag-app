import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import build_rag_chain, ask
from src.logger import setup_logger

logger = setup_logger("evaluation")


def load_eval_dataset(path: str = "evaluation/eval_dataset.json") -> list:
    logger.info(f"load_eval_dataset called with path='{path}'")
    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} Q&A pair(s) from '{path}'")
        for i, item in enumerate(dataset):
            logger.debug(f"  Item {i}: question='{item.get('question', '')}', "
                         f"expected='{str(item.get('expected_answer', ''))[:80]}...'")
        return dataset
    except FileNotFoundError:
        logger.error(f"Eval dataset file not found at '{path}'")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in '{path}': {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Failed to load eval dataset: {e}", exc_info=True)
        return []


def run_evaluation():
    logger.info("run_evaluation started")
    try:
        logger.debug("Building RAG chain ...")
        chain = build_rag_chain()
        logger.debug("Loading evaluation dataset ...")
        dataset = load_eval_dataset()

        if not dataset:
            logger.warning("Eval dataset is empty or missing. Nothing to evaluate.")
            print("No evaluation dataset found or dataset is empty.")
            print("Fill in evaluation/eval_dataset.json with Q&A pairs and retry.")
            return

        has_expected = bool(dataset[0].get("expected_answer"))
        if not has_expected:
            logger.warning("No expected_answer fields found in eval dataset. Running without comparison.")
            print("No expected answers defined in eval_dataset.json.")
            print("Running in manual mode (no pass/fail).")

        total = len(dataset)
        correct = 0

        for i, item in enumerate(dataset):
            question = item.get("question", "")
            expected = item.get("expected_answer", "")

            logger.info(f"[{i+1}/{total}] Evaluating: '{question}'")
            try:
                result = ask(question, chain)
                answer = result.get("answer", "")

                logger.debug(f"[{i+1}/{total}] Answer: '{answer[:200]}...'")
                logger.debug(f"[{i+1}/{total}] Sources: {len(result.get('sources', []))}, "
                             f"Latency: {result.get('latency', 0)}s")

                if expected and has_expected:
                    passed = expected.lower() in answer.lower()
                    logger.info(f"[{i+1}/{total}] Expected: '{expected[:100]}...' -> "
                                f"{'PASS' if passed else 'FAIL'}")
                    if passed:
                        correct += 1

                print(f"[{i+1}/{total}] Q: {question}")
                print(f"   A: {answer[:200]}")
                print(f"   {result['latency']}s")
                if expected and has_expected:
                    print(f"   Expected: {expected[:200]}")
                print()

            except Exception as e:
                logger.error(f"[{i+1}/{total}] Evaluation failed for '{question}': {e}", exc_info=True)
                print(f"[{i+1}/{total}] Q: {question}")
                print(f"   ERROR: {e}\n")

        if has_expected:
            accuracy = correct / total * 100
            logger.info(f"Evaluation complete: {correct}/{total} correct ({accuracy:.1f}%)")
            print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        else:
            logger.info(f"Evaluation complete: {total} questions processed")
            print(f"Evaluated {total} questions (no expected answers to compare).")

    except Exception as e:
        logger.error(f"Evaluation run failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_evaluation()
