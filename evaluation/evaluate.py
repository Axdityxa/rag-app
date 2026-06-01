import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import ask
from src.logger import setup_logger

logger = setup_logger("evaluation")


def load_eval_dataset(path: str = "evaluation/eval_dataset.json") -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} Q&A pair(s)")
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
    try:
        dataset = load_eval_dataset()

        if not dataset:
            logger.warning("Eval dataset is empty or missing")
            print("No evaluation dataset found or dataset is empty.")
            print("Fill in evaluation/eval_dataset.json with Q&A pairs and retry.")
            return

        has_expected = bool(dataset[0].get("expected_answer"))
        if not has_expected:
            logger.warning("No expected_answer fields in eval dataset")

        total = len(dataset)
        correct = 0

        for i, item in enumerate(dataset):
            question = item.get("question", "")
            expected = item.get("expected_answer", "")

            try:
                result = ask(question, mode="ask")
                answer = result.get("answer", "")

                if expected and has_expected:
                    passed = expected.lower() in answer.lower()
                    logger.info(f"[{i+1}/{total}] {'PASS' if passed else 'FAIL'}: {question[:60]}")
                    if passed:
                        correct += 1
                else:
                    logger.info(f"[{i+1}/{total}] {question[:60]}")

                print(f"[{i+1}/{total}] Q: {question}")
                print(f"   A: {answer[:200]}")
                print(f"   {result['latency']}s")
                if expected and has_expected:
                    print(f"   Expected: {expected[:200]}")
                print()

            except Exception as e:
                logger.error(f"[{i+1}/{total}] Failed: {e}", exc_info=True)
                print(f"[{i+1}/{total}] Q: {question}")
                print(f"   ERROR: {e}\n")

        if has_expected:
            accuracy = correct / total * 100
            logger.info(f"Done: {correct}/{total} correct ({accuracy:.1f}%)")
            print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        else:
            logger.info(f"Done: {total} questions processed")
            print(f"Evaluated {total} questions (no expected answers to compare).")

    except Exception as e:
        logger.error(f"Evaluation run failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_evaluation()
