import os

from langchain_ollama import ChatOllama

from src.logger import setup_logger

logger = setup_logger("llm")


def get_llm():
    try:
        model = os.getenv("LLM_MODEL", "qwen2.5-coder:3b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0)
    except Exception as e:
        logger.error(f"Failed to initialise LLM: {e}", exc_info=True)
        raise
