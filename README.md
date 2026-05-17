# RAG App

Ask questions against your PDFs using local LLMs (Ollama).

## Setup

1. Install [Ollama](https://ollama.ai/download) and pull models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` (defaults work with local Ollama).

## Usage

1. Drop PDFs into `data/`
2. Launch the UI:
   ```bash
   python -m streamlit run src/app.py
   ```
3. Click "(Re)index documents" in the sidebar, then ask questions.

## Project structure

```
src/
  ingest.py        Load & chunk PDFs
  vector_store.py  Embeddings + ChromaDB
  rag.py           Retrieve + generate answers
  app.py           Streamlit UI
  logger.py        Colored logging
evaluation/
  evaluate.py      Test accuracy against Q&A dataset
```

## Logging

Every step is logged at DEBUG level with colored output (red for errors).
