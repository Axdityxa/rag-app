# AI CTO

Architecture-aware engineering copilot: index your repo, ask questions, plan features, analyze impact, and trace flows — with hybrid retrieval, reranking, and grounded citations.

**Stack:** Streamlit · LangChain · ChromaDB · BM25 · Ollama

## Pipeline

```
Index repo → structure-aware chunking → embeddings → Chroma + BM25
User query → query rewrite (plan/impact) → hybrid retrieval → MMR → rerank top 5
→ mode prompt → LLM → guardrails + inline citations
```

## Setup

1. [Ollama](https://ollama.ai/download):
   ```bash
   ollama pull qwen2.5-coder:3b
   ollama pull nomic-embed-text
   ```

2. Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Default reranker is lightweight BM25 (no `torchvision` / `transformers` spam).

3. Copy `.env.example` → `.env` (optional).

## Usage

```bash
python -m streamlit run src/app.py
```

1. Enter repo path or upload zip → **Index / Re-index**
2. Pick mode → chat

## Evaluation

```bash
# Ensure codebase is indexed first (index this repo's src/ or root)
python -m evaluation.evaluate_cto
```

Metrics: **recall@k** (expected file in retrieved sources) and keyword checks.

## Project structure

```
src/
  app.py           Streamlit UI
  code_ingest.py   Scan, structure-aware chunking, zip extract
  bm25_store.py    BM25 corpus persistence
  retrieval.py     Hybrid + MMR + multi-query + rerank
  query.py         Pre-retrieval query rewrite
  guardrails.py    Empty-context refusal, citation warnings
  rag.py           CTO chains + ask()
  prompts.py       Mode prompts + citation rules
  vector_store.py  ChromaDB
  llm.py           Ollama chat model
evaluation/
  cto_eval.json    Eval questions for this project
  evaluate_cto.py  Recall@k runner
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_FETCH_K` | 20 | Candidates before rerank |
| `RETRIEVAL_TOP_K` | 5 | Chunks sent to LLM |
| `USE_RERANKER` | true | Rerank top candidates before LLM |
| `RERANKER_BACKEND` | bm25 | `bm25` (light) or `crossencoder` (needs sentence-transformers) |
| `RERANKER_MODEL` | ms-marco-MiniLM-L6-v2 | Only for `crossencoder` backend |
| `MAX_CONTEXT_CHARS` | 12000 | Context budget for small models |

## Troubleshooting

**Hundreds of `ModuleNotFoundError: torchvision` lines**

Streamlit’s file watcher was scanning the `transformers` package (from the old cross-encoder reranker). Fixes applied:

- `.streamlit/config.toml` sets `fileWatcherType = "none"`
- Default reranker is **BM25 on candidates** (`RERANKER_BACKEND=bm25`), not HuggingFace

For cross-encoder: `pip install sentence-transformers` and `RERANKER_BACKEND=crossencoder`.

