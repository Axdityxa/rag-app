import zipfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import setup_logger

logger = setup_logger("code_ingest")

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".md", ".json", ".yaml", ".yml",
    ".sql", ".txt", ".toml", ".cfg", ".ini",
    ".html", ".css", ".scss",
}

IGNORE_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target",
    "chroma_db", "workspace", ".idea", ".vscode",
    "coverage", ".pytest_cache", ".mypy_cache",
    "eggs", ".eggs",
}

IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock",
}

MAX_FILE_BYTES = 500_000


def _should_skip_dir(name: str) -> bool:
    return name in IGNORE_DIRS


def _read_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            logger.warning(f"Skipping large file: {path}")
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning(f"Could not read {path}: {e}")
        return None


def load_codebase(repo_path: str) -> list[Document]:
    """Walk a directory and load supported source/doc files as LangChain documents."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    docs: list[Document] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS or _should_skip_dir(part) for part in path.parts):
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if path.name in IGNORE_FILES:
            continue

        content = _read_text_file(path)
        if content is None or not content.strip():
            continue

        rel = path.relative_to(root).as_posix()
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": rel,
                    "file_path": str(path),
                    "language": path.suffix.lstrip(".").lower(),
                    "type": "code",
                },
            )
        )

    logger.info(f"Loaded {len(docs)} file(s) from {root}")
    return docs


def extract_zip(zip_bytes: bytes, dest_dir: str) -> str:
    """Extract an uploaded zip to dest_dir; return the extraction root path."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "upload.zip"
    zip_path.write_bytes(zip_bytes)

    extract_to = dest / "repo"
    if extract_to.exists():
        import shutil
        shutil.rmtree(extract_to)
    extract_to.mkdir()

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    zip_path.unlink(missing_ok=True)

    children = [p for p in extract_to.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        root = children[0]
    else:
        root = extract_to

    logger.info(f"Extracted repo to {root}")
    return str(root)


LANG_SEPARATORS = {
    "py": ["\n\nclass ", "\n\ndef ", "\n\nasync def ", "\n\n", "\n", " ", ""],
    "js": ["\n\nexport ", "\n\nfunction ", "\n\nclass ", "\n\nconst ", "\n\n", "\n", " ", ""],
    "ts": ["\n\nexport ", "\n\nfunction ", "\n\nclass ", "\n\ninterface ", "\n\n", "\n", " ", ""],
    "tsx": ["\n\nexport ", "\n\nfunction ", "\n\nconst ", "\n\n", "\n", " ", ""],
    "md": ["\n## ", "\n### ", "\n\n", "\n", " ", ""],
}


def chunk_codebase(
    docs: list[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Structure-aware chunking per language, then prefix with file path."""
    if not docs:
        return []

    all_chunks: list[Document] = []
    by_lang: dict[str, list[Document]] = {}
    for doc in docs:
        lang = doc.metadata.get("language", "txt")
        by_lang.setdefault(lang, []).append(doc)

    for lang, lang_docs in by_lang.items():
        separators = LANG_SEPARATORS.get(lang, ["\n\n", "\n", " ", ""])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        all_chunks.extend(splitter.split_documents(lang_docs))

    chunks = all_chunks

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk.page_content = f"File: {source}\n\n{chunk.page_content}"

    logger.info(f"Created {len(chunks)} chunks from {len(docs)} file(s)")
    return chunks
