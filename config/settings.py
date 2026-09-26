"""
Centralized application configuration.

Every configurable value is read from environment variables (loaded from
.env by python-dotenv in app.py). Nothing here hard-codes a secret, a
credential, or a collection name.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val not in (None, "") else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val not in (None, "") else default
    except ValueError:
        return default


class Config:
    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-key-change-me")
    DEBUG = _get_bool("FLASK_DEBUG", False)
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = _get_int("FLASK_PORT", 5000)

    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    EMBEDDING_DIMENSIONS = _get_int("EMBEDDING_DIMENSIONS", 768)

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_knowledge_base")

    # Chunking / retrieval
    CHUNK_SIZE = _get_int("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 150)
    TOP_K = _get_int("TOP_K", 5)
    SIMILARITY_THRESHOLD = _get_float("SIMILARITY_THRESHOLD", 0.5)

    # Uploads
    MAX_UPLOAD_SIZE_MB = _get_int("MAX_UPLOAD_SIZE_MB", 20)
    MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    UPLOAD_FOLDER = str(BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads"))
    ALLOWED_EXTENSIONS = set(
        e.strip().lower()
        for e in os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt").split(",")
        if e.strip()
    )

    # Derived paths
    DATA_DIR = str(BASE_DIR / "data")
    DOCUMENT_REGISTRY_PATH = str(BASE_DIR / "data" / "documents.json")
    RAG_PROMPT_PATH = str(BASE_DIR / "config" / "rag_prompt.txt")
    LOG_DIR = str(BASE_DIR / "logs")

    @classmethod
    def validate(cls):
        """Return a list of human-readable configuration problems, if any."""
        problems = []
        if not cls.GEMINI_API_KEY:
            problems.append("GEMINI_API_KEY is not set.")
        if not cls.QDRANT_URL:
            problems.append("QDRANT_URL is not set.")
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return problems
