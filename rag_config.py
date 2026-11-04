"""Shared configuration for the PromptForge Week 3 RAG project."""

from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DOCUMENT_PATH = DATA_DIR / "source_document.txt"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EVALUATION_QUESTIONS_PATH = DATA_DIR / "evaluation_questions.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "northstar_student_support"

# Reuses the exact embedding model documented in the Week 2 README.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Reuses the Groq model/configuration documented in the Week 1 README.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_MIN_SIMILARITY = 0.25


def get_min_similarity() -> float:
    """Read and validate the configurable cosine-similarity threshold."""
    raw_value = os.getenv("RAG_MIN_SIMILARITY", str(DEFAULT_MIN_SIMILARITY))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("RAG_MIN_SIMILARITY must be a number between -1 and 1.") from exc
    if not -1.0 <= value <= 1.0:
        raise ValueError("RAG_MIN_SIMILARITY must be between -1 and 1.")
    return value
