"""
RAG Application - Flask entry point.

Endpoints:
  GET  /                      -> dashboard UI
  GET  /api/health            -> system status (Gemini, Qdrant, KB stats)
  POST /api/documents         -> upload + ingest a document
  GET  /api/documents         -> list documents
  DELETE /api/documents/<id>  -> delete a document + its vectors
  POST /api/chat              -> ask a question against the knowledge base
"""

import logging
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from utils.helpers import ok, err, new_id, now_iso
from utils.file_validation import validate_upload, sanitize_filename, FileValidationError
from services.document_registry import DocumentRegistry
from services.gemini_service import GeminiService, GeminiServiceError
from services.qdrant_service import QdrantService, QdrantServiceError
from services.document_service import DocumentService, DocumentServiceError
from services.rag_service import RagService, RagServiceError

# ---------------------------------------------------------------------------
# Logging (never logs secrets)
# ---------------------------------------------------------------------------
os.makedirs(Config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("rag_app")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = Config.SECRET_KEY

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.DATA_DIR, exist_ok=True)

registry = DocumentRegistry(Config.DOCUMENT_REGISTRY_PATH)

_config_problems = Config.validate()
for p in _config_problems:
    logger.warning("Configuration warning: %s", p)

gemini_service = None
qdrant_service = None
document_service = None
rag_service = None
_init_error = None

try:
    gemini_service = GeminiService(
        api_key=Config.GEMINI_API_KEY,
        generation_model=Config.GEMINI_MODEL,
        embedding_model=Config.GEMINI_EMBEDDING_MODEL,
        embedding_dimensions=Config.EMBEDDING_DIMENSIONS,
    )
    qdrant_service = QdrantService(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=Config.QDRANT_COLLECTION_NAME,
        vector_size=Config.EMBEDDING_DIMENSIONS,
    )
    qdrant_service.ensure_collection()

    document_service = DocumentService(
        gemini_service, qdrant_service, registry,
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP,
    )

    with open(Config.RAG_PROMPT_PATH, "r", encoding="utf-8") as f:
        _system_prompt = f.read()

    rag_service = RagService(
        gemini_service, qdrant_service, _system_prompt,
        top_k=Config.TOP_K, similarity_threshold=Config.SIMILARITY_THRESHOLD,
    )
except (GeminiServiceError, QdrantServiceError) as exc:
    _init_error = str(exc)
    logger.error("Service initialization failed: %s", exc)


# ---------------------------------------------------------------------------
# Routes: UI
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes: health
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    gemini_ok = gemini_service.health_check() if gemini_service else False
    qdrant_ok = qdrant_service.health_check() if qdrant_service else False
    stats = registry.stats()
    return ok({
        "gemini": {"configured": bool(Config.GEMINI_API_KEY), "connected": gemini_ok,
                   "model": Config.GEMINI_MODEL},
        "qdrant": {"connected": qdrant_ok, "collection": Config.QDRANT_COLLECTION_NAME},
        "knowledge_base": stats,
        "config_warnings": _config_problems,
        "init_error": _init_error,
    })


def _require_services():
    if not (gemini_service and qdrant_service and document_service and rag_service):
        return err(
            _init_error or "Application services are not configured. Check server logs and .env.",
            code="SERVICE_UNAVAILABLE", status=503,
        )
    return None


# ---------------------------------------------------------------------------
# Routes: documents
# ---------------------------------------------------------------------------
@app.route("/api/documents", methods=["GET"])
def list_documents():
    return ok({"documents": registry.list_all()})


@app.route("/api/documents", methods=["POST"])
def upload_document():
    guard = _require_services()
    if guard:
        return guard

    file_storage = request.files.get("file")
    try:
        ext = validate_upload(file_storage, Config.ALLOWED_EXTENSIONS, Config.MAX_CONTENT_LENGTH)
    except FileValidationError as exc:
        return err(str(exc), code="INVALID_FILE", status=400)

    original_filename = file_storage.filename
    safe_name = sanitize_filename(original_filename)
    document_id = new_id()
    stored_name = f"{document_id}_{safe_name}"
    stored_path = os.path.join(Config.UPLOAD_FOLDER, stored_name)

    # Prevent path traversal: confirm the resolved path stays inside UPLOAD_FOLDER
    upload_root = os.path.realpath(Config.UPLOAD_FOLDER)
    resolved_path = os.path.realpath(stored_path)
    if not resolved_path.startswith(upload_root + os.sep):
        return err("Invalid filename.", code="INVALID_FILE", status=400)

    file_storage.save(stored_path)
    size_bytes = os.path.getsize(stored_path)

    registry.create({
        "id": document_id,
        "filename": original_filename,
        "file_type": ext,
        "size_bytes": size_bytes,
        "uploaded_at": now_iso(),
        "status": "uploading",
        "chunk_count": 0,
        "error": None,
        "stored_path": stored_path,
    })

    try:
        document_service.process(document_id, stored_path, ext, original_filename)
    except (DocumentServiceError, GeminiServiceError, QdrantServiceError) as exc:
        # status/error already recorded on the registry by document_service
        return err(str(exc), code="INGESTION_FAILED", status=422)
    except Exception:
        logger.error("Unhandled ingestion error:\n%s", traceback.format_exc())
        registry.update(document_id, status="failed", error="Unexpected server error during processing.")
        return err("Unexpected server error during processing.", code="INTERNAL_ERROR", status=500)

    return ok({"document": registry.get(document_id)}, status=201)


@app.route("/api/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id):
    guard = _require_services()
    if guard:
        return guard

    doc = registry.get(document_id)
    if not doc:
        return err("Document not found.", code="NOT_FOUND", status=404)

    try:
        document_service.delete(document_id, stored_path=doc.get("stored_path"))
    except QdrantServiceError as exc:
        return err(str(exc), code="DELETE_FAILED", status=502)

    return ok({"deleted": document_id})


# ---------------------------------------------------------------------------
# Routes: chat
# ---------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    guard = _require_services()
    if guard:
        return guard

    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    history = body.get("history", [])
    if not isinstance(history, list):
        history = []
    # keep history bounded and well-shaped
    history = [
        {"role": t.get("role"), "text": t.get("text", "")}
        for t in history[-10:]
        if isinstance(t, dict) and t.get("role") in ("user", "assistant", "model")
    ]
    for t in history:
        t["role"] = "user" if t["role"] == "user" else "model"

    try:
        result = rag_service.answer(question, history=history)
    except RagServiceError as exc:
        return err(str(exc), code="INVALID_QUESTION", status=400)
    except (GeminiServiceError, QdrantServiceError) as exc:
        return err(str(exc), code="RAG_FAILED", status=502)
    except Exception:
        logger.error("Unhandled chat error:\n%s", traceback.format_exc())
        return err("Unexpected server error while answering.", code="INTERNAL_ERROR", status=500)

    return ok(result)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_e):
    return err(
        f"File exceeds the maximum upload size of {Config.MAX_UPLOAD_SIZE_MB} MB.",
        code="FILE_TOO_LARGE", status=413,
    )


@app.errorhandler(404)
def not_found(_e):
    return err("Not found.", code="NOT_FOUND", status=404)


@app.errorhandler(500)
def internal_error(e):
    logger.error("Unhandled server error: %s", e)
    return err("Internal server error.", code="INTERNAL_ERROR", status=500)


if __name__ == "__main__":
    if _init_error:
        logger.warning("Starting with degraded services: %s", _init_error)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
