"""
Document ingestion pipeline:
  file -> validate -> extract text -> clean -> chunk -> embed -> store in Qdrant

Ingestion and question-answering logic are kept fully separate: this module
never talks to Gemini's generation endpoint, only its embedding endpoint
(via GeminiService.embed_documents), and it never performs retrieval.
"""

import logging
import os
from typing import List

from pypdf import PdfReader
from docx import Document as DocxDocument

from utils.chunking import chunk_text, clean_text, Chunk
from utils.file_validation import FileValidationError

logger = logging.getLogger(__name__)


class DocumentServiceError(Exception):
    pass


def extract_pdf(path: str) -> List[Chunk]:
    """Extract text per page so page numbers can be preserved in metadata."""
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise DocumentServiceError(f"Could not open PDF (it may be corrupted): {exc}")

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise DocumentServiceError("This PDF is password-protected and cannot be read.")

    per_page_text = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed extracting a PDF page: %s", exc)
            text = ""
        per_page_text.append(text)

    if not any(t.strip() for t in per_page_text):
        raise DocumentServiceError(
            "No extractable text was found in this PDF. It may be a scanned "
            "image without OCR, which this application does not support."
        )
    return per_page_text  # list[str], index 0 = page 1


def extract_docx(path: str) -> str:
    try:
        doc = DocxDocument(path)
    except Exception as exc:
        raise DocumentServiceError(f"Could not open DOCX (it may be corrupted): {exc}")

    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text and cell.text.strip():
                    parts.append(cell.text)

    text = "\n\n".join(parts)
    if not text.strip():
        raise DocumentServiceError("No extractable text was found in this DOCX file.")
    return text


def extract_txt(path: str) -> str:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_exc = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            if not text.strip():
                raise DocumentServiceError("The uploaded .txt file is empty.")
            return text
        except DocumentServiceError:
            raise
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    raise DocumentServiceError(f"Could not decode text file: {last_exc}")


class DocumentService:
    def __init__(self, gemini_service, qdrant_service, registry,
                 chunk_size: int, chunk_overlap: int):
        self.gemini = gemini_service
        self.qdrant = qdrant_service
        self.registry = registry
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, document_id: str, file_path: str, ext: str, filename: str):
        """
        Run the full ingestion pipeline for an already-saved, already-validated
        file. Updates the registry's status as it progresses so the UI can
        poll for Uploading -> Extracting -> Chunking -> Embedding -> Indexing
        -> Completed (or Failed with a message).
        """
        try:
            self.registry.update(document_id, status="extracting")
            chunks: List[Chunk] = []

            if ext == "pdf":
                pages = extract_pdf(file_path)
                for page_num, page_text in enumerate(pages, start=1):
                    page_chunks = chunk_text(
                        page_text, self.chunk_size, self.chunk_overlap, page_number=page_num
                    )
                    chunks.extend(page_chunks)
            elif ext == "docx":
                text = extract_docx(file_path)
                chunks = chunk_text(clean_text(text), self.chunk_size, self.chunk_overlap)
            elif ext == "txt":
                text = extract_txt(file_path)
                chunks = chunk_text(clean_text(text), self.chunk_size, self.chunk_overlap)
            else:
                raise DocumentServiceError(f"Unsupported file type: {ext}")

            # Re-index chunk_index sequentially across the whole document
            for i, c in enumerate(chunks):
                c.chunk_index = i

            if not chunks:
                raise DocumentServiceError("No usable text chunks were produced from this document.")

            self.registry.update(document_id, status="chunking", chunk_count=len(chunks))

            self.registry.update(document_id, status="embedding")
            texts = [c.text for c in chunks]
            vectors = self.gemini.embed_documents(texts)

            self.registry.update(document_id, status="indexing")
            payload_chunks = [
                {
                    "chunk_id": f"{document_id}:{c.chunk_index}",
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "text": c.text,
                }
                for c in chunks
            ]
            self.qdrant.upsert_chunks(document_id, filename, payload_chunks, vectors)

            self.registry.update(document_id, status="completed", chunk_count=len(chunks), error=None)
            logger.info("Document %s (%s) indexed with %d chunks", document_id, filename, len(chunks))

        except (DocumentServiceError, FileValidationError) as exc:
            logger.warning("Ingestion failed for %s: %s", filename, exc)
            self.registry.update(document_id, status="failed", error=str(exc))
            raise
        except Exception as exc:
            logger.error("Unexpected ingestion failure for %s: %s", filename, exc)
            self.registry.update(document_id, status="failed", error="An unexpected error occurred while processing this document.")
            raise DocumentServiceError("An unexpected error occurred while processing this document.") from exc

    def delete(self, document_id: str, stored_path: str = None):
        self.qdrant.delete_document(document_id)
        if stored_path and os.path.exists(stored_path):
            try:
                os.remove(stored_path)
            except OSError as exc:
                logger.warning("Could not remove stored file %s: %s", stored_path, exc)
        self.registry.delete(document_id)
