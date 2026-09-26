"""
File validation utilities: extension/MIME checks, size checks, filename
sanitization, and path-traversal prevention.
"""

import os
import uuid
from werkzeug.utils import secure_filename

# Lightweight magic-byte signatures so we don't force a hard dependency on
# libmagic (which is not always available in minimal containers). This is a
# practical, not exhaustive, check that catches the common "renamed file"
# attack while staying dependency-free.
_SIGNATURES = {
    "pdf": [b"%PDF-"],
    "docx": [b"PK\x03\x04"],  # docx is a zip archive
}


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def sanitize_filename(filename: str) -> str:
    """Prevent path traversal and strip unsafe characters."""
    safe = secure_filename(filename)
    if not safe:
        safe = f"upload_{uuid.uuid4().hex[:8]}"
    return safe


def validate_upload(file_storage, allowed_extensions, max_size_bytes: int) -> str:
    """
    Validate a Werkzeug FileStorage object.

    Returns the sanitized, validated extension on success.
    Raises FileValidationError with a user-friendly message on failure.
    """
    if file_storage is None or file_storage.filename == "":
        raise FileValidationError("No file was selected.")

    original_name = file_storage.filename
    ext = get_extension(original_name)

    if ext not in allowed_extensions:
        raise FileValidationError(
            f"'.{ext or 'unknown'}' files are not supported. "
            f"Allowed types: {', '.join(sorted(allowed_extensions))}."
        )

    # Determine actual size by seeking to the end (safe for the werkzeug
    # SpooledTemporaryFile-backed stream used during multipart parsing).
    stream = file_storage.stream
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)

    if size == 0:
        raise FileValidationError("The uploaded file is empty.")

    if size > max_size_bytes:
        raise FileValidationError(
            f"File is too large ({size / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {max_size_bytes / (1024 * 1024):.0f} MB."
        )

    # Magic-byte sanity check where we have a known signature for the format.
    header = stream.read(8)
    stream.seek(0)
    signatures = _SIGNATURES.get(ext)
    if signatures and not any(header.startswith(sig) for sig in signatures):
        raise FileValidationError(
            f"The file does not look like a valid .{ext} file "
            "(it may be corrupted or mislabeled)."
        )

    return ext
