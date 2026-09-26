"""
Local document registry.

Qdrant stores per-chunk vectors + metadata, but the UI needs document-level
bookkeeping (filename, status, chunk count, upload date, errors) that is
awkward to derive from a vector store on every request. This module keeps
that small registry in a JSON file on disk, guarded by a lock for the
single-process Flask dev/production (threaded) server.
"""

import json
import os
import threading
from typing import Dict, List, Optional

_lock = threading.Lock()


class DocumentRegistry:
    STATUSES = (
        "uploading",
        "extracting",
        "chunking",
        "embedding",
        "indexing",
        "completed",
        "failed",
    )

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._write({})

    def _read(self) -> Dict[str, dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: Dict[str, dict]):
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.path)

    def create(self, doc: dict):
        with _lock:
            data = self._read()
            data[doc["id"]] = doc
            self._write(data)

    def update(self, doc_id: str, **fields):
        with _lock:
            data = self._read()
            if doc_id in data:
                data[doc_id].update(fields)
                self._write(data)

    def get(self, doc_id: str) -> Optional[dict]:
        with _lock:
            return self._read().get(doc_id)

    def delete(self, doc_id: str):
        with _lock:
            data = self._read()
            data.pop(doc_id, None)
            self._write(data)

    def list_all(self) -> List[dict]:
        with _lock:
            data = self._read()
            return sorted(data.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True)

    def stats(self) -> dict:
        docs = self.list_all()
        return {
            "document_count": len(docs),
            "chunk_count": sum(d.get("chunk_count", 0) or 0 for d in docs),
        }
