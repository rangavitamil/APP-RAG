"""Small shared helpers: JSON API responses, ids, timestamps."""

import uuid
from datetime import datetime, timezone

from flask import jsonify


def new_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ok(data=None, status: int = 200):
    return jsonify({"success": True, "data": data if data is not None else {}}), status


def err(message: str, code: str = "ERROR", status: int = 400):
    return jsonify({"success": False, "error": {"code": code, "message": message}}), status
