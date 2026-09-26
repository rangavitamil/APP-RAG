"""
Configurable text chunking.

Splits cleaned text into overlapping chunks along natural boundaries
(paragraph -> sentence -> word) instead of cutting at arbitrary character
positions, while respecting a target chunk_size and chunk_overlap.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    char_start: int = 0
    char_end: int = 0


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sentences(paragraph: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s for s in sentences if s]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    page_number: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk a block of text (optionally scoped to a single page) into
    Chunk objects of roughly chunk_size characters, with chunk_overlap
    characters of overlap between consecutive chunks.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    units: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_into_sentences(para))

    chunks: List[Chunk] = []
    current = ""
    idx = 0
    cursor = 0  # approximate character offset into `text`

    def flush(buf: str, start: int):
        nonlocal idx
        buf = buf.strip()
        if buf:
            chunks.append(
                Chunk(
                    text=buf,
                    chunk_index=idx,
                    page_number=page_number,
                    char_start=start,
                    char_end=start + len(buf),
                )
            )
            idx += 1

    start_offset = 0
    for unit in units:
        candidate = (current + " " + unit).strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                flush(current, start_offset)
                # build overlap: keep the tail of `current` for context continuity
                if chunk_overlap > 0:
                    overlap_text = current[-chunk_overlap:]
                    start_offset = start_offset + max(0, len(current) - len(overlap_text))
                    current = (overlap_text + " " + unit).strip()
                else:
                    start_offset = start_offset + len(current)
                    current = unit
            else:
                # single unit longer than chunk_size: hard-split on words
                words = unit.split(" ")
                buf = ""
                for w in words:
                    if len(buf) + len(w) + 1 > chunk_size:
                        flush(buf, start_offset)
                        start_offset += len(buf)
                        buf = w
                    else:
                        buf = (buf + " " + w).strip()
                current = buf
            if len(current) > chunk_size:
                # candidate itself is still too big (very long unit) - handled above
                pass

    if current:
        flush(current, start_offset)

    return chunks
