"""
Question-answering pipeline:
  question -> validate -> embed query -> Qdrant search -> apply threshold
  -> build context -> Gemini generate -> answer + sources

Kept fully separate from the ingestion pipeline (document_service.py).
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000  # hard cap so we never send an entire document


class RagServiceError(Exception):
    pass


class RagService:
    def __init__(self, gemini_service, qdrant_service, system_prompt: str,
                 top_k: int, similarity_threshold: float):
        self.gemini = gemini_service
        self.qdrant = qdrant_service
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def answer(self, question: str, history: Optional[List[dict]] = None) -> dict:
        question = (question or "").strip()
        if not question:
            raise RagServiceError("Please enter a question.")
        if len(question) > 2000:
            raise RagServiceError("Question is too long (max 2000 characters).")

        query_vector = self.gemini.embed_query(question)

        results = self.qdrant.search(
            query_vector=query_vector,
            top_k=self.top_k,
            score_threshold=self.similarity_threshold,
        )

        if not results:
            return {
                "answer": (
                    "The uploaded knowledge base does not contain enough relevant "
                    "information to answer this question. Try rephrasing, or upload "
                    "a document that covers this topic."
                ),
                "sources": [],
            }

        context_blocks = []
        used = 0
        for i, r in enumerate(results, start=1):
            block = f"[{i}] Source: {r['filename']}"
            if r.get("page_number"):
                block += f" (page {r['page_number']})"
            block += f"\n{r['text']}"
            if used + len(block) > MAX_CONTEXT_CHARS:
                break
            context_blocks.append(block)
            used += len(block)

        context = "\n\n---\n\n".join(context_blocks)
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

        answer_text = self.gemini.generate_answer(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            history=history,
        )

        sources = [
            {
                "filename": r["filename"],
                "document_id": r["document_id"],
                "page_number": r.get("page_number"),
                "chunk_index": r.get("chunk_index"),
                "relevance": round(min(max(r["score"], 0.0), 1.0) * 100),
                "snippet": (r["text"][:220] + "...") if len(r["text"]) > 220 else r["text"],
            }
            for r in results[: len(context_blocks)]
        ]

        return {"answer": answer_text, "sources": sources}
