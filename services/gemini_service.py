"""
Gemini API integration.

All calls happen server-side only. The API key never reaches the browser.
Uses the official `google-genai` SDK (`from google import genai`).
"""

import logging
from typing import List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiServiceError(Exception):
    pass


class GeminiService:
    def __init__(self, api_key: str, generation_model: str, embedding_model: str,
                 embedding_dimensions: int):
        if not api_key:
            raise GeminiServiceError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    # ---------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks (RETRIEVAL_DOCUMENT task type)."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question (RETRIEVAL_QUERY task type)."""
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts:
            return []
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.embedding_dimensions,
                ),
            )
            return [list(e.values) for e in result.embeddings]
        except Exception as exc:
            logger.error("Gemini embedding call failed: %s", exc)
            raise GeminiServiceError(
                "Failed to generate embeddings from the Gemini API."
            ) from exc

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------
    def generate_answer(self, system_prompt: str, user_prompt: str,
                         history: List[dict] = None) -> str:
        """
        Generate a grounded answer.
        `history` is an optional list of {"role": "user"|"model", "text": str}
        for conversational continuity.
        """
        contents = []
        for turn in (history or []):
            role = "user" if turn.get("role") == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=turn.get("text", ""))])
            )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        )

        try:
            response = self.client.models.generate_content(
                model=self.generation_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            text = getattr(response, "text", None)
            if not text:
                raise GeminiServiceError("Gemini returned an empty response.")
            return text
        except GeminiServiceError:
            raise
        except Exception as exc:
            logger.error("Gemini generation call failed: %s", exc)
            raise GeminiServiceError(
                "Failed to get a response from the Gemini API."
            ) from exc

    def health_check(self) -> bool:
        """Lightweight check that the API key + model are usable."""
        try:
            self.client.models.embed_content(
                model=self.embedding_model,
                contents=["health check"],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.embedding_dimensions,
                ),
            )
            return True
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", exc)
            return False
