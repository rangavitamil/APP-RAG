"""
Qdrant vector database integration.

Handles collection creation, upsert, similarity search, and deletion.
Nothing here uses fake/mock data - every call talks to a real Qdrant
instance configured via QDRANT_URL / QDRANT_API_KEY.
"""

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantServiceError(Exception):
    pass


class QdrantService:
    def __init__(self, url: str, api_key: Optional[str], collection_name: str,
                 vector_size: int):
        self.collection_name = collection_name
        self.vector_size = vector_size
        try:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=15)
        except Exception as exc:
            logger.error("Could not create Qdrant client: %s", exc)
            raise QdrantServiceError(
                "Could not connect to Qdrant. Check QDRANT_URL / QDRANT_API_KEY."
            ) from exc

    # ---------------------------------------------------------------
    def ensure_collection(self):
        """Create the collection if it doesn't exist, matching embedding dims."""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception as exc:
            logger.error("Qdrant unavailable while listing collections: %s", exc)
            raise QdrantServiceError("Qdrant is unavailable.") from exc

        if self.collection_name in existing:
            info = self.client.get_collection(self.collection_name)
            configured_size = info.config.params.vectors.size
            if configured_size != self.vector_size:
                raise QdrantServiceError(
                    f"Existing Qdrant collection '{self.collection_name}' has vector "
                    f"size {configured_size}, but the embedding model produces "
                    f"{self.vector_size}-dim vectors. Use a different "
                    f"QDRANT_COLLECTION_NAME or recreate the collection."
                )
            return

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self.collection_name)
        except Exception as exc:
            logger.error("Failed to create Qdrant collection: %s", exc)
            raise QdrantServiceError("Failed to create the Qdrant collection.") from exc

    # ---------------------------------------------------------------
    def upsert_chunks(self, document_id: str, filename: str, chunks: List[dict],
                       vectors: List[List[float]]):
        """
        chunks: list of {chunk_id, chunk_index, text, page_number}
        vectors: parallel list of embedding vectors
        """
        if len(chunks) != len(vectors):
            raise QdrantServiceError("Chunk/vector count mismatch during indexing.")

        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "page_number": chunk.get("page_number"),
                        "text": chunk["text"],
                    },
                )
            )

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            logger.error("Qdrant upsert failed: %s", exc)
            raise QdrantServiceError("Failed to store embeddings in Qdrant.") from exc

    # ---------------------------------------------------------------
    def search(self, query_vector: List[float], top_k: int, score_threshold: float) -> List[dict]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            raise QdrantServiceError("Failed to search the knowledge base.") from exc

        return [
            {
                "score": r.score,
                "document_id": r.payload.get("document_id"),
                "filename": r.payload.get("filename"),
                "chunk_id": r.payload.get("chunk_id"),
                "chunk_index": r.payload.get("chunk_index"),
                "page_number": r.payload.get("page_number"),
                "text": r.payload.get("text"),
            }
            for r in results
        ]

    # ---------------------------------------------------------------
    def delete_document(self, document_id: str):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            logger.error("Qdrant deletion failed for document %s: %s", document_id, exc)
            raise QdrantServiceError("Failed to delete document vectors from Qdrant.") from exc

    # ---------------------------------------------------------------
    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False
