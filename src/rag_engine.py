"""
DocuMind RAG Engine
FINAL Production-Safe Version
"""

import os
import re
import uuid
import logging
from typing import List, Tuple
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from endee import Endee
from endee.exceptions import ConflictException


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://127.0.0.1:8080")
ENDEE_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME = "documind_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id: str
    filename: str
    text: str
    similarity: float


# ──────────────────────────────────────────────
# RAG Engine
# ──────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        logger.info("🔹 Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        logger.info("🔹 Connecting to Endee at %s", ENDEE_HOST)

        self.client = Endee(ENDEE_TOKEN) if ENDEE_TOKEN else Endee()
        self.client.set_base_url(f"{ENDEE_HOST}/api/v1")

        self._ensure_index()

    # --------------------------------------------------
    # Ensure Index Exists (Bulletproof Logic)
    # --------------------------------------------------
    def _ensure_index(self):
        try:
            # Try to get existing index first
            self.index = self.client.get_index(name=INDEX_NAME)
            logger.info("✅ Index detected in Endee")
            return
        except Exception:
            logger.info("Index not found. Creating index '%s'...", INDEX_NAME)

        try:
            self.client.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                space_type="cosine",
                precision="float32"
            )
            logger.info("✅ Index created successfully")
        except ConflictException:
            # If race condition happens and index exists
            logger.info("Index already exists (conflict handled)")
        except Exception as e:
            raise RuntimeError(f"❌ Index creation failed: {e}")

        self.index = self.client.get_index(name=INDEX_NAME)

    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------
    @staticmethod
    def _split_text(text: str, chunk_size=450, overlap=50):
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()

        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    # --------------------------------------------------
    # Ingest
    # --------------------------------------------------
    def ingest_text(self, text: str, filename="document"):
        chunks = self._split_text(text)

        items = []

        for i, chunk in enumerate(chunks):
            embedding = self.embedder.encode(
                chunk,
                normalize_embeddings=True
            ).tolist()

            items.append({
                "id": f"{filename}_{i}",
                "vector": embedding,
                "meta": {
                    "filename": filename,
                    "text": chunk
                }
            })

        self.index.upsert(items)

        return {
            "filename": filename,
            "num_chunks": len(items),
            "doc_id": str(uuid.uuid4())[:8]
        }

    # --------------------------------------------------
    # Search
    # --------------------------------------------------
    def search(self, query: str, top_k=3) -> List[SearchResult]:
        vector = self.embedder.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.index.query(
            vector=vector,
            top_k=top_k
        )

        formatted = []

        for r in results:
            meta = r.get("meta", {})
            formatted.append(
                SearchResult(
                    chunk_id=r.get("id", ""),
                    filename=meta.get("filename", ""),
                    text=meta.get("text", ""),
                    similarity=r.get("similarity", 0.0)
                )
            )

        return formatted

    # --------------------------------------------------
    # Build Context (Required by QA Pipeline)
    # --------------------------------------------------
    def build_context(self, query: str, top_k=3) -> Tuple[str, List[SearchResult]]:
        results = self.search(query, top_k)

        if not results:
            return "", []

        context_parts = []

        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} — {r.filename} (similarity: {r.similarity:.3f})]\n{r.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        return context, results