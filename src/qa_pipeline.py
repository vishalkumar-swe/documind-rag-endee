"""
DocuMind QA Pipeline
Wraps the RAG engine with an LLM to provide grounded answers.
Supports OpenAI and a lightweight local fallback (no-LLM extractive mode).
"""

import os
import logging
from typing import Optional

from src.rag_engine import RAGEngine, SearchResult

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ──────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────

def _call_openai(system_prompt: str, user_prompt: str) -> str:
    """Call the OpenAI Chat Completions API."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _extractive_answer(query: str, results: list[SearchResult]) -> str:
    """
    Lightweight fallback: returns the top matching chunk as the answer.
    Useful when no LLM API key is configured.
    """
    if not results:
        return "No relevant information found in the knowledge base."
    top = results[0]
    return (
        f"[Extractive answer from '{top.filename}' — similarity {top.similarity:.3f}]\n\n"
        f"{top.text}"
    )


# ──────────────────────────────────────────────
# QA Pipeline
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocuMind, a precise and helpful document Q&A assistant.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information, say so clearly.
Cite the source filename when possible. Be concise yet thorough."""


class QAPipeline:
    """
    High-level QA pipeline:
      1. Retrieve relevant chunks via Endee (RAGEngine)
      2. Build a context string
      3. Call an LLM (or fallback to extractive mode)
    """

    def __init__(self, rag_engine: Optional[RAGEngine] = None):
        self.rag = rag_engine or RAGEngine()
        self.use_llm = bool(OPENAI_API_KEY)
        if not self.use_llm:
            logger.warning(
                "OPENAI_API_KEY not set — running in extractive mode (no LLM generation)."
            )

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        Answer a question using the RAG pipeline.

        Returns a dict with:
          - question:  original question
          - answer:    generated (or extractive) answer
          - sources:   list of source chunk metadata used
          - mode:      'generative' | 'extractive'
        """
        context, results = self.rag.build_context(question, top_k=top_k)

        if not results:
            return {
                "question": question,
                "answer":   "I couldn't find any relevant information in the knowledge base.",
                "sources":  [],
                "mode":     "no_results",
            }

        if self.use_llm:
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
            answer = _call_openai(SYSTEM_PROMPT, user_prompt)
            mode = "generative"
        else:
            answer = _extractive_answer(question, results)
            mode = "extractive"

        sources = [
            {
                "filename":   r.filename,
                "chunk_id":   r.chunk_id,
                "similarity": round(r.similarity, 4),
                "excerpt":    r.text[:200] + ("…" if len(r.text) > 200 else ""),
            }
            for r in results
        ]

        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "mode":     mode,
        }
