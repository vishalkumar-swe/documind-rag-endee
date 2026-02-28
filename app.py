"""
DocuMind — FastAPI Web Application
RAG-powered Document Q&A using Endee Vector Database
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_engine import RAGEngine
from src.qa_pipeline import QAPipeline

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("documind")

# ───────────────────────────────────────────────
# FastAPI App
# ───────────────────────────────────────────────

app = FastAPI(
    title="DocuMind API",
    description="RAG-powered Document Q&A using Endee Vector Database",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────
# Lazy-loaded singletons
# ───────────────────────────────────────────────

_rag: Optional[RAGEngine] = None
_qa: Optional[QAPipeline] = None


def get_rag() -> RAGEngine:
    global _rag
    if _rag is None:
        logger.info("Initializing RAG Engine...")
        _rag = RAGEngine()
    return _rag


def get_qa(rag: RAGEngine = Depends(get_rag)) -> QAPipeline:
    global _qa
    if _qa is None:
        logger.info("Initializing QA Pipeline...")
        _qa = QAPipeline(rag_engine=rag)
    return _qa


# ───────────────────────────────────────────────
# Request Models
# ───────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str
    filename: str = "inline_document"


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ───────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────

@app.get("/", summary="Root endpoint")
def root():
    return {
        "message": "DocuMind API is running 🚀",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "service": "DocuMind",
    }


@app.post("/ingest/text", summary="Ingest raw text")
def ingest_text(req: IngestTextRequest, rag: RAGEngine = Depends(get_rag)):
    try:
        result = rag.ingest_text(req.text, filename=req.filename)
        return {
            "status": "ingested",
            **result,
        }
    except Exception as e:
        logger.exception("Text ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", summary="Upload and ingest a .txt file")
async def ingest_file(
    file: UploadFile = File(...),
    rag: RAGEngine = Depends(get_rag),
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported.",
        )

    try:
        content = (await file.read()).decode("utf-8")
        result = rag.ingest_text(content, filename=file.filename)

        return {
            "status": "ingested",
            **result,
        }

    except Exception as e:
        logger.exception("File ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", summary="Ask a question")
def ask_question(
    req: AskRequest,
    qa: QAPipeline = Depends(get_qa),
):
    try:
        return qa.ask(req.question, top_k=req.top_k)

    except Exception as e:
        logger.exception("QA request failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Semantic search")
def search(
    req: SearchRequest,
    rag: RAGEngine = Depends(get_rag),
):
    try:
        results = rag.search(req.query, top_k=req.top_k)

        return {
            "query": req.query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "filename": r.filename,
                    "similarity": round(r.similarity, 4),
                    "text": r.text,
                }
                for r in results
            ],
        }

    except Exception as e:
        logger.exception("Search request failed")
        raise HTTPException(status_code=500, detail=str(e))


# ───────────────────────────────────────────────
# Run directly (optional)
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )