# 📚 DocuMind — RAG-Powered Document Q&A System

> **Retrieval-Augmented Generation (RAG) using the [Endee](https://github.com/endee-io/endee) vector database**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Endee](https://img.shields.io/badge/Vector%20DB-Endee-orange)](https://endee.io)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🗂 Table of Contents

- [Problem Statement](#-problem-statement)
- [System Design](#-system-design)
- [How Endee Is Used](#-how-endee-is-used)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Running Tests](#-running-tests)
- [Configuration](#-configuration)
- [Example Walkthrough](#-example-walkthrough)

---

## 🎯 Problem Statement

Knowledge workers spend enormous time searching through large document collections for specific answers.  
Traditional keyword search fails to capture *semantic meaning* — a query for "climate solutions" won't match a passage about "carbon-reduction strategies" even though they mean the same thing.

**DocuMind solves this with a RAG pipeline:**

1. Documents are chunked, embedded into dense vector representations, and stored in **Endee** for sub-millisecond similarity search.
2. When a user asks a question, the most semantically relevant chunks are retrieved from Endee.
3. An LLM uses only those retrieved chunks to generate a **grounded, cited answer** — eliminating hallucination.

---

## 🏗 System Design

```
┌───────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                     │
│                                                               │
│  Raw Text / File                                              │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────┐    ┌───────────────────┐    ┌────────────┐  │
│  │  Text       │    │  Sentence         │    │   Endee    │  │
│  │  Chunker    │───▶│  Transformer      │───▶│  Vector DB │  │
│  │  (512 char, │    │  (all-MiniLM-L6)  │    │  (cosine,  │  │
│  │   64 ovlap) │    │  384-dim vectors  │    │   INT8)    │  │
│  └─────────────┘    └───────────────────┘    └────────────┘  │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                         │
│                                                               │
│  User Question                                                │
│       │                                                       │
│       ▼                                                       │
│  ┌──────────────┐    ┌────────────┐    ┌──────────────────┐  │
│  │  Embedding   │    │   Endee    │    │   LLM (GPT-4o)   │  │
│  │  Model       │───▶│  ANN Search│───▶│  + Context       │  │
│  │  (same model │    │  top-k=5   │    │  → Grounded      │  │
│  │   as ingest) │    │  chunks    │    │    Answer        │  │
│  └──────────────┘    └────────────┘    └──────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast, accurate, 384-dim — ideal balance for local inference |
| Similarity metric | Cosine | Normalised embeddings → cosine = dot product; direction > magnitude |
| Precision | INT8 | Endee INT8 quantisation gives ~4× memory savings with negligible accuracy loss |
| Chunk size | 512 chars / 64 overlap | Preserves sentence context; overlap prevents boundary misses |
| LLM | GPT-4o-mini (optional) | Low cost, strong instruction following; fallback to extractive mode |

---

## 🗄 How Endee Is Used

Endee is the **core vector storage and retrieval engine** of DocuMind. Every interaction with the knowledge base goes through Endee.

### 1. Index Creation

```python
from endee import Endee, Precision

client = Endee()          # connects to localhost:8080
client.create_index(
    name="documind_chunks",
    dimension=384,          # matches all-MiniLM-L6-v2 output
    space_type="cosine",    # semantic similarity metric
    precision=Precision.INT8,  # memory-efficient quantisation
)
```

### 2. Upserting Document Vectors

Each chunk is stored with its embedding and full metadata (text, filename, doc_id):

```python
index = client.get_index("documind_chunks")
index.upsert([
    {
        "id":     "abc123_c0001",
        "vector": [0.12, -0.45, ...],   # 384-dim float list
        "meta":   {
            "doc_id":   "abc123",
            "filename": "climate_report.txt",
            "text":     "The Paris Agreement commits nations to…",
            "chunk_idx": 1,
        },
    }
])
```

### 3. Semantic Search

The query is embedded with the same model and sent to Endee's ANN index:

```python
query_vector = embedder.encode("What is the Paris Agreement?").tolist()
results = index.query(vector=query_vector, top_k=5)
# → returns list of {id, similarity, meta} sorted by cosine similarity
```

### Why Endee?

- **High-performance HNSW indexing** — sub-millisecond ANN queries at scale
- **Simple REST / SDK API** — no complex configuration required
- **Docker-ready** — single `docker compose up` to start
- **Up to 1B vectors on a single node** — production-grade scalability
- **INT8 precision support** — 4× memory reduction with minimal quality loss

---

## 📁 Project Structure

```
documind/
├── app.py                   # FastAPI REST API server
├── demo.py                  # CLI demo (no server required)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml       # Runs Endee + DocuMind together
├── src/
│   ├── __init__.py
│   ├── rag_engine.py        # Core: chunking, embedding, Endee upsert/search
│   └── qa_pipeline.py       # QA: context building + LLM generation
└── tests/
    └── test_documind.py     # Unit tests (chunking, ingestion, search, QA)
```

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for Endee)
- (Optional) OpenAI API key for generative answers

### Step 1 — Fork & Clone

> **Required by evaluation rules:** Star and fork the [Endee repository](https://github.com/endee-io/endee) before proceeding.

```bash
# After forking on GitHub:
git clone https://github.com/<your-username>/endee
cd endee

# Then clone DocuMind alongside it
git clone https://github.com/<your-username>/documind
cd documind
```

### Step 2 — Start Endee

```bash
docker compose up endee -d
# Endee is now running at http://localhost:8080
```

Verify:
```bash
curl http://localhost:8080/api/v1/index/list
# → {"indexes": []}
```

### Step 3 — Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 — Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-...    ← enables generative mode
#   OPENAI_MODEL=gpt-4o-mini
#   ENDEE_HOST=http://localhost:8080
```

---

## 🚀 Running the Application

### Option A — CLI Demo (Quickest)

No server needed. Ingests 3 sample documents and runs Q&A:

```bash
python demo.py

# Ask a custom question:
python demo.py --question "What are the benefits of renewable energy?"
```

Sample output:
```
════════════════════════════════════════════════════════════
  DocuMind — RAG Demo (powered by Endee Vector DB)
════════════════════════════════════════════════════════════

[1/3] Initialising RAG engine …
[2/3] Ingesting sample documents …
  ✓ climate_change.txt   (4 chunks, doc_id=a1b2c3d4e5f6)
  ✓ machine_learning.txt (4 chunks, doc_id=f6e5d4c3b2a1)
  ✓ space_exploration.txt (4 chunks, doc_id=123456789abc)

[3/3] Running 5 Q&A queries …

────────────────────────────────────────────────────────────
Q: What caused climate change?
A: Climate change has been primarily driven by human activities since the 1800s,
   especially the burning of fossil fuels such as coal, oil, and gas, which
   release heat-trapping gases. [Source: climate_change.txt]
Mode: extractive
Sources:
  • climate_change.txt  (sim=0.9732)  Climate change refers to long-term shifts…
```

### Option B — REST API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

### Option C — Full Docker Stack

```bash
# Set your OpenAI key (optional)
export OPENAI_API_KEY=sk-...

docker compose up --build
```

Both services start:
- Endee: `http://localhost:8080`
- DocuMind API: `http://localhost:8000`

---

## 🔌 API Reference

### `POST /ingest/text`

Ingest raw text into the knowledge base.

```json
{
  "text": "The Great Wall of China is one of the greatest wonders…",
  "filename": "great_wall.txt"
}
```

Response:
```json
{
  "status": "ingested",
  "doc_id": "a1b2c3d4e5f6",
  "filename": "great_wall.txt",
  "num_chunks": 3
}
```

### `POST /ingest/file`

Upload a `.txt` file:

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@my_document.txt"
```

### `POST /ask`

Ask a question and get a grounded answer:

```json
{
  "question": "What is the Great Wall of China made of?",
  "top_k": 5
}
```

Response:
```json
{
  "question": "What is the Great Wall of China made of?",
  "answer": "The Great Wall was built from stone, brick, tamped earth…",
  "sources": [
    {
      "filename": "great_wall.txt",
      "chunk_id": "a1b2c3_c0001",
      "similarity": 0.9412,
      "excerpt": "The Great Wall of China is one of…"
    }
  ],
  "mode": "generative"
}
```

### `POST /search`

Raw semantic search (no LLM generation):

```json
{
  "query": "renewable energy sources",
  "top_k": 3
}
```

### `GET /health`

Service health check.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers:
- Text chunking (edge cases, overlap, whitespace)
- Document ingestion (chunk count, Endee upsert called)
- Semantic search (result parsing, empty results)
- QA pipeline (extractive mode, no-results graceful handling)

All tests use mocks for Endee and the embedding model — no live server required.

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ENDEE_HOST` | `http://localhost:8080` | Endee server URL |
| `ENDEE_AUTH_TOKEN` | `""` | Endee auth token (leave blank for open mode) |
| `OPENAI_API_KEY` | `""` | OpenAI key (enables generative mode) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |

---

## 🔄 Example Walkthrough

```python
from src.rag_engine import RAGEngine
from src.qa_pipeline import QAPipeline

# 1. Start Endee (docker compose up endee -d)

# 2. Initialise
rag = RAGEngine()   # connects to Endee, creates index if needed
qa  = QAPipeline(rag_engine=rag)

# 3. Ingest a document
with open("my_report.txt") as f:
    rag.ingest_text(f.read(), filename="my_report.txt")

# 4. Ask questions
result = qa.ask("What were the main findings?")
print(result["answer"])
# → "The main findings indicate that… [Source: my_report.txt]"
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)
