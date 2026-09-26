# Cortex — RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) application. Upload
PDF/DOCX/TXT documents, ask questions in a chat interface, and get answers
grounded in your own documents — with the exact sources cited.

**Stack:** Python · Flask · Gemini API (generation + embeddings) · Qdrant
(vector database) · vanilla HTML/CSS/JS (glassmorphism UI, no frontend framework).

---

## 1. Architecture

**Ingestion** (`services/document_service.py`):

```
Upload → validate (type/size/corruption) → extract text (per-page for PDF)
       → clean → chunk (configurable size/overlap) → embed (Gemini)
       → store vectors + metadata (Qdrant) → update status
```

**Question answering** (`services/rag_service.py`):

```
Question → validate → embed (Gemini) → similarity search (Qdrant)
         → apply score threshold → build bounded context → Gemini generate
         → grounded answer + cited sources
```

The two pipelines are fully independent — ingestion never calls the
generation endpoint, and Q&A never re-processes documents.

## 2. Project structure

```
rag-app/
├── app.py                    Flask app + REST endpoints
├── requirements.txt
├── .env.example               copy to .env and fill in
├── config/
│   ├── settings.py            all configuration, env-var driven
│   └── rag_prompt.txt         the RAG system prompt sent to Gemini
├── services/
│   ├── gemini_service.py      Gemini generation + embeddings (server-side only)
│   ├── qdrant_service.py      collection mgmt, upsert, search, delete
│   ├── document_registry.py   local JSON registry (filename, status, chunk count)
│   ├── document_service.py    text extraction + ingestion pipeline
│   └── rag_service.py         retrieval + grounded generation
├── utils/
│   ├── chunking.py            paragraph/sentence-aware chunking with overlap
│   ├── file_validation.py     extension/size/magic-byte checks, path safety
│   └── helpers.py             JSON response helpers
├── templates/index.html
└── static/{style.css, app.js}
```

## 3. Setup

### 3.1 Prerequisites
- Python 3.10+
- A Gemini API key: https://aistudio.google.com/apikey
- A Qdrant instance — either:
  - **Local (Docker):** `docker run -p 6333:6333 qdrant/qdrant`
  - **Qdrant Cloud (free tier):** https://cloud.qdrant.io

### 3.2 Install

```bash
cd rag-app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your-real-key
QDRANT_URL=http://localhost:6333          # or your Qdrant Cloud URL
QDRANT_API_KEY=                            # required for Qdrant Cloud
```

### 3.3 Run

```bash
python app.py
```

Open http://localhost:5000.

On first run the app checks whether the configured Qdrant collection
(`QDRANT_COLLECTION_NAME`, default `rag_knowledge_base`) exists and creates
it automatically, sized to match `EMBEDDING_DIMENSIONS`.

## 4. Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API key (never sent to the browser) | — (required) |
| `GEMINI_MODEL` | Text generation model | `gemini-3.5-flash` |
| `GEMINI_EMBEDDING_MODEL` | Embedding model | `gemini-embedding-2` |
| `EMBEDDING_DIMENSIONS` | Output vector size (must match the Qdrant collection) | `768` |
| `QDRANT_URL` | Qdrant endpoint | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant Cloud API key | — |
| `QDRANT_COLLECTION_NAME` | Collection name | `rag_knowledge_base` |
| `CHUNK_SIZE` | Target characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks | `150` |
| `TOP_K` | Chunks retrieved per question | `5` |
| `SIMILARITY_THRESHOLD` | Minimum cosine similarity to keep a match (0–1) | `0.5` |
| `MAX_UPLOAD_SIZE_MB` | Max upload size | `20` |
| `ALLOWED_EXTENSIONS` | Accepted file types | `pdf,docx,txt` |

> **Model names change over time.** `gemini-3.5-flash` and `gemini-embedding-2`
> were the current stable/recommended Gemini models as of this build (Sept
> 2026). If either has since been deprecated, check
> https://ai.google.dev/gemini-api/docs/models and update `GEMINI_MODEL` /
> `GEMINI_EMBEDDING_MODEL` in `.env` — no code changes needed.

## 5. API reference

All responses use `{"success": true, "data": {...}}` or
`{"success": false, "error": {"code": "...", "message": "..."}}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Gemini/Qdrant connectivity + knowledge-base stats |
| GET | `/api/documents` | List documents and their status |
| POST | `/api/documents` | Upload a file (`multipart/form-data`, field `file`) |
| DELETE | `/api/documents/<id>` | Delete a document and its vectors |
| POST | `/api/chat` | `{"question": "...", "history": [...]}` → grounded answer + sources |

## 6. Document ingestion workflow (what happens on upload)

1. File is validated: extension, non-zero size, size limit, and a magic-byte
   check (catches a renamed `.exe` claiming to be a `.pdf`, for example).
2. Filename is sanitized and the file is saved under `uploads/` with a
   generated id prefix — this plus a resolved-path check blocks path
   traversal.
3. Text is extracted (`pypdf` for PDF — page by page, so page numbers are
   preserved; `python-docx` for DOCX, including table cells; UTF‑8/Latin‑1
   fallback for TXT).
4. Text is cleaned and chunked with configurable size/overlap, splitting on
   paragraph/sentence boundaries rather than arbitrary character cuts.
5. Chunks are embedded in a batch call to Gemini and upserted into Qdrant
   with metadata (`document_id`, `filename`, `chunk_index`, `page_number`).
6. The local registry (`data/documents.json`) is updated at each stage so
   the UI can show Uploading → Extracting → Chunking → Embedding → Indexing
   → Completed, or Failed with the specific error.

## 7. RAG query workflow

1. Question is validated (non-empty, length-capped).
2. Question is embedded with `task_type=RETRIEVAL_QUERY` (vs.
   `RETRIEVAL_DOCUMENT` for chunks — this asymmetry measurably improves
   retrieval quality with Gemini's embedding models).
3. Qdrant returns the top-`k` chunks above `SIMILARITY_THRESHOLD`. If none
   qualify, the app returns a clear "not enough information" response
   instead of calling Gemini at all.
4. Matched chunks are assembled into a context block (hard-capped at
   ~12,000 characters so a huge match never balloons the prompt) and sent
   to Gemini together with the RAG system prompt (`config/rag_prompt.txt`),
   which instructs the model to answer only from context and say so when it
   can't.
5. The response is returned with a parallel `sources` array (filename, page
   number when available, a relevance percentage from the similarity score,
   and a snippet) — never fabricated, always the chunks actually retrieved.

## 8. Testing this yourself

Manual checklist used while building this:
- Upload a PDF, DOCX, and TXT — confirm each shows `Completed` with a
  correct chunk count.
- Upload an empty file, a non-`.pdf/.docx/.txt` file, and a renamed file
  with the wrong content — confirm each is rejected with a specific message.
- Ask a question clearly answered by an uploaded document — confirm the
  answer and cited source/page match.
- Ask an unrelated question — confirm the app says the knowledge base
  doesn't have enough information rather than guessing.
- Delete a document — confirm its chunks stop showing up as sources on a
  follow-up question.
- Resize the browser to a phone width — confirm the tab switcher works and
  nothing scrolls horizontally.
- View page source / devtools network tab — confirm no API key appears
  anywhere in HTML/JS/responses.

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Health panel shows Qdrant offline | `QDRANT_URL` wrong, Qdrant not running, or firewall blocking the port |
| Health panel shows Gemini offline | `GEMINI_API_KEY` missing/invalid, or the configured model name has been retired |
| Upload fails with a vector-size error | `EMBEDDING_DIMENSIONS` doesn't match an *existing* Qdrant collection — use a new `QDRANT_COLLECTION_NAME` or recreate the collection |
| PDF upload fails with "no extractable text" | The PDF is a scanned image with no text layer (OCR is out of scope here) |
| 413 on upload | File exceeds `MAX_UPLOAD_SIZE_MB` |

## 10. Deployment considerations

- Run behind a production WSGI server (`gunicorn app:app`), not
  `flask run`/`python app.py`.
- Put a reverse proxy (nginx/Caddy) in front for TLS and to enforce upload
  size at the edge too.
- `data/documents.json` and `uploads/` should live on persistent storage
  (a volume) if you deploy in a container — they are not committed to git.
- Consider moving the document registry from a JSON file to a real database
  if you expect concurrent writers or a large document count.
- Never commit `.env`; only `.env.example` is tracked.
