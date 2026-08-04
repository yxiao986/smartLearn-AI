# SmartLearn Agent

Ask questions about a PDF and get answers with page citations you can click.

Upload a lecture PDF or paper, ask in plain language, and every answer comes back
with the pages it came from — click one and the preview jumps straight there, so
you can check the answer against the source instead of trusting it.

## Deployment

The app is deployed to Vercel (frontend) and Railway (backend), but **both are
currently switched off to avoid hosting costs**. They can be turned back on at
any time; until then the links below will not respond.

- Frontend: https://smartlearn-aibyxyf.vercel.app/
- Backend: https://smartlearn-ai-production-87bf.up.railway.app/ (`/docs` for the API)

To try it in the meantime, run it locally — see [Getting started](#getting-started).

## What it does

- Upload a PDF and ask questions about it in plain language
- Every answer carries the page numbers the evidence came from
- Clicking a citation opens the PDF preview at that page
- Follow-up questions can rely on what was already asked
- When the document does not cover the question, it says so instead of inventing an answer

## How it works

The whole PDF is never sent to the model — a long document does not fit in the
context window, and it gets slower and more expensive with every page. Instead:

1. **On upload** — read the PDF with `pypdf`, keeping the page number of every piece of text
2. Split the text into chunks of about 700 characters, overlapping by 120
3. Turn each chunk into a vector with `all-MiniLM-L6-v2` and store it in a FAISS index
4. **On each question** — embed the question the same way and retrieve the 3 closest chunks
5. Build one prompt from the recent conversation, those chunks, and the question
6. Return the answer plus the pages the chunks came from

Because every chunk carries its page number through the pipeline, the citations
come from the retrieval step rather than from the model, so they point at real pages.

## Tech stack

| Part | Choice |
| --- | --- |
| Frontend | React 18 + Vite, deployed on Vercel |
| Backend | FastAPI (Python), deployed on Railway |
| PDF parsing | pypdf |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2`, run locally |
| Vector search | FAISS (`IndexFlatIP`) |
| Answering | Any model via OpenRouter |
| Chat history | PostgreSQL via SQLAlchemy (optional) |
| Answer rendering | react-markdown + KaTeX |

## Getting started

**Prerequisites:** Python 3.11+, Node 18+, and an [OpenRouter](https://openrouter.ai/) API key.
PostgreSQL is optional.

### 1. Install

```bash
git clone https://github.com/yxiao986/smartLearn-AI.git
cd smartLearn-AI

python -m venv venv
venv\Scripts\activate                 # Windows
# source venv/bin/activate            # macOS / Linux
pip install -r smartlearn-backend/requirements.txt

cd smartlearn-frontend && npm install && cd ..
```

The first run downloads the embedding model (~90 MB), so it takes a moment.

### 2. Configure

```bash
cp .env.example .env
```

Then put your OpenRouter key in `.env`. See [Environment variables](#environment-variables).

### 3. Run

Two terminals.

**Backend** — run this from the repository root, not from `smartlearn-backend/`:

```bash
uvicorn main:app --app-dir smartlearn-backend --reload
```

**Frontend:**

```bash
cd smartlearn-frontend
npm run dev
```

Open http://localhost:5173, upload a PDF, and ask a question.

> **Run the backend from the repository root.** The artifact directory is a
> relative path, so starting uvicorn from inside `smartlearn-backend/` writes
> chunks and indexes to the wrong place.

## Environment variables

All of these live in `.env` at the repository root.

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter key. Without it the app falls back to a simple extractive answer instead of calling a model. |
| `VITE_API_URL` | Yes | Backend URL the frontend calls. Defaults to `http://localhost:8000`. |
| `ALLOWED_ORIGINS` | Yes in production | Comma-separated origins allowed by CORS. Defaults to `http://localhost:5173`. |
| `OPENROUTER_MODEL` | No | Model to answer with. Defaults to `google/gemma-4-26b-a4b-it:free`. |
| `DAY3_DB_URL` | No | PostgreSQL URL for chat history. Without it, history is kept in memory and lost on restart. |

> `VITE_API_URL` is baked into the frontend at **build** time. Changing it on a
> deployed site does nothing until you rebuild.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service name |
| `GET` | `/health` | Health check |
| `POST` | `/upload?chat_id=<id>` | Upload a PDF (multipart `file`). Parses, chunks, embeds and indexes it. |
| `GET` | `/documents/{chat_id}/file` | The uploaded PDF, for the preview |
| `POST` | `/chat` | `{chat_id, message, current_page?}` → `{answer, citations, sources}` |

Interactive docs are at `/docs` on a running backend.

`chat_id` identifies one session. The frontend generates one per browser tab and
keeps it in `sessionStorage`.

## Project structure

```
smartlearn-backend/
  main.py              FastAPI routes and in-memory document store
  services/
    rag.py             Chunking, embeddings, FAISS, retrieval, citations, history
    llm.py             OpenRouter client and the grounded system prompt
    pdf.py             Page-level text extraction
  requirements.txt
  Dockerfile

smartlearn-frontend/
  src/
    App.jsx            Shared upload and active-citation state
    ChatPanel.jsx      Message list, markdown/KaTeX answers, citation chips
    PdfPreview.jsx     PDF viewer that jumps to a cited page
    PdfUploader.jsx    File picker and upload button
    api.js             Backend calls and the per-tab chat id
    index.css          All styling

artifacts/             Generated chunks, embeddings and FAISS indexes (git-ignored)
test_files/            Sample PDFs for manual testing
docs/                  Deployment notes, test logs, demo deck
```

## Limitations

- **No OCR.** A scanned PDF with no text layer is rejected rather than answered from nothing.
- **Documents are held in memory.** Restarting the backend means uploading again. Only chat history survives, and only when `DAY3_DB_URL` is set.
- **Chunks are a fixed size**, so a table or an equation can be split across two of them.
- **Vague follow-ups retrieve poorly.** Earlier turns are sent to the model, but the search still uses the words in the new question — so "tell me more" has little to match on.
- **One PDF per session.**
