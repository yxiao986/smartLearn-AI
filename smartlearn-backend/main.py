import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from services import rag

# In-memory only: a restart drops every document, so there is nothing to gain
# from also keeping the uploaded PDF on disk.
documents: dict[str, dict] = {}

# Appendix A: PostgreSQL history storage (optional)
db_url = os.getenv("DAY3_DB_URL")
db_session_factory = None
if db_url:
  try:
    engine = rag.build_history_engine(db_url)
    rag.ensure_history_tables(engine)
    db_session_factory = rag.build_history_session_factory(engine)
    print(f"Database initialized at {db_url}")
  except Exception as e:
    print(f"Warning: Failed to initialize database: {e}")
    db_session_factory = None

class ChatRequest(BaseModel):
    chat_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    current_page: int = Field(None, description="Current PDF page for context")

app = FastAPI(title="Smartlearn Lite API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Smartlearn Lite API is running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(chat_id: str, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be PDF")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        pages = rag.extract_pages_for_rag(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    total_chars = sum(len(page["text"]) for page in pages)
    if total_chars == 0:
        raise HTTPException(status_code=422, detail="PDF contains no readable text. OCR is not supported.")

    try:
        document = rag.prepare_rag_document(
            document_id=chat_id,
            filename=file.filename,
            pages=pages,
            chunk_mode="character_overlap",
            chunk_size=700,
            overlap=120,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            artifact_root="artifacts"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare RAG document: {str(e)}")

    document["pdf_bytes"] = pdf_bytes
    documents[chat_id] = document

    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars
    }

@app.get("/documents/{chat_id}/file")
def get_document_file(chat_id: str):
    if chat_id not in documents:
        raise HTTPException(status_code=404, detail=f"Chat ID '{chat_id}' not found.")

    pdf_bytes = documents[chat_id].get("pdf_bytes")

    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="No PDF stored for this chat session.")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"}
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.chat_id not in documents:
        raise HTTPException(
            status_code=404,
            detail=f"Chat ID '{request.chat_id}' not found. Please upload a PDF first using POST /upload?chat_id={request.chat_id}"
        )

    document = documents[request.chat_id]

    try:
        if db_session_factory:
            # Appendix A: Use database-backed history
            result = rag.answer_chat_turn_with_history_store(
                document,
                request.chat_id,
                request.message,
                db_session_factory,
                top_k=3,
                candidate_pool=60,
                answer_model="poolside/laguna-s-2.1:free",
                current_page=request.current_page
            )
        else:
            # Lab C: Use in-memory history
            result = rag.answer_chat_turn(
                document,
                request.message,
                top_k=3,
                candidate_pool=60,
                answer_model="poolside/laguna-s-2.1:free",
                current_page=request.current_page
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )

    return result
