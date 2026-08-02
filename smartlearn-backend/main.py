import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from services.pdf import extract_pages
from pydantic import BaseModel, Field
import re
from services.llm import answer_from_pages
import os
from fastapi.middleware.cors import CORSMiddleware

documents = {}

class ChatRequest(BaseModel):
    chat_id: str = Field(default="day2-demo")
    message: str = Field(min_length=2, max_length=2000)

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
    allow_headers=["Content-Type"],
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
        pages = extract_pages(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    total_chars = sum(len(page["text"]) for page in pages)

    if total_chars == 0:
        raise HTTPException(status_code=422, detail="PDF contains no readable text. OCR is not supported.")

    documents[chat_id] = pages
    return {
        "status": "success",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.chat_id not in documents:
        raise HTTPException(
            status_code=404,
            detail="Chat ID not found. Please upload a PDF first using /upload?chat_id=" + request.chat_id
        )

    pages = documents[request.chat_id]

    try:
        answer = answer_from_pages(pages, request.message)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream AI service failed: {str(e)}"
        )

    page_numbers = set()
    for match in re.finditer(r'\[Page (\d+)\]', answer):
        page_num = int(match.group(1))
        if any(p["page"] == page_num for p in pages):
            page_numbers.add(page_num)

    citations = sorted(list(page_numbers))

    return {
        "answer": answer,
        "citations": citations
    }
