import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

_model_cache = {}


def clean_text(text: str) -> str:
  """Normalize extracted PDF text by removing artifacts and excessive whitespace.

  Removes:
  - Null bytes (\x00)
  - Soft hyphens (\xad)
  - Repeated whitespace
  - Noisy line breaks (multiple consecutive newlines)
  """
  text = text.replace('\x00', '')
  text = text.replace('\xad', '')
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'\n\n+', '\n', text)
  return text.strip()


def extract_pages_for_rag(pdf_input: bytes | str | Path) -> list[dict]:
  """Extract pages from PDF with original page numbers, no hard limit.

  Accepts bytes, file path (str), or Path object.
  Filters out empty/whitespace-only pages.
  Returns list of {page: int, text: str} dicts with cleaned text.
  """
  if isinstance(pdf_input, (str, Path)):
    pdf_bytes = Path(pdf_input).read_bytes()
  else:
    pdf_bytes = pdf_input

  reader = PdfReader(BytesIO(pdf_bytes))

  records = []
  for page_number, page in enumerate(reader.pages, start=1):
    raw_text = page.extract_text() or ""
    cleaned = clean_text(raw_text)

    if cleaned:
      records.append({
        "page": page_number,
        "text": cleaned
      })

  return records


def save_json(obj: Any, path: str | Path) -> None:
  """Save Python object to JSON file, creating parent folders if needed.

  Writes UTF-8 JSON with 2-space indentation.
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  with open(path, 'w', encoding='utf-8') as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
  """Read JSON file back into Python object."""
  path = Path(path)

  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)


def preview_records(records: list[dict], columns: Optional[list[str]] = None) -> None:
  """Print a small table preview of records for inspection.

  Shows chosen columns (default: page, text).
  Useful for debugging page/chunk artifacts in notebooks.
  """
  if not records:
    print("No records to preview")
    return

  if columns is None:
    columns = ["page", "text"]

  print(f"\n{'─' * 80}")
  print(f"Preview: {len(records)} records")
  print(f"{'─' * 80}\n")

  for record in records[:5]:
    for col in columns:
      value = record.get(col, "N/A")
      if isinstance(value, str) and len(value) > 60:
        value = value[:57] + "..."
      print(f"{col:12} | {value}")
    print()

  if len(records) > 5:
    print(f"... and {len(records) - 5} more records\n")


def slice_long_text(text: str, chunk_size: int) -> list[str]:
  """Split oversized text into chunks at word boundaries.

  Prefers natural boundaries (spaces, newlines) over mid-word splits.
  Returns list of text pieces, each ≤ chunk_size characters.
  """
  if len(text) <= chunk_size:
    return [text]

  chunks = []
  pos = 0

  while pos < len(text):
    end = min(pos + chunk_size, len(text))

    if end == len(text):
      chunks.append(text[pos:])
      break

    last_space = text.rfind(' ', pos, end)
    if last_space > pos:
      end = last_space + 1

    chunks.append(text[pos:end].strip())
    pos = end

  return [c for c in chunks if c]


def chunk_by_paragraph(records: list[dict], chunk_size: int) -> list[dict]:
  """Split records by paragraph, using `slice_long_text` for oversized paragraphs.

  Preserves page numbers and paragraph order.
  Paragraphs are split on double newlines.
  """
  chunks = []
  chunk_id = 0

  for record in records:
    page = record["page"]
    text = record["text"]

    paragraphs = text.split('\n\n')

    for para in paragraphs:
      para = para.strip()
      if not para:
        continue

      if len(para) <= chunk_size:
        chunks.append({
          "chunk_id": f"page_{page}_chunk_{chunk_id}",
          "page": page,
          "text": para,
          "chunk_mode": "paragraph"
        })
        chunk_id += 1
      else:
        sub_chunks = slice_long_text(para, chunk_size)
        for sub_text in sub_chunks:
          chunks.append({
            "chunk_id": f"page_{page}_chunk_{chunk_id}",
            "page": page,
            "text": sub_text,
            "chunk_mode": "paragraph"
          })
          chunk_id += 1

  return chunks


def chunk_by_characters(records: list[dict], chunk_size: int, overlap: int = 0) -> list[dict]:
  """Split records into fixed-size character chunks with optional overlap.

  No paragraph awareness; splits allowed mid-word.
  Overlap > 0 reuses characters between consecutive chunks.
  """
  chunks = []
  chunk_id = 0

  for record in records:
    page = record["page"]
    text = record["text"]

    if overlap >= chunk_size:
      overlap = max(0, chunk_size - 1)

    step = chunk_size - overlap

    for start in range(0, len(text), step):
      end = min(start + chunk_size, len(text))
      chunk_text = text[start:end].strip()

      if chunk_text:
        chunks.append({
          "chunk_id": f"page_{page}_chunk_{chunk_id}",
          "page": page,
          "text": chunk_text,
          "chunk_mode": "character_overlap" if overlap > 0 else "character"
        })
        chunk_id += 1

  return chunks


def chunk_with_langchain_recursive(
    records: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[list[str]] = None
) -> list[dict]:
  """Split records using LangChain's RecursiveCharacterTextSplitter.

  Respects semantic boundaries (double newline → single newline → space → character).
  Cleaner output on noisy PDF text where paragraph breaks are incomplete.

  Args:
    records: List of {page, text} records from extract_pages_for_rag
    chunk_size: Target chunk size in characters
    chunk_overlap: Overlap between consecutive chunks
    separators: List of separators to try in order. Defaults to ["\n\n", "\n", " ", ""]

  Returns:
    List of chunks with {chunk_id, page, text, chunk_mode: "langchain_recursive"}

  Raises:
    ImportError: If langchain-text-splitters is not installed
  """
  try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
  except ImportError:
    raise ImportError(
      "langchain-text-splitters is required for langchain_recursive mode. "
      "Install it with: pip install langchain-text-splitters"
    )

  if separators is None:
    separators = ["\n\n", "\n", " ", ""]

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=separators,
    keep_separator=False
  )

  chunks = []
  chunk_id = 0

  for record in records:
    page = record["page"]
    text = record["text"]

    split_texts = splitter.split_text(text)

    for chunk_text in split_texts:
      chunk_text = chunk_text.strip()
      if chunk_text:
        chunks.append({
          "chunk_id": f"page_{page}_chunk_{chunk_id}",
          "page": page,
          "text": chunk_text,
          "chunk_mode": "langchain_recursive"
        })
        chunk_id += 1

  return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str,
    chunk_size: int,
    overlap: int = 0
) -> list[dict]:
  """Select and apply the requested chunking strategy.

  Args:
    records: List of {page, text} records from extract_pages_for_rag
    chunk_mode: "paragraph", "character", "character_overlap", or "langchain_recursive"
    chunk_size: Target chunk size in characters
    overlap: Overlap characters (used for "character_overlap" and "langchain_recursive")

  Returns:
    List of chunks with {chunk_id, page, text, chunk_mode}
  """
  if chunk_mode == "paragraph":
    return chunk_by_paragraph(records, chunk_size)
  elif chunk_mode == "character":
    return chunk_by_characters(records, chunk_size, overlap=0)
  elif chunk_mode == "character_overlap":
    return chunk_by_characters(records, chunk_size, overlap=overlap)
  elif chunk_mode == "langchain_recursive":
    return chunk_with_langchain_recursive(records, chunk_size, overlap)
  else:
    raise ValueError(f"Unknown chunk_mode: {chunk_mode}. Must be one of: paragraph, character, character_overlap, langchain_recursive")


def model_tag(model_name: str) -> str:
  """Convert model name into safe filename suffix.

  Example: "sentence-transformers/all-MiniLM-L6-v2" → "all-MiniLM-L6-v2"
  """
  return model_name.split('/')[-1]


def resolve_model_source(model_name: str) -> str:
  """Prefer local cached model; fall back to HuggingFace.

  Returns the model identifier to pass to SentenceTransformer.
  """
  return model_name


def get_device() -> str:
  """Choose CPU or CUDA based on availability.

  Returns "cuda" if CUDA is available, else "cpu".
  """
  return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, device: str):
  """Load or reuse a SentenceTransformer model instance.

  Caches models to avoid redundant loading.
  """
  key = (model_name, device)

  if key not in _model_cache:
    model = SentenceTransformer(model_name, device=device)
    _model_cache[key] = model

  return _model_cache[key]


def embed_texts(texts: list[str], model, device: str, batch_size: int = 32) -> np.ndarray:
  """Encode texts into normalized float32 vectors.

  Args:
    texts: List of text strings
    model: SentenceTransformer instance
    device: "cpu" or "cuda"
    batch_size: Number of texts to encode at once

  Returns:
    numpy array of shape (len(texts), embedding_dim), dtype float32
  """
  embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
  return embeddings.astype(np.float32)


def artifact_paths_for(
    document_id: str,
    pdf_name: str,
    model_tag_str: str,
    device: str,
    chunk_mode: str,
    chunk_size: int,
    overlap: int,
    artifact_root: str | Path
) -> dict:
  """Decide where artifacts should be saved.

  Returns dict with keys: raw_pages_path, chunk_path, embedding_path, manifest_path
  """
  artifact_root = Path(artifact_root)
  doc_folder = artifact_root / f"{document_id}_{Path(pdf_name).stem}"
  doc_folder.mkdir(parents=True, exist_ok=True)

  return {
    "raw_pages_path": doc_folder / "raw_pages.json",
    "chunk_path": doc_folder / f"chunks_{chunk_mode}_{chunk_size}_{overlap}.json",
    "embedding_path": doc_folder / f"embeddings_{model_tag_str}_{device}.npy",
    "manifest_path": doc_folder / f"manifest_{model_tag_str}.json"
  }


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
    batch_size: int = 32,
    artifact_root: str | Path = "artifacts"
) -> dict:
  """Build or reuse the full pages -> chunks -> embeddings -> manifest bundle.

  Orchestrates extraction, chunking, embedding, and saves all artifacts.
  Reuses saved outputs when signature matches.

  Args:
    document_id: Unique identifier for this document
    pdf_name: Original PDF filename (used in folder name)
    pages: List of {page, text} records from extract_pages_for_rag
    chunk_mode: "paragraph", "character", or "character_overlap"
    model_name: Model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: Target chunk size in characters
    overlap: Overlap for character_overlap mode
    batch_size: Batch size for embedding
    artifact_root: Root directory for artifacts

  Returns:
    Bundle dict with keys: pages, chunks, embeddings, manifest
    - pages: list of {page, text}
    - chunks: list of chunk dicts from build_chunks()
    - embeddings: np.ndarray of shape (num_chunks, embedding_dim), dtype float32
    - manifest: dict with metadata (num_pages, num_chunks, embedding_dim, device, chunk_mode, chunk_size, overlap, model_name, paths)
  """
  artifact_root = Path(artifact_root)
  tag = model_tag(model_name)
  device = get_device()

  paths = artifact_paths_for(
    document_id, pdf_name, tag, device, chunk_mode, chunk_size, overlap, artifact_root
  )

  save_json(pages, paths["raw_pages_path"])

  chunks = build_chunks(pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap)
  save_json(chunks, paths["chunk_path"])

  texts = [c["text"] for c in chunks]
  model = load_model(model_name, device)
  embeddings = embed_texts(texts, model, device, batch_size=batch_size)

  np.save(paths["embedding_path"], embeddings)

  manifest = {
    "document_id": document_id,
    "pdf_name": pdf_name,
    "num_pages": len(pages),
    "chunk_mode": chunk_mode,
    "chunk_size": chunk_size,
    "overlap": overlap,
    "model_name": model_name,
    "num_chunks": len(chunks),
    "embedding_dim": embeddings.shape[1],
    "device": device,
    "chunk_path": str(paths["chunk_path"]),
    "embedding_path": str(paths["embedding_path"]),
    "raw_pages_path": str(paths["raw_pages_path"])
  }

  save_json(manifest, paths["manifest_path"])

  return {
      "pages": pages,
      "chunks": chunks,
      "embeddings": embeddings,
      "manifest": manifest
  }
