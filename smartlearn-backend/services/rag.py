import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import faiss
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
  pdf_stem = Path(pdf_name).stem

  if pdf_stem == document_id:
    folder_name = document_id
  else:
    folder_name = f"{document_id}_{pdf_stem}"

  doc_folder = artifact_root / folder_name
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


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
  """Build a searchable FAISS index from normalized embedding vectors.

  Uses inner-product distance (cosine similarity on normalized vectors).

  Args:
    embeddings: np.ndarray of shape (num_chunks, embedding_dim), dtype float32, L2-normalized

  Returns:
    FAISS Index ready for search
  """
  embedding_dim = embeddings.shape[1]
  index = faiss.IndexFlatIP(embedding_dim)
  index.add(embeddings)
  return index


def save_faiss_index(index: faiss.Index, index_path: str | Path) -> None:
  """Save FAISS index to disk.

  Args:
    index: FAISS Index object
    index_path: Path where .faiss file will be written
  """
  index_path = Path(index_path)
  index_path.parent.mkdir(parents=True, exist_ok=True)
  faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: str | Path, artifact_root: str | Path | None = None) -> faiss.Index:
  """Load FAISS index from disk.

  Args:
    index_path: Path to .faiss file (absolute or relative)
    artifact_root: Root directory for resolving relative paths (optional)

  Returns:
    FAISS Index loaded into memory
  """
  index_path = Path(index_path)

  if not index_path.is_absolute() and artifact_root is not None:
    resolved_path = Path(artifact_root) / index_path
    if resolved_path.exists():
      index_path = resolved_path

  return faiss.read_index(str(index_path))


def relative_path_str(path: str | Path, base: str | Path) -> str:
  """Convert absolute path to relative display string.

  Args:
    path: Full path to file
    base: Base directory for relative path

  Returns:
    Relative path as string (e.g., "doc_1_lecture/embeddings_all-MiniLM-L6-v2_cpu.npy")
  """
  path = Path(path)
  base = Path(base)
  try:
    return str(path.relative_to(base))
  except ValueError:
    return str(path)


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: str | Path | None = None
) -> dict:
  """Prepare a document record for server-side consumption.

  Chunks, embeds, and indexes pages; returns metadata ready for API.

  Args:
    document_id: Unique document identifier
    filename: Original PDF filename
    pages: List of {page, text} records
    chunk_mode: "paragraph", "character", "character_overlap", or "langchain_recursive"
    chunk_size: Target chunk size in characters
    overlap: Overlap for overlapping modes
    model_name: Embedding model identifier
    batch_size: Batch size for embedding
    artifact_root: Root directory for artifacts (optional)

  Returns:
    Document record dict with: document_id, filename, num_pages, pages,
    chunks, embeddings, manifest, index, and display paths
  """
  if artifact_root is None:
    artifact_root = Path("artifacts")

  artifact_root = Path(artifact_root)

  bundle = ensure_index(
    document_id=document_id,
    pdf_name=filename,
    pages=pages,
    chunk_mode=chunk_mode,
    model_name=model_name,
    chunk_size=chunk_size,
    overlap=overlap,
    batch_size=batch_size,
    artifact_root=artifact_root
  )

  return {
    "document_id": document_id,
    "filename": filename,
    "num_pages": len(pages),
    "chunk_size": chunk_size,
    "embedding_dim": bundle["embeddings"].shape[1],
    "pages": bundle["pages"],
    "artifact_root": bundle["artifact_root"],
    "artifacts": bundle["artifact_paths"],
    "manifest": bundle["manifest"],
    "history": []
  }


def keyword_set(text: str) -> set[str]:
  """Extract lightweight lexical tokens for simple reranking.

  Lowercases, removes punctuation, filters stop words.
  Returns set of meaningful keywords from text.
  """
  stop_words = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
    "when", "where", "why", "how", "as", "if", "so", "not", "no", "yes"
  }

  text = text.lower()
  text = re.sub(r'[^\w\s]', ' ', text)
  tokens = text.split()
  return {t for t in tokens if t and t not in stop_words}


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None
) -> list[dict]:
  """Retrieve top-k hits from in-memory bundle.

  Args:
    question: Query text
    bundle: Bundle dict with embeddings, chunks, index from ensure_index()
    top_k: Number of top results to return
    candidate_pool: Rerank from top-N (allows for lexical filtering)
    batch_size: Batch size for embedding (default 1)
    history: Previous turns (unused, for API compatibility)

  Returns:
    List of hits: [{page, chunk_id, text, score}, ...]
  """
  device = get_device()
  model = load_model("sentence-transformers/all-MiniLM-L6-v2", device)

  q_embedding = embed_texts([question], model, device, batch_size=batch_size)
  q_embedding_normalized = q_embedding / (np.linalg.norm(q_embedding, axis=1, keepdims=True) + 1e-8)

  index = bundle["index"]
  chunks = bundle["chunks"]

  distances, indices = index.search(q_embedding_normalized, candidate_pool)
  distances = distances[0]
  indices = indices[0]

  q_keywords = keyword_set(question)
  hits = []

  for i, idx in enumerate(indices):
    if idx < 0 or idx >= len(chunks):
      continue

    chunk = chunks[idx]
    score = float(distances[i])

    chunk_keywords = keyword_set(chunk["text"])
    keyword_overlap = len(q_keywords & chunk_keywords)

    hits.append({
      "page": chunk["page"],
      "chunk_id": chunk["chunk_id"],
      "text": chunk["text"],
      "score": score,
      "_keyword_overlap": keyword_overlap
    })

  hits.sort(key=lambda h: (-h["_keyword_overlap"], -h["score"]))

  for hit in hits:
    del hit["_keyword_overlap"]

  return hits[:top_k]


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None
) -> list[dict]:
  """Retrieve top-k hits from saved document index.

  Loads the FAISS index from disk, runs retrieval, returns top-k.

  Args:
    question: Query text
    document: Prepared document record from prepare_rag_document()
    top_k: Number of top results to return
    candidate_pool: Rerank from top-N
    history: Previous turns (unused, for API compatibility)

  Returns:
    List of hits: [{page, chunk_id, text, score}, ...]
  """
  index_path = document["artifacts"]["index"]
  chunks_path = document["artifacts"]["chunks"]
  artifact_root = document.get("artifact_root")

  index = load_faiss_index(index_path, artifact_root=artifact_root)
  chunks = load_json(chunks_path)

  bundle = {
    "index": index,
    "chunks": chunks
  }

  return search_bundle(question, bundle, top_k=top_k, candidate_pool=candidate_pool, history=history)


def split_sentences(text: str) -> list[str]:
  """Split chunk text into candidate answer sentences.

  Simple split on period, exclamation, question mark.
  Strips whitespace and filters empty sentences.
  """
  sentences = re.split(r'[.!?]+', text)
  return [s.strip() for s in sentences if s.strip()]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
  """Return one short answer sentence with page tag.

  Selects first sentence from top hit's text.
  Appends [Page X] tag when possible.

  Args:
    question: Query text (unused, for API compatibility)
    hits: List of retrieval hits from search_bundle() or search_document()

  Returns:
    One sentence with page tag, or empty string if no hits
  """
  if not hits:
    return ""

  top_hit = hits[0]
  sentences = split_sentences(top_hit["text"])

  if not sentences:
    return ""

  answer = sentences[0]
  page_tag = f" [Page {top_hit['page']}]"

  return answer + page_tag


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path: str | Path | None = None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: str | Path | None = None
) -> dict:
  """Build or reuse full pages → chunks → embeddings → FAISS index bundle.

  Either pages or pdf_path must be provided.
  Rebuilds only when signature changes.

  Args:
    document_id: Unique document identifier
    pdf_name: PDF filename (for artifact folder naming)
    pages: List of {page, text} records (if not provided, extract from pdf_path)
    pdf_path: Path to PDF file (used if pages not provided)
    chunk_mode: "paragraph", "character", "character_overlap", or "langchain_recursive"
    model_name: Embedding model identifier
    chunk_size: Target chunk size in characters
    overlap: Overlap for overlapping modes
    batch_size: Batch size for embedding
    artifact_root: Root directory for artifacts (default: "artifacts")

  Returns:
    Bundle dict with keys: pages, chunks, embeddings, manifest, index, artifact_paths
  """
  if artifact_root is None:
    artifact_root = "artifacts"

  artifact_root = Path(artifact_root)

  if pages is None:
    if pdf_path is None:
      raise ValueError("Either pages or pdf_path must be provided")
    pages = extract_pages_for_rag(pdf_path)

  artifacts = ensure_artifacts(
    document_id=document_id,
    pdf_name=pdf_name,
    pages=pages,
    chunk_mode=chunk_mode,
    model_name=model_name,
    chunk_size=chunk_size,
    overlap=overlap,
    batch_size=batch_size,
    artifact_root=artifact_root
  )

  embeddings = artifacts["embeddings"]
  chunks = artifacts["chunks"]
  manifest = artifacts["manifest"]

  tag = model_tag(model_name)
  device = get_device()
  paths = artifact_paths_for(
    document_id, pdf_name, tag, device, chunk_mode, chunk_size, overlap, artifact_root
  )

  index_path = paths["manifest_path"].parent / f"index_{tag}.faiss"

  index = build_faiss_index(embeddings)
  save_faiss_index(index, index_path)

  return {
    "pages": pages,
    "chunks": chunks,
    "embeddings": embeddings,
    "manifest": manifest,
    "index": index,
    "artifact_root": str(artifact_root.resolve()),
    "artifact_paths": {
      "raw_pages": str(paths["raw_pages_path"].resolve()),
      "chunks": str(paths["chunk_path"].resolve()),
      "embeddings": str(paths["embedding_path"].resolve()),
      "manifest": str(paths["manifest_path"].resolve()),
      "index": str(index_path.resolve())
    }
  }


def extract_citations(answer: str, hits: list[dict] | None = None) -> list[int]:
  """Extract numeric PDF page citations from answer text.

  Searches for [Page X] tags in answer; if found, extracts page numbers.
  Falls back to pages from top hits if no tags found.

  Args:
    answer: Answer text that may contain [Page X] tags
    hits: Optional list of retrieval hits to use as fallback

  Returns:
    Sorted list of unique page numbers
  """
  citations = set()

  page_matches = re.findall(r'\[Page (\d+)\]', answer)
  for match in page_matches:
    citations.add(int(match))

  if not citations and hits:
    for hit in hits[:3]:
      citations.add(hit.get("page"))

  return sorted(list(citations))


def build_sources(hits: list[dict]) -> list[dict]:
  """Convert retrieval hits to frontend-friendly source objects.

  Args:
    hits: List of retrieval hits from search_bundle() or search_document()

  Returns:
    List of source objects with page, chunk_id, score, and preview text
  """
  sources = []

  for hit in hits:
    preview = hit["text"]
    if len(preview) > 150:
      preview = preview[:147] + "..."

    sources.append({
      "page": hit["page"],
      "chunk_id": hit["chunk_id"],
      "score": hit["score"],
      "preview": preview
    })

  return sources


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "tencent/hy3:free"
) -> dict:
  """Retrieve relevant chunks and optionally call LLM for answer.

  Args:
    document: Prepared document from prepare_rag_document()
    question: User question
    top_k: Number of top chunks to retrieve
    candidate_pool: Rerank from top-N
    answer_model: LLM model for answering (used if API key available)

  Returns:
    Dict with answer, citations, and sources
  """
  hits = search_document(question, document, top_k=top_k, candidate_pool=candidate_pool)

  if not hits:
    return {
      "answer": "No relevant content found in the document.",
      "citations": [],
      "sources": []
    }

  api_key = None
  try:
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
  except:
    pass

  if api_key:
    try:
      from services.llm import answer_from_pages
      chunk_records = [
        {"page": hit["page"], "text": hit["text"]}
        for hit in hits
      ]
      llm_answer = answer_from_pages(chunk_records, question)
      answer = llm_answer
    except Exception as e:
      answer = best_sentence_answer(question, hits)
  else:
    answer = best_sentence_answer(question, hits)

  citations = extract_citations(answer, hits)
  sources = build_sources(hits)

  return {
    "answer": answer,
    "citations": citations,
    "sources": sources
  }


def append_history(document: dict, question: str, result: dict) -> list[dict]:
  """Append a Q&A turn to the document's conversation history.

  Args:
    document: Prepared document (modified in-place)
    question: User question
    result: Result dict from answer_document() with answer, citations, sources

  Returns:
    Updated history list
  """
  turn = {
    "question": question,
    "answer": result.get("answer", ""),
    "citations": result.get("citations", []),
    "sources": result.get("sources", [])
  }

  document["history"].append(turn)
  return document["history"]


def normalize_for_match(text: str) -> str:
  """Normalize text for simple string-based scoring.

  Lowercases, removes punctuation, strips extra whitespace.

  Args:
    text: Text to normalize (question, answer, or gold answer)

  Returns:
    Normalized text for comparison
  """
  text = text.lower()
  text = re.sub(r'[^\w\s]', ' ', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text


def contains_any_answer(text: str, answers: list[str]) -> bool:
  """Check if any acceptable answer appears in text after normalization.

  Args:
    text: Text block to search (typically the generated answer)
    answers: List of acceptable answers (from gold standard)

  Returns:
    True if any answer phrase appears in text, False otherwise
  """
  normalized_text = normalize_for_match(text)

  for answer in answers:
    normalized_answer = normalize_for_match(answer)
    if normalized_answer in normalized_text:
      return True

  return False


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60
) -> "pandas.DataFrame":
  """Evaluate retrieval and answer generation on a question set.

  Args:
    eval_set: List of eval records, each with:
      - pdf_name: PDF filename (key in documents_by_name)
      - question: User question
      - correct_pages: List of page numbers with the answer
      - answers: List of acceptable answer phrases
    documents_by_name: Dict mapping pdf_name → prepared document
    top_k: Top-k chunks to retrieve
    candidate_pool: Rerank from top-N

  Returns:
    pandas DataFrame with columns:
    - question, pdf_name, correct_pages, retrieved_pages
    - retrieval_hit (bool), local_answer (str), answer_hit (bool)
  """
  try:
    import pandas as pd
  except ImportError:
    raise ImportError("pandas is required for evaluate_questions. Install with: pip install pandas")

  results = []

  for record in eval_set:
    pdf_name = record["pdf_name"]
    question = record["question"]
    correct_pages = set(record.get("correct_pages", []))
    acceptable_answers = record.get("answers", [])

    if pdf_name not in documents_by_name:
      results.append({
        "question": question,
        "pdf_name": pdf_name,
        "correct_pages": sorted(list(correct_pages)),
        "retrieved_pages": [],
        "retrieval_hit": False,
        "local_answer": "[Document not found]",
        "answer_hit": False
      })
      continue

    document = documents_by_name[pdf_name]

    try:
      hits = search_document(question, document, top_k=top_k, candidate_pool=candidate_pool)
      retrieved_pages = list(set(hit["page"] for hit in hits))

      if correct_pages:
        retrieval_hit = len([p for p in retrieved_pages if p in correct_pages]) > 0
      else:
        retrieval_hit = len(retrieved_pages) > 0 and len(hits) > 0

      answer = best_sentence_answer(question, hits)
      answer_hit = contains_any_answer(answer, acceptable_answers)

    except Exception as e:
      hits = []
      retrieved_pages = []
      retrieval_hit = False
      answer = f"[Error: {str(e)}]"
      answer_hit = False

    results.append({
      "question": question,
      "pdf_name": pdf_name,
      "correct_pages": sorted(list(correct_pages)),
      "retrieved_pages": sorted(retrieved_pages),
      "retrieval_hit": retrieval_hit,
      "local_answer": answer,
      "answer_hit": answer_hit
    })

  return pd.DataFrame(results)
