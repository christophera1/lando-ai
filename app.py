"""
Land Development AI - Chat over uploaded PDFs with page-based retrieval and citations.
Uses Claude Opus for chat (see CHAT_MODEL). Vision ingest uses VISION_MODEL.
Answers use a concise format with sources; strict quote validation is optional (admin debug).
"""

import base64
import os
import re
import json
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import numpy as np
import faiss
import fitz
import torch
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

# --- Constants ---
TOP_K = 8
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
FINDER_PAGE_LIMIT = 6
FINDER_PAGE_LIMIT_TABLE_HEAVY = 18
# Max pages listed in Finder text + returned as finder_pages (UI follows same list).
FINDER_REVIEW_PAGE_CAP = 5
# In Finder Mode, allow scrolling this many pages before/after the suggested LDC anchor page.
FINDER_LDC_SCROLL_RADIUS = 5
NOT_FOUND_MESSAGE = "Regulation not found in this document."
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Main Q&A model (swap back to claude-sonnet-4-6 for faster/cheaper replies).
CHAT_MODEL = "claude-opus-4-7"
# Chart/table transcription at ingest; Sonnet is usually enough and cheaper per page.
VISION_MODEL = "claude-sonnet-4-6"

DATA_DIR = Path("data")
INBOX_DIR = DATA_DIR / "inbox"
INDEX_DIR = DATA_DIR / "index"
ASSETS_DIR = Path("assets")
PREFERRED_LOGO_PATH = ASSETS_DIR / "lando-logo.png"
_asset_images = [
    p for p in sorted(ASSETS_DIR.glob("*"))
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
]
LOGO_PATH = PREFERRED_LOGO_PATH if PREFERRED_LOGO_PATH.exists() else (_asset_images[0] if _asset_images else PREFERRED_LOGO_PATH)
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
# Larger batches use GPU VRAM better during ingest; lower if you hit OOM.
ENCODE_BATCH_SIZE = 128
# Light stopwords for retrieval (no stemming)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "what", "when", "where", "which", "who", "how", "this", "if", "or", "but",
}

# --- Page config ---
st.set_page_config(
    page_title="lando.ai",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "L",
    layout="wide",
)

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []  # For "Retrieved Evidence" expander
if "admin_view" not in st.session_state:
    st.session_state.admin_view = False


def ensure_data_dirs() -> None:
    """Create required data directories if they don't exist."""
    for d in (DATA_DIR, INBOX_DIR, INDEX_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _clean_api_key_candidate(raw: Optional[str]) -> str:
    """Treat empty strings and obvious template values as missing."""
    if raw is None:
        return ""
    k = str(raw).strip().strip('"').strip("'")
    if not k:
        return ""
    low = k.lower()
    if "your_key_here" in low or "changeme" in low or "replace_me" in low:
        return ""
    if k in ("...", "xxx", "YOUR_KEY_HERE"):
        return ""
    return k


def get_api_key() -> str:
    """
    Resolve the Anthropic API key from secrets, environment, local file, or UI (sidebar / banner).
    Streamlit Community Cloud injects Secrets into st.secrets (and sometimes into the environment).
    """
    key = ""
    # 1) Streamlit secrets (Community Cloud + local .streamlit/secrets.toml when present)
    try:
        secrets = st.secrets
        if "ANTHROPIC_API_KEY" in secrets:
            key = _clean_api_key_candidate(str(secrets["ANTHROPIC_API_KEY"]))
    except Exception:
        key = ""

    # 2) Environment (some hosts mirror Secrets here)
    if not key:
        key = _clean_api_key_candidate(os.environ.get("ANTHROPIC_API_KEY"))

    # 3) Fallback: load from .streamlit/secrets.toml next to this script (works if run from wrong cwd)
    if not key:
        secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                text = secrets_path.read_text(encoding="utf-8")
                m = re.search(r"ANTHROPIC_API_KEY\s*=\s*[\"']([^\"']+)[\"']", text)
                if m:
                    key = _clean_api_key_candidate(m.group(1))
            except Exception:
                pass

    if not key:
        # Session-only key (banner form or Admin sidebar); never log or echo this value.
        key = _clean_api_key_candidate(
            str(
                st.session_state.get("_lando_session_api_key")
                or st.session_state.get("sidebar_api_key")
                or st.session_state.get("api_key", "")
                or "",
            )
        )

    return key


def _has_server_or_file_api_key() -> bool:
    """True if a usable key is configured via secrets file / st.secrets / env (not only UI paste)."""
    try:
        secrets = st.secrets
        if "ANTHROPIC_API_KEY" in secrets and _clean_api_key_candidate(str(secrets["ANTHROPIC_API_KEY"])):
            return True
    except Exception:
        pass
    if _clean_api_key_candidate(os.environ.get("ANTHROPIC_API_KEY")):
        return True
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            text = secrets_path.read_text(encoding="utf-8")
            m = re.search(r"ANTHROPIC_API_KEY\s*=\s*[\"']([^\"']+)[\"']", text)
            if m and _clean_api_key_candidate(m.group(1)):
                return True
        except Exception:
            pass
    return False


def get_app_access_password() -> str:
    """
    Optional gate: if APP_ACCESS_PASSWORD is set in st.secrets, the whole app is locked
    until the user enters it (use on Streamlit Cloud even when the app URL is "public").
    """
    try:
        if "APP_ACCESS_PASSWORD" in st.secrets:
            return str(st.secrets["APP_ACCESS_PASSWORD"]).strip()
    except Exception:
        pass
    return ""


def require_app_access() -> None:
    """Stop the script with a login form until APP_ACCESS_PASSWORD matches (if configured)."""
    pwd = get_app_access_password()
    if not pwd:
        return
    if st.session_state.get("_lando_access_ok"):
        return
    st.title("lando.ai")
    st.caption("This deployment is password-protected.")
    entered = st.text_input("Access password", type="password", key="_lando_gate_pw")
    if st.button("Continue", type="primary"):
        a = (entered or "").strip().encode("utf-8")
        b = pwd.encode("utf-8")
        if len(a) == len(b) and hmac.compare_digest(a, b):
            st.session_state._lando_access_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def render_missing_api_key_banner() -> None:
    """Explain how to enable chat when ANTHROPIC_API_KEY is missing (typical on Streamlit Cloud)."""
    if get_api_key():
        return
    st.error("Chat is off: add your Anthropic API key to Streamlit **Secrets**.")
    st.markdown(
        """
1. Open your app on **[Streamlit Community Cloud](https://share.streamlit.io)**.
2. Click **⋮ Manage app** → **Settings** → **Secrets**.
3. Paste (replace with your real key):

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

4. Click **Save**, then **⋮ Manage app** → **Reboot app**.

**Also set app visibility to Private** (Manage app → **Settings** → **App visibility**) so random visitors cannot open your URL.

**Optional — extra lock:** add a line to Secrets so only people who know it can use the app even if the URL is shared:

```toml
APP_ACCESS_PASSWORD = "choose-a-strong-password"
```

Then redeploy / reboot once.
        """
    )
    st.divider()
    st.markdown(
        "**Quick fix (this browser only)** — paste your real `sk-ant-…` key and click **Save key for this session** "
        "(Streamlit may drop a plain text field on rerun; this stores it in session until you close the tab)."
    )
    with st.form("lando_session_api_key_form", clear_on_submit=False):
        _tmp_key = st.text_input(
            "Anthropic API key",
            type="password",
            label_visibility="visible",
            help="Prefer Streamlit Secrets for production. This path is for quick testing.",
        )
        if st.form_submit_button("Save key for this session"):
            cleaned = _clean_api_key_candidate(_tmp_key)
            if cleaned:
                st.session_state["_lando_session_api_key"] = cleaned
                st.success("Key saved for this session. Scroll down to chat.")
                st.rerun()
            else:
                st.error("That does not look like a real key (too short or still a placeholder).")
    st.caption(
        "Troubleshooting: the secret name must be exactly **ANTHROPIC_API_KEY** (all caps, underscores). "
        "Remove placeholder text like YOUR_KEY_HERE. After editing Secrets, use **Reboot app** once."
    )


def resolve_embedding_device() -> str:
    """Use CUDA for MiniLM encode/index when PyTorch was built with GPU support."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@st.cache_resource(show_spinner=False)
def get_embedding_model(device: str) -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def hash_pdf(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chunks() -> List[Dict[str, Any]]:
    return load_json(CHUNKS_PATH, default=[])


def save_chunks(chunks: List[Dict[str, Any]]) -> None:
    save_json(CHUNKS_PATH, chunks)


def load_manifest() -> Dict[str, Any]:
    return load_json(MANIFEST_PATH, default={})


def save_manifest(manifest: Dict[str, Any]) -> None:
    save_json(MANIFEST_PATH, manifest)


def chunk_text(
    text: str,
    max_chars: int = 3000,
    overlap: int = 300,
) -> List[str]:
    """Simple sliding-window chunking over characters."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def extract_page_text_pymupdf(page: fitz.Page) -> str:
    """
    Extract text with spatial sort (better than naive order for many zoning/engineering PDFs).
    Falls back to block text if the sorted dump is empty.
    """
    text = (page.get_text("text", sort=True) or "").strip()
    if text:
        return text
    parts: List[str] = []
    for block in page.get_text("blocks") or []:
        if len(block) >= 5 and isinstance(block[4], str) and block[4].strip():
            parts.append(block[4].strip())
    return "\n".join(parts)


def page_has_meaningful_images(page: fitz.Page, min_cover: float = 0.02) -> bool:
    """True if embedded images cover at least min_cover of page area (filters tiny logos)."""
    infos = page.get_images(full=True)
    if not infos:
        return False
    page_area = page.rect.width * page.rect.height
    if page_area < 1:
        return True
    for info in infos:
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            return True
        for r in rects:
            if (r.width * r.height) / page_area >= min_cover:
                return True
    return False


def should_enrich_page_with_vision(page: fitz.Page, text: str) -> bool:
    """Run vision when the page likely has figures or text extraction looks too thin."""
    t = text.strip()
    if page_has_meaningful_images(page):
        return True
    if len(t) < 120:
        return True
    # Charts/maps drawn as vectors (no embedded bitmap) still need a render pass.
    try:
        if len(page.get_drawings() or []) > 100:
            return True
    except Exception:
        pass
    return False


def enrich_page_visual_with_claude(
    client: Anthropic,
    page: fitz.Page,
    *,
    doc_name: str,
    page_num: int,
) -> str:
    """Render page to PNG and transcribe charts/tables/maps for RAG."""
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    prompt = (
        "You are indexing a land development / zoning / civil engineering PDF.\n"
        "Transcribe information from charts, graphs, tables, maps, and diagrams: "
        "titles, axis labels, units, legends, and important numbers. Use bullet points. "
        "Be faithful to what is visible; do not invent data.\n"
        "If the page has no charts, tables, maps, or diagrams (body text only), reply exactly: NO_FIGURES"
    )
    resp = client.messages.create(
        model=VISION_MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    body = (resp.content[0].text or "").strip()
    if not body or body.upper().startswith("NO_FIGURES"):
        return ""
    return (
        f"[Chart/figure transcription from page render — {doc_name}, page {page_num}]\n"
        f"{body}"
    )


def ingest_pdfs(
    vision_enrich: bool = False,
    vision_mode: str = "auto",
    force_reingest: bool = False,
    api_key: str = "",
) -> Dict[str, Any]:
    """
    Ingest PDFs from INBOX_DIR, updating FAISS index and metadata.
    Computes embeddings only for new/changed PDFs.
    """
    ensure_data_dirs()
    model = get_embedding_model(resolve_embedding_device())
    embed_dim = model.get_sentence_embedding_dimension()

    inbox_pdfs = sorted(INBOX_DIR.glob("*.pdf"))
    manifest_old = load_manifest()
    chunks_old = load_chunks()

    file_hashes: Dict[str, str] = {}
    file_timestamps: Dict[str, str] = {}
    for path_str, meta in manifest_old.items():
        file_hashes[path_str] = meta.get("hash", "")
        file_timestamps[path_str] = meta.get("timestamp", "")

    now_iso = datetime.utcnow().isoformat() + "Z"

    new_or_changed_paths: List[Path] = []
    hashes_new: Dict[str, str] = {}
    for pdf_path in inbox_pdfs:
        path_str = str(pdf_path.resolve())
        h = hash_pdf(pdf_path)
        hashes_new[path_str] = h
        if force_reingest or path_str not in manifest_old or manifest_old[path_str].get("hash") != h:
            new_or_changed_paths.append(pdf_path)

    # Early exit: nothing to update
    if not new_or_changed_paths and chunks_old:
        return {
            "updated_files": [],
            "skipped_files": [str(p.resolve()) for p in inbox_pdfs],
            "total_chunks": len(chunks_old),
            "last_updated": max(file_timestamps.values()) if file_timestamps else now_iso,
        }

    # Load existing index and reconstruct embeddings (if present)
    old_embeddings: np.ndarray
    if FAISS_INDEX_PATH.exists() and chunks_old:
        index_old = faiss.read_index(str(FAISS_INDEX_PATH))
        old_embeddings = index_old.reconstruct_n(0, index_old.ntotal)
    else:
        old_embeddings = np.zeros((0, embed_dim), dtype="float32")

    # Keep chunks and embeddings for unchanged files
    new_chunks: List[Dict[str, Any]] = []
    new_embeddings_list: List[np.ndarray] = []
    for i, ch in enumerate(chunks_old):
        file_path = ch.get("file_path")
        if not file_path:
            continue
        if any(str(p.resolve()) == file_path for p in new_or_changed_paths):
            continue
        new_chunks.append(ch)
        if i < len(old_embeddings):
            new_embeddings_list.append(old_embeddings[i])

    # Rebuild manifest entries as we go
    manifest_new: Dict[str, Any] = {}

    # Helper to register chunk indices per file
    def register_chunk(file_path: str, idx: int, hash_value: str) -> None:
        entry = manifest_new.setdefault(
            file_path,
            {
                "hash": hash_value,
                "chunk_indices": [],
                "timestamp": file_timestamps.get(file_path, now_iso),
            },
        )
        entry["chunk_indices"].append(idx)

    # First, register existing chunks (unchanged files)
    for idx, ch in enumerate(new_chunks):
        file_path = ch["file_path"]
        hash_value = hashes_new.get(file_path, file_hashes.get(file_path, ""))
        register_chunk(file_path, idx, hash_value)

    # Ingest new/changed files
    updated_files: List[str] = []
    for pdf_path in new_or_changed_paths:
        path_str = str(pdf_path.resolve())
        doc_name = pdf_path.name
        client: Optional[Anthropic] = None
        if vision_enrich and api_key.strip():
            try:
                client = Anthropic(api_key=api_key.strip())
            except Exception:
                client = None

        try:
            doc_fitz = fitz.open(str(pdf_path))
        except Exception:
            continue

        file_chunks: List[Dict[str, Any]] = []
        try:
            for page_index in range(len(doc_fitz)):
                page = doc_fitz[page_index]
                page_num = page_index + 1
                text = extract_page_text_pymupdf(page)
                if client:
                    try:
                        should_run_vision = (
                            vision_mode == "all"
                            or should_enrich_page_with_vision(page, text)
                        )
                        if should_run_vision:
                            extra = enrich_page_visual_with_claude(
                                client, page, doc_name=doc_name, page_num=page_num
                            )
                            if extra:
                                text = (text + "\n\n" + extra).strip()
                    except Exception:
                        pass
                for local_chunk_id, chunk_str in enumerate(
                    chunk_text(text, max_chars=3500, overlap=300)
                ):
                    file_chunks.append(
                        {
                            "doc_name": doc_name,
                            "file_path": path_str,
                            "page_num": page_num,
                            "chunk_id": local_chunk_id,
                            "text": chunk_str,
                        }
                    )
        finally:
            doc_fitz.close()

        if not file_chunks:
            continue

        texts = [c["text"] for c in file_chunks]
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=ENCODE_BATCH_SIZE,
        )
        embeddings = embeddings.astype("float32")

        start_idx = len(new_chunks)
        new_chunks.extend(file_chunks)
        new_embeddings_list.append(embeddings)

        for offset in range(len(file_chunks)):
            idx = start_idx + offset
            register_chunk(path_str, idx, hashes_new.get(path_str, ""))

        file_timestamps[path_str] = now_iso
        updated_files.append(path_str)

    # Flatten embeddings and build/save FAISS index
    if new_embeddings_list:
        all_embeddings = (
            np.vstack(
                [
                    e if isinstance(e, np.ndarray) and e.ndim == 2 else np.atleast_2d(e)
                    for e in new_embeddings_list
                ]
            ).astype("float32")
        )
        index = faiss.IndexFlatL2(embed_dim)
        index.add(all_embeddings)
        faiss.write_index(index, str(FAISS_INDEX_PATH))

    save_chunks(new_chunks)
    save_manifest(manifest_new)

    return {
        "updated_files": updated_files,
        "skipped_files": [
            str(p.resolve())
            for p in inbox_pdfs
            if str(p.resolve()) not in updated_files
        ],
        "total_chunks": len(new_chunks),
        "last_updated": now_iso,
    }


def build_context_block(retrieved: list[dict]) -> str:
    """Format retrieved chunks as [CONTEXT] block for the model."""
    lines = ["[CONTEXT]"]
    for p in retrieved:
        doc_name = p.get("doc_name", "Document")
        lines.append(f"({doc_name} - Page {p['page_num']})")
        lines.append(p["text"].strip() or "(no text)")
        lines.append("")
    return "\n".join(lines)


def _normalize_whitespace(s: str) -> str:
    """Normalize runs of whitespace to single space and strip."""
    return " ".join(s.split())


def build_page_text_map(retrieved: List[Dict[str, Any]]) -> Dict[Tuple[str, int], str]:
    """Build (doc_name, page_num) -> concatenated page text from retrieved chunks."""
    out: Dict[Tuple[str, int], List[str]] = {}
    for r in retrieved:
        doc_name = r.get("doc_name", "Document")
        page_num = int(r.get("page_num", 0))
        text = (r.get("text") or "").strip()
        key = (doc_name, page_num)
        out.setdefault(key, []).append(text)
    return {k: " ".join(v) for k, v in out.items()}


def _extract_search_terms(question: str, max_terms: int = 8) -> List[str]:
    """Extract simple keyword hints from user question for Finder Mode."""
    terms = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question.lower())
    filtered = [t for t in terms if t not in STOPWORDS]
    # Keep order, dedupe
    seen = set()
    out: List[str] = []
    for t in filtered:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _finder_table_heavy_query(question: str) -> bool:
    """True when question likely needs wide table/chart coverage (e.g. parking schedules)."""
    q = question.lower()
    keys = (
        "parking",
        "stall",
        "space",
        "vehicle",
        "chart",
        "table",
        "schedule",
        "non-residential",
        "nonresidential",
        "square foot",
        "sq ft",
        "sf ",
        "gsf",
        "gross floor",
        "5000",
        "restaurant",
    )
    return any(k in q for k in keys)


def _finder_keyword_terms(question: str) -> List[str]:
    """Terms for full-index keyword page discovery (complements embedding retrieval)."""
    terms = _extract_search_terms(question, max_terms=16)
    q = question.lower()
    extra: List[str] = []
    if any(w in q for w in ("parking", "stall", "space", "car", "vehicle")):
        extra += [
            "parking",
            "park",
            "space",
            "spaces",
            "stall",
            "stalls",
            "vehicle",
            "non-residential",
            "nonresidential",
            "non residential",
            "schedule",
            "chart",
            "table",
            "required",
            "minimum",
            "ratio",
        ]
    if any(w in q for w in ("restaurant", "food", "eating", "drinking", "kitchen")):
        extra += ["restaurant", "eating", "drinking", "food", "service", "commercial"]
    nums = re.findall(r"\b\d[\d,]{0,8}\b", q)
    extra += [n.replace(",", "") for n in nums]
    out: List[str] = []
    seen = set()
    for t in terms + extra:
        t = t.strip().lower()
        if len(t) < 2:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:28]


def merge_keyword_page_candidates(
    question: str,
    chunks: List[Dict[str, Any]],
    semantic_rows: List[Dict[str, Any]],
    *,
    max_keyword_pages: int = 42,
) -> List[Dict[str, Any]]:
    """
    Merge semantic retrieval with keyword-scored pages so long tables/charts
    still surface in Finder Mode when embeddings miss the right row.
    """
    terms = _finder_keyword_terms(question)
    if not terms:
        return list(semantic_rows)

    page_best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for ch in chunks:
        fp = (ch.get("file_path") or "").strip()
        page = int(ch.get("page_num", 0))
        if not fp or page <= 0:
            continue
        text_l = (ch.get("text") or "").lower()
        if not text_l:
            continue
        score = sum(1 for t in terms if t in text_l)
        if score <= 0:
            continue
        key = (fp, page)
        snippet_full = (ch.get("text") or "").strip()
        snippet = snippet_full[:420].rstrip() + ("..." if len(snippet_full) > 420 else "")
        doc_name = ch.get("doc_name", Path(fp).name)
        prev = page_best.get(key)
        if not prev or score > int(prev.get("kw_score", 0)):
            page_best[key] = {
                "doc_name": doc_name,
                "file_path": fp,
                "page_num": page,
                "chunk_id": -1,
                "text": snippet,
                "kw_score": score,
            }

    if not page_best:
        return list(semantic_rows)

    base = max((float(r.get("distance", 0.0)) for r in semantic_rows), default=1.0)
    kw_sorted = sorted(
        page_best.items(),
        key=lambda kv: int(kv[1].get("kw_score", 0)),
        reverse=True,
    )[:max_keyword_pages]

    merged_rows: List[Dict[str, Any]] = []
    merged_rows.extend(semantic_rows)
    for rank, ((_fp, _page), info) in enumerate(kw_sorted):
        score = int(info.get("kw_score", 0))
        fp = str(info.get("file_path", "")).strip()
        page = int(info.get("page_num", 0))
        distance = base + 2.0 + (rank * 0.05) - min(score, 30) * 0.05
        merged_rows.append(
            {
                "doc_name": info.get("doc_name", Path(fp).name),
                "file_path": fp,
                "page_num": page,
                "chunk_id": -1,
                "text": info.get("text", ""),
                "distance": distance,
            }
        )

    dedup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in merged_rows:
        fp = row.get("file_path")
        page = int(row.get("page_num", 0))
        if not fp or page <= 0:
            continue
        key = (str(fp).strip(), page)
        d = float(row.get("distance", 1e9))
        if key not in dedup or d < float(dedup[key].get("distance", 1e9)):
            dedup[key] = {k: v for k, v in row.items() if k != "kw_score"}

    return sorted(dedup.values(), key=lambda x: float(x.get("distance", 1e9)))


def build_finder_mode_result(
    question: str,
    retrieved: List[Dict[str, Any]],
    max_pages: int = FINDER_PAGE_LIMIT,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build Finder Mode response with best pages to inspect manually.
    This is used when high-confidence validation fails.
    """
    if not retrieved:
        text = (
            "**Confidence: Low (Finder Mode)**\n\n"
            "I could not find enough reliable context to answer directly.\n\n"
            "Start your search by uploading/indexing more relevant PDFs, then ask again."
        )
        return text, []

    review_cap = min(max_pages, FINDER_REVIEW_PAGE_CAP)
    page_candidates: List[Dict[str, Any]] = []
    seen_pages: set = set()
    for r in sorted(retrieved, key=lambda x: float(x.get("distance", 1e9))):
        file_path = r.get("file_path")
        page_num = int(r.get("page_num", 0))
        doc_name = r.get("doc_name", "Document")
        if not file_path or page_num <= 0:
            continue
        key = (file_path, page_num)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        snippet = (r.get("text") or "").strip()
        if len(snippet) > 380:
            snippet = snippet[:380].rstrip() + "..."
        page_candidates.append(
            {
                "doc_name": doc_name,
                "file_path": file_path,
                "page_num": page_num,
                "distance": float(r.get("distance", 1e9)),
                "snippet": snippet,
            }
        )
        if len(page_candidates) >= review_cap:
            break

    keyword_hints = _extract_search_terms(question)
    keyword_line = ", ".join(keyword_hints) if keyword_hints else "(no strong keywords extracted)"

    lines = [
        "**Confidence: Low (Finder Mode)**",
        "",
        "I cannot confirm an exact answer from retrieved text.",
        "Start your search on these likely pages and scan for the keywords below:",
        "",
        f"**Suggested keywords:** {keyword_line}",
        "",
    ]
    if _finder_table_heavy_query(question):
        lines.append(
            "_Tip: long parking or use tables often span several PDF pages — if the row you need "
            "is not obvious in a preview, scroll the adjacent page numbers in the same document._"
        )
        lines.append("")
    lines.append("**Best pages to review now:**")
    for p in page_candidates:
        lines.append(
            f"- `{p['doc_name']}` page {p['page_num']} "
            f"(retrieval score {p['distance']:.4f})"
        )
    return "\n".join(lines), page_candidates


@st.cache_data(show_spinner=False)
def render_pdf_page_png(file_path: str, page_num: int, zoom: float = 1.6) -> bytes:
    """Render a PDF page as PNG bytes for in-app reading."""
    doc = fitz.open(file_path)
    try:
        idx = max(0, page_num - 1)
        page = doc[idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def pdf_page_count(file_path: str) -> int:
    """Return total page count for a PDF, or 0 on failure."""
    try:
        doc = fitz.open(file_path)
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return 0


def _is_ldc_finder_entry(entry: Dict[str, Any]) -> bool:
    dn = (entry.get("doc_name") or "").lower()
    fp = (entry.get("file_path") or "").lower().replace("\\", "/")
    return "ldc" in dn or "ldc_" in fp or "/ldc" in fp


def pick_ldc_anchor_from_finder_pages(
    finder_pages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick best LDC row to anchor the ±page browser (prefer chart-transcribed snippet)."""
    ldc_hits = [p for p in finder_pages if _is_ldc_finder_entry(p)]
    if not ldc_hits:
        return None
    chart_mark = "[chart/figure transcription from page render"
    for p in ldc_hits:
        sn = (p.get("snippet") or "").lower()
        if chart_mark in sn:
            return p
    return ldc_hits[0]


def render_finder_mode_pages_ui(
    finder_pages: List[Dict[str, Any]],
    *,
    key_prefix: str,
    expanded: bool = True,
) -> None:
    """LDC ±N page browser plus optional thumbnails for other Finder hits."""
    if not finder_pages:
        return
    with st.expander("🔎 Finder Mode — review the right PDF pages", expanded=expanded):
        anchor = pick_ldc_anchor_from_finder_pages(finder_pages)
        if anchor:
            fp_path = str(anchor.get("file_path") or "").strip()
            anchor_pg = int(anchor.get("page_num") or 0)
            doc_name = anchor.get("doc_name", "LDC")
            n_pages = pdf_page_count(fp_path)
            if fp_path and anchor_pg > 0 and n_pages > 0:
                lo = max(1, anchor_pg - FINDER_LDC_SCROLL_RADIUS)
                hi = min(n_pages, anchor_pg + FINDER_LDC_SCROLL_RADIUS)
                st.markdown(
                    f"**LDC chart area** — `{doc_name}` · anchor **page {anchor_pg}** "
                    f"(browse **pages {lo}–{hi}** in this part)."
                )
                default_pg = min(max(anchor_pg, lo), hi)
                sel = st.slider(
                    "Page in this LDC part",
                    min_value=lo,
                    max_value=hi,
                    value=default_pg,
                    key=f"{key_prefix}_ldc_scroll",
                    help=f"Scroll ±{FINDER_LDC_SCROLL_RADIUS} pages around the suggested chart/table.",
                )
                try:
                    st.image(render_pdf_page_png(fp_path, int(sel)), width="stretch")
                except Exception:
                    st.caption("Could not render this page.")
                st.divider()
            else:
                st.caption("Could not open LDC PDF for page browser.")
        else:
            st.caption(
                "No LDC document in the top Finder hits — open thumbnails below, "
                "or add the right LDC part to the index."
            )
        with st.expander("All suggested pages (thumbnails)", expanded=False):
            for idx, fp in enumerate(finder_pages[:14]):
                st.markdown(f"**{fp.get('doc_name', 'Document')} - Page {fp.get('page_num')}**")
                st.caption(f"File: {fp.get('file_path', '')}")
                snippet = fp.get("snippet", "")
                if snippet:
                    st.text(snippet)
                try:
                    page_png = render_pdf_page_png(fp["file_path"], int(fp["page_num"]))
                    st.image(page_png, width="stretch")
                except Exception:
                    st.caption("Could not render this page preview.")
                st.divider()


def _context_has_subsection_ref(retrieved: List[Dict[str, Any]]) -> bool:
    """Return True if any retrieved chunk text contains a subsection-style reference."""
    subsection_pattern = re.compile(
        r"\b(?:Sec\.|Section)\s*\d+[\s\-\.]*(?:\d+[\s\-\.]*)*(?:\([a-z]\))?\b",
        re.IGNORECASE,
    )
    for r in retrieved:
        if subsection_pattern.search(r.get("text") or ""):
            return True
    return False


def validate_engineer_rule(
    response: str,
    retrieved_doc_pages: Dict[str, List[int]],
    retrieved_chunks: List[Dict[str, Any]],
) -> tuple[str, bool, List[Tuple[str, int]]]:
    """
    Enforce ENGINEER RULE with code-level checks.
    Validates quote/citation pairs independently; only fails when no valid pair exists.
    - Requires at least one quote that appears verbatim on a valid cited page.
    - Ignores extra/duplicate citations and minor format deviations; does not fail on them.
    """
    response_stripped = response.strip()
    if re.search(r"regulation\s+not\s+found|not\s+found\s+in\s+this\s+document", response_stripped, re.IGNORECASE):
        return NOT_FOUND_MESSAGE, True, []
    if response_stripped == NOT_FOUND_MESSAGE:
        return NOT_FOUND_MESSAGE, True, []

    page_text_map = build_page_text_map(retrieved_chunks)
    context_has_subsection = _context_has_subsection_ref(retrieved_chunks)

    quote_pattern = re.compile(r'"([^"]{20,})"')
    quote_matches = list(quote_pattern.finditer(response))

    # Parse all citations; keep only valid (doc_name, page_num) in retrieved_doc_pages
    citation_pattern = re.compile(
        r"\[(?P<name>[^\]]+?),\s*Page\s+(?P<page>\d+)\]"
    )
    all_citation_matches = list(citation_pattern.finditer(response))
    valid_cited_pages: set = set()
    detected_citations: List[Tuple[str, int]] = []
    has_subsection_citation = False

    for m in all_citation_matches:
        name_part = m.group("name").strip()
        try:
            page_num = int(m.group("page"))
        except ValueError:
            continue
        doc_name = name_part.split(", Sec.")[0].strip() if ", Sec." in name_part else name_part
        if ", Sec." in name_part or "Sec." in name_part:
            has_subsection_citation = True
        if doc_name not in retrieved_doc_pages or page_num not in retrieved_doc_pages[doc_name]:
            continue
        key = (doc_name, page_num)
        valid_cited_pages.add(key)
        detected_citations.append((doc_name, page_num))

    # Only fail if there are no valid citations at all
    if not valid_cited_pages:
        return NOT_FOUND_MESSAGE, False, []

    # Only fail if there are no quotes at all
    if not quote_matches:
        return NOT_FOUND_MESSAGE, False, detected_citations

    # When context has subsection, require at least one citation to include subsection (any match counts)
    if context_has_subsection and not has_subsection_citation:
        return NOT_FOUND_MESSAGE, False, detected_citations

    # Per-page quote verification: at least one quote must appear on at least one valid cited page
    at_least_one_quote_on_cited_page = False
    for qm in quote_matches:
        quote_text = _normalize_whitespace(qm.group(1))
        if not quote_text:
            continue
        for key in valid_cited_pages:
            page_text = page_text_map.get(key, "")
            if quote_text in _normalize_whitespace(page_text):
                at_least_one_quote_on_cited_page = True
                break
        if at_least_one_quote_on_cited_page:
            break

    if not at_least_one_quote_on_cited_page:
        return NOT_FOUND_MESSAGE, False, detected_citations

    # Governing sources: each listed source must have at least one valid citation
    governing_section = re.search(
        r"(?si)Governing\s+Sources\s*:?\s*(.*?)(?=Answer:|Conflict:|Quote:|Citations:|\Z)",
        response_stripped,
    )
    if governing_section:
        section_text = governing_section.group(1)
        governing_sources_listed = [
            doc_name for doc_name in retrieved_doc_pages
            if doc_name in section_text
        ]
        cited_docs = {name for name, _ in detected_citations}
        for doc_name in governing_sources_listed:
            if doc_name not in cited_docs:
                return NOT_FOUND_MESSAGE, False, detected_citations

    return response_stripped, True, detected_citations


def extract_citations_from_response(
    response: str,
    retrieved_doc_pages: Dict[str, List[int]],
) -> List[Tuple[str, int]]:
    """
    Parse [doc_name, Page N] citations and keep only those that match retrieved context pages.
    Used for source list when running in relaxed (non-gating) validation mode.
    """
    citation_pattern = re.compile(
        r"\[(?P<name>[^\]]+?),\s*Page\s+(?P<page>\d+)\]",
        re.IGNORECASE,
    )
    seen: set = set()
    out: List[Tuple[str, int]] = []
    for m in citation_pattern.finditer(response):
        name_part = m.group("name").strip()
        try:
            page_num = int(m.group("page"))
        except ValueError:
            continue
        low = name_part.lower()
        if ", sec." in low:
            doc_name = name_part[: low.index(", sec.")].strip()
        else:
            doc_name = name_part.strip()
        if doc_name not in retrieved_doc_pages or page_num not in retrieved_doc_pages[doc_name]:
            continue
        key = (doc_name, page_num)
        if key in seen:
            continue
        seen.add(key)
        out.append((doc_name, page_num))
    return out


# --- Sidebar: Ingestion + status (admin only) ---
require_app_access()
ensure_data_dirs()
inbox_pdfs = sorted(INBOX_DIR.glob("*.pdf"))
render_missing_api_key_banner()

# --- Main: Chat ---
st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(165deg, #f8fafc 0%, #eef2ff 38%, #f0fdfa 100%);
      }
      section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
      }
      .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
        max-width: 56rem;
      }
      [data-testid="stChatMessage"] {
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 16px;
        padding: 0.65rem 0.9rem 0.85rem 0.9rem !important;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 2px 14px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.65rem !important;
        backdrop-filter: blur(6px);
      }
      [data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 1px solid rgba(15, 23, 42, 0.1) !important;
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.06);
        background: rgba(255, 255, 255, 0.95) !important;
      }
      [data-testid="stExpander"] details {
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.55);
        overflow: hidden;
      }
      [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        letter-spacing: -0.01em;
      }
      .lando-hero-wrap {
        max-width: 920px;
        margin: 8px auto 20px auto;
      }
      .lando-hero {
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        padding: 22px 24px 20px 24px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: linear-gradient(
          145deg,
          #f0fdf4 0%,
          #ecfeff 42%,
          #eff6ff 100%
        );
        box-shadow:
          0 1px 2px rgba(15, 23, 42, 0.04),
          0 12px 40px -12px rgba(15, 118, 110, 0.18),
          0 8px 24px -16px rgba(37, 99, 235, 0.12);
      }
      .lando-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(
          900px 240px at 10% -20%,
          rgba(45, 212, 191, 0.22),
          transparent 55%
        );
        pointer-events: none;
      }
      .lando-hero-inner {
        position: relative;
        z-index: 1;
      }
      .lando-hero-kicker {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #0f766e;
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(15, 118, 110, 0.2);
        border-radius: 999px;
        padding: 5px 12px;
        margin-bottom: 12px;
      }
      .lando-hero-title {
        font-size: clamp(1.35rem, 2.6vw, 1.75rem);
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
        color: #0f172a;
        margin: 0 0 10px 0;
      }
      .lando-hero-sub {
        font-size: 1.02rem;
        line-height: 1.55;
        color: #334155;
        max-width: 52rem;
        margin: 0 0 16px 0;
      }
      .lando-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .lando-chip {
        font-size: 0.84rem;
        font-weight: 500;
        color: #0f172a;
        border-radius: 999px;
        padding: 6px 14px;
        border: 1px solid rgba(51, 65, 85, 0.14);
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 1px 0 rgba(255, 255, 255, 0.9) inset;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([5, 1])
with top_left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=200)
        st.caption("Cited answers over your uploaded LDC, code, and civil PDFs.")
    else:
        st.title("lando.ai")
        st.caption("Land development research with sources on every answer.")
with top_right:
    st.caption("View")
    st.session_state.admin_view = st.toggle(
        "Admin",
        value=bool(st.session_state.get("admin_view", False)),
        help="Show ingestion, API key, debug, and retrieval internals.",
    )

st.markdown(
    """
    <div class="lando-hero-wrap">
      <div class="lando-hero">
        <div class="lando-hero-inner">
          <div class="lando-hero-kicker">PDF-grounded · Citations on every answer</div>
          <div class="lando-hero-title">Land Development Research Assistant</div>
          <div class="lando-hero-sub">
            Ask in plain language. You will get a careful answer with quoted sources,
            or Finder Mode with the exact PDF pages to open when the text alone is not enough.
          </div>
          <div class="lando-chip-row">
            <span class="lando-chip">Code &amp; LDC lookup</span>
            <span class="lando-chip">Source-first answers</span>
            <span class="lando-chip">Finder Mode previews</span>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.admin_view:
    with st.sidebar:
        st.markdown("### Index & ingest")
        _ed = resolve_embedding_device()
        if _ed == "cuda" and torch.cuda.is_available():
            st.caption(f"Embeddings device: GPU — {torch.cuda.get_device_name(0)}")
        else:
            st.caption(
                "Embeddings device: CPU. For GPU indexing, install PyTorch with CUDA "
                "([pytorch.org](https://pytorch.org/get-started/locally/)) matching your driver, then restart."
            )
        if inbox_pdfs:
            st.caption("PDFs found in `data/inbox`:")
            for p in inbox_pdfs:
                st.write(f"- {p.name}")
        else:
            st.caption("No PDFs found in `data/inbox`. Place PDFs there to index them.")

        st.checkbox(
            "Transcribe charts with vision (Claude)",
            value=False,
            key="vision_ingest",
            help=(
                "For pages with figures or little text, renders the page as an image and asks Claude "
                "to transcribe charts, tables, and maps. Uses your API key and adds cost per page processed."
            ),
        )
        st.checkbox(
            "Force vision on every page (best chart coverage, slower/costlier)",
            value=False,
            key="vision_force_all_pages",
            help=(
                "Runs Claude vision on every page during ingest, even if heuristics do not "
                "detect a chart/table. Useful for hard-to-read chart pages."
            ),
        )
        st.checkbox(
            "Force re-ingest all PDFs in inbox",
            value=False,
            key="force_reingest",
            help=(
                "Rebuilds embeddings/chunks for every PDF in data/inbox even if file hashes "
                "have not changed."
            ),
        )
        st.checkbox(
            "Navigation-first mode (show pages, skip direct answer)",
            value=False,
            key="navigation_first_mode",
            help=(
                "Prioritize finding and showing the best PDF pages to read. "
                "When enabled, chat responses skip direct legal conclusions and go straight to Finder Mode."
            ),
        )

        if st.button("Ingest / Update Index", use_container_width=True):
            key_for_ingest = get_api_key()
            want_vision = bool(st.session_state.get("vision_ingest"))
            force_all_vision_pages = bool(st.session_state.get("vision_force_all_pages"))
            force_reingest = bool(st.session_state.get("force_reingest"))
            if want_vision and not key_for_ingest:
                st.warning("Add your Anthropic API key (sidebar) to use chart transcription.")
            with st.spinner(
                "Ingesting PDFs and updating index..."
                + (" (vision may take a while)" if want_vision and key_for_ingest else "")
            ):
                ingest_info = ingest_pdfs(
                    vision_enrich=want_vision and bool(key_for_ingest),
                    vision_mode="all" if force_all_vision_pages and want_vision else "auto",
                    force_reingest=force_reingest,
                    api_key=key_for_ingest or "",
                )
            st.success("Index updated.")
            st.write(f"Updated files: {len(ingest_info.get('updated_files', []))}")
            st.write(f"Total chunks indexed: {ingest_info.get('total_chunks', 0)}")
            st.write(f"Last updated: {ingest_info.get('last_updated')}")
        else:
            manifest = load_manifest()
            chunks_meta = load_chunks()
            if manifest:
                last_updated = max(
                    (v.get("timestamp", "") for v in manifest.values()),
                    default="N/A",
                )
                st.caption(f"Indexed files: {len(manifest)}")
                st.caption(f"Chunks indexed: {len(chunks_meta)}")
                st.caption(f"Last updated: {last_updated}")
            else:
                st.caption("No index built yet.")

with st.expander("✨ Tips for stronger questions", expanded=False):
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Name the doc**")
        st.caption("Mention the instrument or part, e.g. `LDC Part 4 stormwater fees`.")
    with t2:
        st.markdown("**Ask for pages**")
        st.caption("When unsure, ask where to read, e.g. `show me the pages to review`.")
    with t3:
        st.markdown("**Tables & charts**")
        st.caption("Add words like `table`, `chart`, `fee`, or `schedule` for figure-heavy pages.")

# Show chat history
for msg_i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if st.session_state.admin_view and msg.get("retrieved"):
        with st.expander("📎 Retrieved Evidence (this turn)"):
            for p in msg["retrieved"]:
                snippet = (p["text"].strip() or "(no text)")[:400]
                if len(p["text"].strip() or "") > 400:
                    snippet += "..."
                st.markdown(f"**Page {p['page_num']}**")
                st.text(snippet)
                st.divider()
    if st.session_state.admin_view and msg.get("validation_debug"):
        with st.expander("🔍 Debug (validation)"):
            vd = msg["validation_debug"]
            st.write(f"Validation passed: {vd.get('validation_passed')}")
            st.write(f"Citations: {vd.get('citations')}")
            st.write(f"Retrieved pages: {vd.get('retrieved_pages')}")
    if (not st.session_state.admin_view) and msg.get("sources"):
        st.caption("Sources")
        for s in msg["sources"]:
            st.write(f"- {s}")
    if msg.get("finder_pages"):
        render_finder_mode_pages_ui(
            msg["finder_pages"],
            key_prefix=f"hist_{msg_i}",
            expanded=False,
        )

# --- Retrieval using FAISS index ---
def retrieve_faiss_chunks(
    question: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    ensure_data_dirs()
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return []

    chunks = load_chunks()
    if not chunks:
        return []

    model = get_embedding_model(resolve_embedding_device())
    query_vec = model.encode(
        [question],
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=ENCODE_BATCH_SIZE,
    )
    query_vec = query_vec.astype("float32")

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    candidate_k = min(max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k), len(chunks))
    distances, indices = index.search(query_vec, candidate_k)

    query_l = question.lower()
    chart_query = any(
        kw in query_l
        for kw in (
            "chart",
            "table",
            "graph",
            "figure",
            "plot",
            "fee",
            "rate",
            "schedule",
            "$",
            "parking",
            "stall",
            "space",
            "restaurant",
            "square foot",
            "sq ft",
            "gsf",
            "gross floor",
        )
    )

    retrieved: List[Dict[str, Any]] = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx < 0 or idx >= len(chunks):
            continue
        ch = chunks[idx]
        text = ch.get("text", "")
        is_chart_transcribed = "[Chart/figure transcription from page render" in text
        adjusted_distance = float(dist)
        if is_chart_transcribed:
            # Prefer chart-enriched chunks for chart/table questions.
            adjusted_distance -= 0.45 if chart_query else 0.15
        retrieved.append(
            {
                "doc_name": ch.get("doc_name", "Document"),
                "file_path": ch.get("file_path"),
                "page_num": ch.get("page_num", 0),
                "chunk_id": ch.get("chunk_id", rank),
                "text": text,
                "distance": adjusted_distance,
            }
        )
    retrieved.sort(key=lambda x: x.get("distance", 1e9))
    return retrieved[:top_k]


# --- Chat input ---
resolved_api_key = get_api_key()
has_api_key = bool(resolved_api_key)

prompt = st.chat_input(
    "Ask about the document..."
    if has_api_key
    else "Set ANTHROPIC_API_KEY in Secrets or use Save key in the banner above.",
    disabled=not has_api_key,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve top_k chunks from FAISS
    retrieved = retrieve_faiss_chunks(prompt, top_k=TOP_K)
    semantic_for_finder: List[Dict[str, Any]] = list(retrieved)
    if _finder_table_heavy_query(prompt):
        aug_q = (
            f"{prompt} parking stalls spaces required per square foot gross floor area "
            "non-residential use schedule table chart"
        )
        semantic_for_finder.extend(retrieve_faiss_chunks(aug_q, top_k=TOP_K))
    finder_retrieval = merge_keyword_page_candidates(
        prompt, load_chunks(), semantic_for_finder
    )
    finder_page_cap = (
        FINDER_PAGE_LIMIT_TABLE_HEAVY if _finder_table_heavy_query(prompt) else FINDER_PAGE_LIMIT
    )
    st.session_state.last_retrieved = retrieved
    context_block = build_context_block(retrieved)
    # Build mapping doc_name -> retrieved page numbers
    retrieved_doc_pages: Dict[str, List[int]] = {}
    for r in retrieved:
        name = r.get("doc_name", "Document")
        page = int(r.get("page_num", 0))
        retrieved_doc_pages.setdefault(name, [])
        if page not in retrieved_doc_pages[name]:
            retrieved_doc_pages[name].append(page)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            api_key = resolved_api_key
            finder_mode = False
            finder_pages: List[Dict[str, Any]] = []
            navigation_first_mode = bool(st.session_state.get("navigation_first_mode", False))
            if not api_key:
                response_text = (
                    "**API key required.** Add your Anthropic API key in one of these ways:\n\n"
                    "1. **Secrets:** Create a `.streamlit/secrets.toml` file with:\n   `ANTHROPIC_API_KEY = \"your-key\"`\n\n"
                    "2. **Sidebar:** Enter your key in the sidebar (stored only in this session)."
                )
            elif navigation_first_mode:
                finder_mode = True
                response_text, finder_pages = build_finder_mode_result(
                    prompt, finder_retrieval, max_pages=finder_page_cap
                )
                validation_passed = False
                detected_citations = []
            else:
                client = Anthropic(api_key=api_key)
                system_prompt = """You are a Land Development AI assistant. Use ONLY the excerpts in [CONTEXT]. Do not use outside knowledge.

Reply in a short, plain structure:

**Answer**
Give the direct answer in a few sentences (or a tight bullet list if that is clearer). No long legal memo, no A/B/C/D worksheets, no separate "intensity analysis" sections unless the user explicitly asks for that depth.

**Sources**
List every place you relied on, one per line, using this exact citation form so it can be parsed:
- [<doc_name>, Page N]

Rules for citations:
- <doc_name> must match a document name exactly as shown in [CONTEXT] (the name in parentheses before each excerpt).
- N must be a page number that actually appears in [CONTEXT] for that document.
- If the excerpt includes a clear subsection label (e.g. Sec. 4-1(a)), you may optionally write [<doc_name>, Sec. 4-1(a), Page N] instead—but still include Page N.

If [CONTEXT] does not support a reliable answer, say so briefly and list the closest relevant excerpts you were given (still with citations). Use the exact sentence below only when nothing in context is on point:
"Regulation not found in this document."

You may include a short optional quote from context in your Answer, but you are not required to use a special Quote: line or a minimum quote length."""

                user_content = f"""{context_block}

---
USER QUESTION:
{prompt}"""

                response = client.messages.create(
                    model=CHAT_MODEL,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
                raw_response_text = response.content[0].text
                _, validation_passed, detected_citations_strict = validate_engineer_rule(
                    raw_response_text,
                    retrieved_doc_pages,
                    retrieved,
                )
                # Relaxed mode: always show the model answer; do not fall back to Finder on strict validation failure.
                response_text = raw_response_text.strip()
                detected_citations = extract_citations_from_response(
                    raw_response_text, retrieved_doc_pages
                )
                if not detected_citations and detected_citations_strict:
                    detected_citations = list(detected_citations_strict)
                if validation_passed:
                    response_text = f"**Confidence: High**\n\n{response_text}"
                elif not response_text:
                    response_text = raw_response_text or NOT_FOUND_MESSAGE

            # If we never called validation (e.g., missing API key), set debug defaults
            if "validation_passed" not in locals():
                validation_passed = False
                detected_citations = []

            st.markdown(response_text)

        # Retrieved Evidence expander for this turn (admin only)
        if st.session_state.admin_view:
            with st.expander("📎 Retrieved Evidence"):
                for p in retrieved:
                    snippet = (p["text"].strip() or "(no text)")[:400]
                    if len(p["text"].strip() or "") > 400:
                        snippet += "..."
                    st.markdown(f"**{p.get('doc_name', 'Document')} – Page {p['page_num']}**")
                    st.text(snippet)
                    st.divider()
        elif detected_citations:
            unique_sources = sorted({f"{name}, Page {page}" for name, page in detected_citations})
            st.caption("Sources")
            for src in unique_sources:
                st.write(f"- {src}")

        if finder_mode and finder_pages:
            render_finder_mode_pages_ui(
                finder_pages,
                key_prefix=f"live_{len(st.session_state.messages)}",
                expanded=True,
            )

        # Debug expanders (admin only)
        if st.session_state.admin_view:
            with st.expander("🔍 Debug (validation)"):
                st.write(f"Strict quote/citation validation passed: {validation_passed}")
                st.write(f"Detected citations: {detected_citations}")
                st.write(f"Retrieved pages by doc: {retrieved_doc_pages}")

            with st.expander("📊 Retrieved chunks (FAISS scores)"):
                for p in retrieved:
                    st.write(
                        f"{p.get('doc_name', 'Document')} - Page {p['page_num']} "
                        f"(chunk {p.get('chunk_id')}), distance={p.get('distance'):.4f}"
                    )

    stored_sources = sorted({f"{name}, Page {page}" for name, page in detected_citations})
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "retrieved": retrieved,
        "sources": stored_sources,
        "validation_debug": {
            "validation_passed": validation_passed,
            "citations": detected_citations,
            "retrieved_pages": retrieved_doc_pages,
        },
        "finder_pages": finder_pages if "finder_pages" in locals() else [],
    })

# --- API key in sidebar (admin only) ---
if st.session_state.admin_view:
    with st.sidebar:
        st.divider()
        st.markdown("#### API access")
        st.caption(
            "Use **Streamlit Cloud → Settings → Secrets** with `ANTHROPIC_API_KEY` for deploys. "
            "Never commit a real key to GitHub — `.streamlit/secrets.toml` is gitignored."
        )
        if _has_server_or_file_api_key():
            st.success("API key is loaded from secrets (value is not shown).")
        else:
            st.text_input(
                "Anthropic API key (session only)",
                type="password",
                key="sidebar_api_key",
                label_visibility="collapsed",
                placeholder="sk-ant-api03-…",
                help="Stored only in this browser session. For production, use Streamlit Secrets instead.",
            )
            if get_api_key():
                st.success("API key loaded for this session.")
            else:
                st.warning("No API key yet — add secrets or paste a key above.")

