"""
Standalone FastAPI Server for e& Egypt RAG Chatbot Dashboard

Connects the custom HTML/CSS/JS frontend directly to step3_graph.py.
Handles chat queries, session persistence, and thread management.

Usage:
    python server.py
"""

import os
import json
import uvicorn
import re
import docx
import pdfplumber
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from step3_graph import run_query

from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
THREADS_FILE = os.path.join(BASE_DIR, "data", "threads.json")
LOGS_FILE = os.path.join(BASE_DIR, "data", "production_logs.jsonl")
UPLOAD_DIR = r"C:\Users\Omar\.gemini\antigravity-ide\brain\3373c860-4807-483d-aa43-9116ae27ff19\.user_uploaded"
LOCAL_IMAGES_DIR = os.path.join(BASE_DIR, "images")
KB_ROOT = os.path.join(BASE_DIR, "e& Knowledge base - with English version")
EN_DIR = os.path.join(KB_ROOT, "English")
AR_DIR = os.path.join(KB_ROOT, "e& Knowledge base")

# Copy images to local directory if needed
os.makedirs(LOCAL_IMAGES_DIR, exist_ok=True)
if os.path.exists(UPLOAD_DIR):
    import shutil
    for fname in os.listdir(UPLOAD_DIR):
        src = os.path.join(UPLOAD_DIR, fname)
        dst = os.path.join(LOCAL_IMAGES_DIR, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

app = FastAPI(title="e& Egypt Knowledge Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static images
if os.path.exists(LOCAL_IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=LOCAL_IMAGES_DIR), name="images")
elif os.path.exists(UPLOAD_DIR):
    app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")


# ---------------------------------------------------------------------------
# Data Persistence & Document Helpers
# ---------------------------------------------------------------------------

def _find_document_path(filename: str, lang: str = "en") -> Optional[str]:
    """Locate the requested knowledge base document on disk."""
    if not filename:
        return None
    
    clean_name = os.path.basename(filename).strip()
    candidates = [
        os.path.join(EN_DIR, clean_name),
        os.path.join(AR_DIR, clean_name),
        os.path.join(KB_ROOT, clean_name),
    ]
    
    base_name = os.path.splitext(clean_name)[0]
    if lang == "ar":
        candidates.extend([
            os.path.join(AR_DIR, f"{base_name}.docx"),
            os.path.join(AR_DIR, f"{base_name}.pdf"),
            os.path.join(EN_DIR, f"{base_name}.pdf"),
        ])
    else:
        candidates.extend([
            os.path.join(EN_DIR, f"{base_name}.pdf"),
            os.path.join(AR_DIR, f"{base_name}.docx"),
        ])
    
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _load_threads() -> dict:
    if os.path.exists(THREADS_FILE):
        try:
            with open(THREADS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_threads(data: dict):
    os.makedirs(os.path.dirname(THREADS_FILE), exist_ok=True)
    with open(THREADS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _log_production_query(query: str, result: dict):
    """Log the raw query, retrieved context, and generated answer to JSONL for monitoring."""
    os.makedirs(os.path.dirname(LOGS_FILE), exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_query": query,
        "language": result.get("language", "en"),
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "retrieved_contexts": [doc["text"] for doc in result.get("retrieved_docs", [])]
    }
    with open(LOGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []


class ThreadSaveRequest(BaseModel):
    id: str
    name: str
    messages: List[Dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def get_index():
    """Serve the custom index.html dashboard."""
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api/documents/open")
async def open_document(file: str, lang: Optional[str] = "en", download: Optional[bool] = False):
    """Serve the real knowledge base document directly (PDF inline viewing or DOCX attachment)."""
    path = _find_document_path(file, lang or "en")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Document '{file}' not found")
    
    filename = os.path.basename(path)
    if filename.lower().endswith(".pdf"):
        media_type = "application/pdf"
        disposition = "attachment" if download else "inline"
    elif filename.lower().endswith(".docx"):
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        disposition = "attachment"
    else:
        media_type = "application/octet-stream"
        disposition = "attachment"
    
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'}
    )


@app.get("/api/documents/preview")
async def preview_document(
    file: str,
    lang: Optional[str] = "en",
    highlight: Optional[str] = None,
    page: Optional[int] = None,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
    para_idx: Optional[int] = None
):
    """Render real document contents as clean, styled HTML with cited words/passages highlighted."""
    path = _find_document_path(file, lang or "en")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Document '{file}' not found")
    
    filename = os.path.basename(path)
    is_arabic = (lang == "ar" or filename.lower().endswith(".docx"))
    
    clean_hl = re.sub(r'\[\d+\]', '', highlight or '').strip() if highlight else ''

    def highlight_page_html(page_html: str, target_phrase: str) -> str:
        """Inject <mark> around the best match of target_phrase within a page's HTML.
        
        Strategy: use a SHORT anchor (first ~60 chars) to reliably locate the passage,
        then extend the mark forward to cover the full target length so the whole
        retrieved chunk is highlighted — not just one sentence.
        """
        if not target_phrase:
            return page_html

        clean_target = ' '.join(re.sub(r'\[\d+\]', '', target_phrase).split()).strip()
        if len(clean_target) < 4:
            return page_html

        tag_re = re.compile(r'<[^>]+>')
        stripped = tag_re.sub('', page_html)
        norm_stripped = ' '.join(stripped.split())

        # ── Build anchor probes (shortest reliable prefix that can be found) ──
        clean_core = re.sub(r'^[•\-\*\s]+', '', clean_target)
        
        # Try progressively shorter anchors until we get a hit
        anchor_lengths = [60, 45, 30, 20]
        anchor_match_idx = -1
        anchor_len_used = 0

        for alen in anchor_lengths:
            for base in [clean_target, clean_core]:
                anchor = ' '.join(base[:alen].split())  # re-normalise after slicing
                if len(anchor) < 8:
                    continue
                idx = norm_stripped.find(anchor)
                if idx >= 0:
                    anchor_match_idx = idx
                    anchor_len_used = len(anchor)
                    break
            if anchor_match_idx >= 0:
                break

        if anchor_match_idx >= 0:
            mark_start = anchor_match_idx
            # Extend end to cover the full original target length from this position
            desired_end = mark_start + len(clean_target)
            # But cap at a sentence-end boundary (., !, ?, ؟, •) within ±200 chars of desired_end
            search_from = min(desired_end, len(norm_stripped) - 1)
            # Walk forward to the nearest sentence boundary from desired_end (up to +200)
            mark_end = min(desired_end, len(norm_stripped))
            for offset in range(0, 200):
                pos = search_from + offset
                if pos >= len(norm_stripped):
                    break
                if norm_stripped[pos] in '.!?؟':
                    mark_end = pos + 1
                    break
                if norm_stripped[pos] == '•':
                    mark_end = pos
                    break

            # Inject <mark> tags by walking original HTML character-by-character
            result = []
            stripped_pos = 0
            in_tag = False
            mark_opened = False
            mark_closed = False

            i = 0
            while i < len(page_html):
                ch = page_html[i]
                if ch == '<':
                    in_tag = True
                    if mark_opened and not mark_closed and stripped_pos >= mark_end:
                        result.append('</mark>')
                        mark_closed = True
                    result.append(ch)
                elif ch == '>':
                    in_tag = False
                    result.append(ch)
                elif not in_tag:
                    if not mark_opened and stripped_pos == mark_start:
                        result.append('<mark class="doc-highlight doc-highlight-primary" id="activeCitationMark">')
                        mark_opened = True
                    result.append(ch)
                    stripped_pos += 1
                    if mark_opened and not mark_closed and stripped_pos >= mark_end:
                        result.append('</mark>')
                        mark_closed = True
                else:
                    result.append(ch)
                i += 1

            if mark_opened and not mark_closed:
                result.append('</mark>')

            return ''.join(result)

        # ── Fallback: highlight specific numeric/code tokens only ──
        tokens = re.findall(
            r'(\*\d+(?:\*\d+)*#|\d[\d,\.]+\s*(?:جنيه|EGP|GB|MB|جيجا|ميكس|دقيقة|Min|units?|وحدة))',
            clean_target, re.IGNORECASE
        )
        tokens = list(dict.fromkeys(t.strip() for t in tokens if t.strip()))
        res = page_html
        for tok in tokens:
            try:
                pat = re.escape(tok)
                res = re.sub(
                    f'(?<!<mark class="doc-highlight">)({pat})(?!</mark>)',
                    r'<mark class="doc-highlight">\1</mark>',
                    res, count=1, flags=re.IGNORECASE
                )
            except Exception:
                pass
        return res


    content_html = ""
    
    # Add top callout for the cited passage if available
    if clean_hl:
        dir_attr = "rtl" if is_arabic else "ltr"
        page_info = f" (Page {page})" if page else ""
        content_html += f'''
        <div class="doc-cited-callout" dir="{dir_attr}">
          <div class="doc-cited-callout-header">
            <i class="ti ti-target"></i>
            <span>Cited Reference Passage{page_info} (الفقرة المستشهد بها)</span>
          </div>
          <div class="doc-cited-callout-body">{clean_hl}</div>
        </div>
        '''

    try:
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                effective_page = page if (page and 1 <= page <= total_pages) else None
                
                for page_num, pdf_page in enumerate(pdf.pages, 1):
                    text = pdf_page.extract_text() or ""
                    is_target = (effective_page is None or effective_page == page_num)
                    targeted_cls = " doc-page-targeted" if (is_target and effective_page) else ""
                    page_card = f'<div class="doc-page-card{targeted_cls}" id="doc-page-{page_num}">'
                    page_card += f'<div class="doc-page-badge">Page {page_num}</div>'
                    
                    # Group wrapped lines into logical blocks (headings, bullets, paragraphs)
                    blocks = []
                    current_block = []
                    current_type = None

                    for raw_line in text.split("\n"):
                        l = raw_line.strip()
                        if not l:
                            if current_block:
                                blocks.append((current_type, " ".join(current_block)))
                                current_block = []
                                current_type = None
                            continue
                        
                        # True heading: short keyword-colon-value pattern like "Package: Data Line 8 GB"
                        # NOT a continuation sentence like "Package validity: three months."
                        _kw_match = re.match(r"^(Package|Plan|Recharge|Bundle|Emerald|Data Line|Aqwa Card|Hekaya|\d+\.)", l, re.IGNORECASE)
                        is_heading = bool(
                            _kw_match and
                            len(l) <= 70 and
                            not re.match(r"^(Package|Plan)\s+(name|validity|price|quota|total|compatible|renewal|value)", l, re.IGNORECASE)
                        )
                        is_bullet = bool(l.startswith("•") or l.startswith("-"))

                        if is_heading:
                            if current_block:
                                blocks.append((current_type, " ".join(current_block)))
                            blocks.append(('heading', l))
                            current_block = []
                            current_type = None
                        elif is_bullet:
                            if current_block:
                                blocks.append((current_type, " ".join(current_block)))
                            current_block = [l[1:].strip()]
                            current_type = 'bullet'
                        else:
                            if current_type in ('bullet', 'para'):
                                current_block.append(l)
                            else:
                                current_block = [l]
                                current_type = 'para'

                    if current_block:
                        blocks.append((current_type, " ".join(current_block)))

                    for b_type, b_text in blocks:
                        if b_type == 'heading':
                            page_card += f'<h3 class="doc-heading">{b_text}</h3>'
                        elif b_type == 'bullet':
                            page_card += f'<div class="doc-bullet"><i class="ti ti-point"></i> {b_text}</div>'
                        else:
                            page_card += f'<p class="doc-para">{b_text}</p>'
                    page_card += '</div>'

                    # Apply highlight ONLY to the targeted page
                    if is_target and clean_hl:
                        page_card = highlight_page_html(page_card, clean_hl)

                    content_html += page_card

        elif filename.lower().endswith(".docx"):
            doc = docx.Document(path)
            docx_html = ""
            for p_idx, p in enumerate(doc.paragraphs):
                text = p.text.strip()
                if not text:
                    continue
                is_target_para = (para_idx is not None and para_idx == p_idx)
                if any(run.bold for run in p.runs) or (len(text) < 60 and any(kw in text for kw in ["الباقة", "الشحنة", "نظام", "خط", "المزايا", "الأسعار", "جدول"])):
                    p_tag = f'<h3 class="doc-heading">{text}</h3>'
                elif text.startswith("•") or text.startswith("-"):
                    p_tag = f'<div class="doc-bullet"><i class="ti ti-point"></i> {text[1:].strip()}</div>'
                else:
                    p_tag = f'<p class="doc-para">{text}</p>'
                
                if is_target_para and clean_hl:
                    p_tag = highlight_page_html(p_tag, clean_hl)
                docx_html += p_tag

            for table in doc.tables:
                docx_html += '<div class="doc-table-wrap"><table class="doc-table">'
                for row_idx, row in enumerate(table.rows):
                    docx_html += '<tr>'
                    for cell in row.cells:
                        tag = "th" if row_idx == 0 else "td"
                        docx_html += f'<{tag}>{cell.text.strip()}</{tag}>'
                    docx_html += '</tr>'
                docx_html += '</table></div>'

            # If not highlighted by specific paragraph, search full DOCX text once
            if clean_hl and '<mark' not in docx_html:
                docx_html = highlight_page_html(docx_html, clean_hl)
            content_html += docx_html

    except Exception as e:
        content_html += f'<div class="doc-error-box"><p>Could not extract text: {str(e)}</p></div>'

    return {
        "filename": filename,
        "is_pdf": filename.lower().endswith(".pdf"),
        "is_arabic": is_arabic,
        "highlighted": bool(highlight),
        "target_page": page,
        "open_url": f"/api/documents/open?file={filename}&lang={lang}",
        "download_url": f"/api/documents/open?file={filename}&lang={lang}&download=true",
        "html": content_html
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Process a user query using the step3_graph RAG pipeline."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = run_query(user_query=query, chat_history=req.chat_history or [])
        lang_full = "Arabic (العربية)" if result["language"] == "ar" else "English"
        doc_count = len(result.get("retrieved_docs", []))
        
        thinking_steps = [
            f"🔤 Language Detection: Identified language as {lang_full}",
            f"📚 Vector Retrieval: Retrieved top {doc_count} relevant chunks from ChromaDB",
            f"📝 Context Assembly: Built context window and verified plan boundaries",
            f"🤖 LLM Reasoning: Sent context to Azure OpenAI (gpt-4.1-mini) with zero-hallucination rules"
        ]

        contexts = [doc["text"] for doc in result.get("retrieved_docs", [])]

        # Save to production logs for later review
        _log_production_query(query, result)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "language": result["language"],
            "contexts": contexts,
            "thinking_steps": thinking_steps,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/threads")
async def get_threads():
    """Get all saved chat threads for the sidebar."""
    threads = _load_threads()
    # Sort threads by last updated
    thread_list = list(threads.values())
    thread_list.reverse()
    return {"threads": thread_list}


@app.post("/api/threads")
async def save_thread(req: ThreadSaveRequest):
    """Save or update a chat thread."""
    threads = _load_threads()
    threads[req.id] = {
        "id": req.id,
        "name": req.name,
        "messages": req.messages,
    }
    _save_threads(threads)
    return {"status": "success"}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a chat thread."""
    threads = _load_threads()
    if thread_id in threads:
        del threads[thread_id]
        _save_threads(threads)
    return {"status": "deleted"}


if __name__ == "__main__":
    print("\n🔴 Launching e& Egypt Chat Dashboard at http://localhost:8000 ...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
