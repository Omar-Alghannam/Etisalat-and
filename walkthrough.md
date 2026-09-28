# Walkthrough: Exact Document Coordinates & Targeted Sentence Highlighting

We have implemented the **Right Approach (Correct Fix)** for citation attribution and document highlighting. Instead of fuzzy-matching entire chunk blocks or generic plan titles across the entire document, the system now tracks exact raw document coordinates and pinpoints the specific supporting sentence for each citation.

---

## 🛠️ Changes Implemented

### 1. Document Coordinate Tracking ([`step1_parse.py`](file:///c:/E&_Agentic/step1_parse.py))
- **Per-Page PDF Extraction**: Replaced flat concatenation with `extract_pages_from_pdf()`, tracking `page_num` (1-indexed) and page text.
- **Coordinate Mapping**: Each semantic sub-chunk is mapped back to its source `page_num`, `char_start`, `char_end`, and `raw_span`.
- **DOCX Paragraph Coordinates**: Added `para_idx` tracking to paragraph extraction and chunk metadata.
- **Clean Span Isolation**: Preserves `raw_span` without the synthetic repetitive headers (e.g., `"Plan: Mini Hekaya Internet 63"` is no longer part of the body span).

### 2. Rich Vector Store Metadata ([`step2_vectorstore.py`](file:///c:/E&_Agentic/step2_vectorstore.py))
- ChromaDB collections (`kb_en` and `kb_ar`) store primitive coordinates in each document's metadata:
  - `page_num` (int)
  - `char_start` (int)
  - `char_end` (int)
  - `para_idx` (int)
  - `raw_span` (str)

### 3. Claim-to-Sentence Attribution with Cross-Encoder ([`step3_graph.py`](file:///c:/E&_Agentic/step3_graph.py))
- Upgraded `_extract_supporting_sentence()`:
  - When the LLM generates an answer citing `[i]`, it extracts the exact claim sentences carrying `[i]`.
  - **Decimal-Safe Boundary Splitting**: Uses `(?<!\d)[.!?؟](?!\d)\s+|[•\r\n]+` to preserve telecom prices and quotas like `22.5 EGP` and `0.5 Mix/min` without mid-number splitting.
  - **Soft-Wrap Normalization**: Normalizes PDF line-wrapping newlines into spaces so multi-line sentences remain intact.
  - **Cross-Encoder Scoring**: Leverages the in-memory `_reranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) via `_reranker.predict([[claim, cand] for cand in candidates])` to score candidate sentences with cross-attention precision (~10ms CPU, zero API cost).
  - **In-Memory Cache**: Caches scored sentences by citation index and claim snippet to eliminate redundant computation.
  - The single best supporting sentence becomes `highlight_text`, avoiding repeated generic headers and resolving near-duplicate plan ambiguities.
- Emits rich citation objects in `sources`:
  ```python
  {
      "label": source_label,
      "plan_name": plan_name,
      "source_file": source_file,
      "language": doc_language,
      "page_num": int(doc["metadata"].get("page_num", 1)),
      "char_start": int(doc["metadata"].get("char_start", 0)),
      "char_end": int(doc["metadata"].get("char_end", 0)),
      "highlight_text": target_sentence,
      "citation_idx": i
  }
  ```

### 4. Page-Targeted Document Rendering ([`server.py`](file:///c:/E&_Agentic/server.py))
- `/api/documents/preview` accepts `page`, `char_start`, `char_end`, `para_idx`, and `highlight`.
- For PDFs:
  - Highlighting is applied **strictly to the targeted page** (`page_num == page`). Other pages are untouched, preventing headings from being highlighted multiple times.
  - Tags target page cards with `id="doc-page-{page_num}"` and `class="doc-page-targeted"`.
  - Injects `<mark class="doc-highlight doc-highlight-primary" id="activeCitationMark">`.
- For DOCX:
  - Highlighting is focused on the targeted paragraph (`para_idx`).

### 5. Precision Navigator & Auto-Scroll ([`index.html`](file:///c:/E&_Agentic/index.html))
- `resolveDocInfo` and `buildMessageCitationList` extract `pageNum`, `charStart`, `charEnd`, and `exactChunkText`.
- `updateCitationNavigatorUI` displays citation progress and target page (e.g. `HekayaInternet.pdf • Page 2`).
- `focusActiveCitation()` requests preview with `page`, `char_start`, and scrolls directly to `#activeCitationMark` or `#doc-page-{page}`.

---

## 🚀 How to Test the New System

In your terminal:

1. **Re-parse the documents with coordinates**:
   ```powershell
   python step1_parse.py
   ```

2. **Re-embed into ChromaDB**:
   ```powershell
   python step2_vectorstore.py
   ```

3. **Restart the server**:
   ```powershell
   python server.py
   ```

4. In the browser, press **`Ctrl + F5`** to reload `http://localhost:8000`:
   - Ask a question (e.g., *"What is the difference between Hekaya Mixat and Hekaya Internet?"*).
   - Click citation `[1]`.
   - The modal will open directly to the cited page with **only the supporting sentence highlighted**, with zero repeated marks on titles.
