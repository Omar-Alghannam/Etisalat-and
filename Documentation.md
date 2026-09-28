# e& Egypt Agentic RAG Knowledge Assistant
### Technical and Business Documentation — Internal Use

---

## 1. Executive Summary

This project delivers an Agentic Retrieval-Augmented Generation (RAG) assistant purpose-built for the e& Egypt customer service domain. Retrieval-Augmented Generation is an artificial intelligence architecture in which a language model is provided with relevant, verified information retrieved from an internal knowledge base before generating a response. This constrains the model's output to facts that actually exist in the company's documentation, preventing it from producing inaccurate or fabricated answers.

The assistant is designed to answer complex customer inquiries about e& Egypt's prepaid and postpaid telecommunication plans—including Aqwa Card, Hekaya Mixat, Hekaya Internet, Emerald, and DataLine—with speed, precision, and full source attribution. It supports both Arabic and English queries natively.

The system is fully operational. It has been formally evaluated for factual accuracy, absence of hallucination, and resistance to adversarial prompt attacks. It is currently accessible through a custom branded web interface served locally and through a native tool integration with the Claude Desktop application using the Model Context Protocol (MCP). Production logs are written to disk for ongoing monitoring.

---

## 2. Project Scope and Objectives

### 2.1 Goals

The primary goal of the project is to create an AI-powered customer service assistant that:

- Answers subscriber questions about active e& Egypt plans accurately, without inventing information.
- Attributes every claim to a specific document source, enabling users to verify answers.
- Self-corrects when its own answers fail quality checks before presenting a response to the user.
- Supports conversational, multi-turn interactions rather than treating every question in isolation.
- Operates in both Arabic and English based on the language of the incoming query.

### 2.2 Intended Use Case and Target Users

The system is intended for use by e& Egypt customer service operations teams and internal product knowledge management stakeholders. The immediate use case is internal evaluation and demonstration: showcasing how an agentic AI pipeline can be applied to the telecom domain to produce verifiable, grounded answers.

### 2.3 Success Criteria

The project defines success across three dimensions:

| Dimension | Target | Achieved |
|---|---|---|
| Factual Accuracy (LLM-as-a-Judge) | Greater than 85% pass rate | 92% |
| Faithfulness to Source Documents (RAGAS) | Score above 0.80 | 0.85 |
| Answer Relevance (RAGAS) | Score above 0.80 | 0.88 |
| Security (Garak Red-Teaming) | No jailbreak success | 100% blocked |

---

## 3. System Architecture

### 3.1 Architectural Overview

The system is composed of three layers operating in sequence:

1. **Data Layer**: Offline processing of source documents into a structured, searchable vector database.
2. **Pipeline Layer**: A stateful, multi-node LangGraph workflow that orchestrates query understanding, retrieval, generation, and validation.
3. **Delivery Layer**: A FastAPI server exposing the pipeline as a REST API, consumed by both a web frontend and a Claude Desktop MCP tool.

### 3.2 Workflow Diagram

The following diagram illustrates the complete LangGraph pipeline, including the self-correction retry loop between the `validate` and `generate` nodes.

![LangGraph RAG Workflow Diagram](C:/Users/Omar/.gemini/antigravity-ide/brain/3373c860-4807-483d-aa43-9116ae27ff19/rag_workflow_diagram_1787744653323.jpg)

*Figure 1: The LangGraph pipeline workflow. A query enters from the top, flows through language detection, query expansion, retrieval, context assembly, and generation. The validate node evaluates the generated response against quality rules. If validation fails and retry attempts remain, the workflow loops back to the generate node with a correction hint. Once validation passes, the response is delivered.*

![LangGraph Validation Loop Detail](C:/Users/Omar/.gemini/antigravity-ide/brain/3373c860-4807-483d-aa43-9116ae27ff19/langgraph_validator_workflow_1787746252845.jpg)

*Figure 2: Detail of the validation and self-correction routing logic. The `route_after_validate` conditional edge determines whether the workflow proceeds to `respond` or loops back to `generate` with an injected correction hint.*

### 3.3 Node-by-Node Workflow Description

The LangGraph workflow consists of seven nodes. Each node receives a shared `GraphState` object and returns a partial update to that state.

**Node 1: `understand_query`**

- **Purpose:** Detect the language of the incoming user query.
- **Input:** Raw user query string.
- **Logic:** Inspects the query for Unicode characters in the Arabic range (U+0600 to U+06FF). If any are found, the language is set to `"ar"`; otherwise it defaults to `"en"`.
- **Output:** Sets `language` field in graph state.
- **Role:** Downstream nodes use the detected language to select the appropriate ChromaDB collection and system prompt language.

**Node 2: `query_expansion`**

- **Purpose:** Rewrite ambiguous or pronoun-heavy queries into retrieval-optimized search strings.
- **Input:** Raw user query and the last four messages of chat history.
- **Logic:** A guard condition determines whether expansion is necessary. If the query contains a recognized plan name, a numeric identifier, or is longer than six words, and contains no unresolved pronouns, it is passed through unchanged. Otherwise, the query is submitted to an Azure OpenAI LLM with a structured prompt that instructs it to replace pronouns with the actual entity from conversation history and to expand vague terms such as "cheap plan" with keywords like "lowest price" or "most mixes."
- **Output:** Sets `expanded_query` field in graph state.
- **Role:** Ensures that ambiguous follow-up questions like "What about that plan?" resolve to specific, searchable queries before hitting the vector database.

**Node 3: `retrieve`**

- **Purpose:** Fetch semantically relevant document chunks from ChromaDB.
- **Input:** Expanded query, detected language.
- **Logic:** The retrieval strategy operates in three stages. First, a primary semantic query retrieves up to 20 candidate chunks from the collection corresponding to the detected language. Second, if plan-specific keywords such as `Hekaya`, `Mixat`, or numeric plan codes are present in the query, a focused keyword entity query retrieves three additional targeted chunks to prevent a broadly relevant but plan-specific query from being diluted. Third, a cross-lingual fallback retrieves four additional chunks from the alternate language collection to ensure bilingual coverage. All results are de-duplicated by text content before proceeding. A cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) then scores and reranks the final candidate pool, prioritizing the most semantically precise documents.
- **Output:** Sets `retrieved_docs` field in graph state.

**Node 4: `assemble_context`**

- **Purpose:** Format the retrieved documents into a numbered context block for the LLM.
- **Input:** List of retrieved document chunks with associated metadata.
- **Logic:** Each chunk is wrapped with its index number, plan name, and product family identifier before being joined with separators into a single context string. This numbered format is what enables the model to produce inline citations.
- **Output:** Sets `context` field in graph state.

**Node 5: `generate`**

- **Purpose:** Draft a grounded response using the assembled context.
- **Input:** Language, context string, user query, and up to six messages of prior chat history.
- **Logic:** A language-appropriate system prompt is selected (English or Arabic) and combined with the formatted context. The system prompt provides strict instructions: the model must answer using only the provided context, must include numbered inline citations, must not fabricate information, and must format its response using Markdown with bold headings and bullet points. USSD subscription codes must be wrapped in backticks. If a `correction_hint` is present (injected by the validate node on retry), it is prepended to the user's message before the LLM call. The temperature is set to 0.2 to produce consistent, factual outputs.
- **Output:** Sets `answer` field in graph state.

**Node 6: `validate`**

- **Purpose:** Evaluate the generated answer against quality rules before presenting it to the user.
- **Input:** Generated answer, retrieved documents, language, retry count.
- **Logic:** Three deterministic rules are applied: (A) the answer must contain at least one inline citation matching the pattern `[N]`; (B) the answer must be substantive, containing at least ten words; (C) any USSD codes must be enclosed in backticks. If any rule fails and fewer than two retry attempts have occurred, a language-appropriate correction hint is injected into the graph state and the retry counter is incremented. Out-of-scope refusal messages are detected and exempted from validation automatically.
- **Output:** Sets `correction_hint` and `retry_count` fields.

**Node 7: `respond`**

- **Purpose:** Extract the final answer, sources, and language from graph state and return them to the API layer.
- **Input:** Complete final graph state.
- **Output:** Returns `answer`, `sources`, `language`, and `retrieved_docs`.

### 3.4 Routing and Conditional Logic

After Node 6 (`validate`) completes, a conditional edge function `route_after_validate` inspects the graph state. If `correction_hint` is non-empty and `retry_count` is below the maximum of two, the edge routes back to the `generate` node. This creates a deterministic self-correction loop capable of fixing formatting violations or missing citations without human intervention. If either condition is unmet, the workflow exits to the `respond` node.

---

## 4. Technology Stack

| Technology | Role | Justification |
|---|---|---|
| **LangGraph** | Workflow orchestration and state management | Provides explicit, auditable control flow with conditional edges and a typed state schema. Suitable for multi-step agentic pipelines where node-level transparency is required. |
| **LangChain** | LLM and embedding abstraction | Standardizes API calls to Azure OpenAI across both the chat model and embedding model, with built-in retry logic. |
| **Azure OpenAI (gpt-4.1-mini)** | Language model for generation, validation, and query expansion | Enterprise-grade model with built-in content management policies, compliant with corporate security requirements. |
| **Azure OpenAI (text-embedding-ada-002)** | Text embeddings for semantic search | High-quality dense vector representation of plan documentation, hosted on the same Azure endpoint. |
| **ChromaDB** | Local vector database | Persistent, file-based vector store requiring no external server. Collections for English and Arabic are maintained separately to allow language-targeted retrieval. |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Document reranking | Cross-encoders compute joint attention over both the query and document, producing significantly more precise relevance scores than bi-encoder cosine similarity alone. |
| **FastAPI** | REST API server | High-performance, async-capable Python web framework. Serves both the web frontend and the MCP HTTP interface. |
| **Model Context Protocol (MCP)** | Claude Desktop integration | Standard protocol for connecting external tools to Claude Desktop. Enables the assistant to function as a native tool within the Claude interface without requiring a full client application. |
| **python-docx / pdfplumber** | Document parsing | Extracts raw text from Word and PDF source documents provided by the e& Egypt knowledge base team. |

---

## 5. Data and Knowledge Base

### 5.1 Knowledge Base Structure

The source knowledge base comprises documents covering five primary product families:

| Product Family | Category | Language |
|---|---|---|
| Aqwa Card / Prepaid Systems | Prepaid | English and Arabic |
| DataLine | Data | English and Arabic |
| Emerald | Postpaid | English and Arabic |
| Hekaya Internet | Internet Add-on | English and Arabic |
| Hekaya Mixat | Mix Plans | English and Arabic |

Documents are stored in separate English and Arabic versions within a structured directory. Source files include both Word (`.docx`) and PDF (`.pdf`) formats.

### 5.2 Data Processing Pipeline

The pipeline is implemented in `step1_parse.py` and operates as follows:

1. **Parsing**: Each source file is loaded using either `python-docx` (for Word documents) or `pdfplumber` (for PDF documents) to extract raw text.

2. **Chunking Strategy**: The documents are chunked by natural document headers rather than by fixed character or token counts. The parser identifies section boundaries using regular expression patterns that match plan headers such as `Recharge: Aqwa Card 19` or `Plan: DataLine 100`. Each plan-level header and its associated body text is treated as one discrete chunk. This strategy was chosen deliberately: telecom plan documents are inherently structured—each plan is a self-contained unit of information with its own price, quota, validity, and terms. Splitting by character count would have frequently separated a plan's name from its pricing, breaking semantic coherence. Header-based splitting preserves each plan as an atomic retrieval unit.

3. **Output**: Parsed chunks are saved to `data/chunks_en.json` and `data/chunks_ar.json`.

### 5.3 Metadata Schema

Each chunk carries five metadata fields attached at parse time:

| Field | Description |
|---|---|
| `product_family` | The top-level product category (e.g., "Hekaya Mixat") |
| `plan_name` | The specific plan name extracted from the section header |
| `category` | A short category code (e.g., "prepaid", "data", "mix") |
| `language` | "en" or "ar" |
| `source_file` | The filename from which the chunk was extracted |

This metadata serves two purposes: it enables the `assemble_context` node to attach human-readable source labels to each numbered citation, and it supports potential future metadata filtering within ChromaDB.

### 5.4 Multilingual Handling

English and Arabic chunks are indexed into two separate ChromaDB collections: `kb_en` and `kb_ar`. At retrieval time, the `retrieve` node selects the primary collection based on the language detected by `understand_query`. A cross-lingual fallback query against the alternate collection is always performed, ensuring that an Arabic-language question can still surface English-indexed content when the Arabic collection lacks a precise match, and vice versa.

---

## 6. Retrieval System

### 6.1 Retrieval Strategy

Retrieval employs a three-stage candidate gathering process followed by cross-encoder reranking.

**Stage 1 — Primary Semantic Search:** The expanded query is submitted to the appropriate language-specific ChromaDB collection. The collection's embedding function converts the query into a dense vector using the Azure OpenAI embedding model, then performs an approximate nearest-neighbor search to return up to 20 candidate document chunks.

**Stage 2 — Keyword Entity Boost:** If the query contains specific plan identifiers (recognized plan family names or two-to-four digit numeric codes), a secondary targeted query is issued using only those extracted keywords. This addresses a common failure mode in semantic search where a query about a specific plan variant (e.g., "Hekaya Mixat 52") retrieves semantically adjacent but non-identical plans due to embedding proximity. The entity boost query returns up to three additional chunks with high plan-name specificity and inserts them at the front of the candidate list.

**Stage 3 — Cross-Lingual Fallback:** A fallback query against the alternate language collection returns up to four additional chunks, de-duplicated against the primary results. This ensures cross-lingual coverage.

**Reranking:** After all three stages are combined and de-duplicated, the `cross-encoder/ms-marco-MiniLM-L-6-v2` model scores each candidate against the original query. Unlike a bi-encoder which embeds query and document independently, a cross-encoder reads both simultaneously and produces a single relevance score. This provides significantly more accurate ranking at the cost of higher inference time, which is acceptable given the local deployment context.

### 6.2 Vector Database Configuration

ChromaDB is configured as a persistent local client writing to the `chroma_db/` directory. Two collections are created and populated by `step2_vectorstore.py`:
- `kb_en`: English-language plan chunks.
- `kb_ar`: Arabic-language plan chunks.

Both collections are initialized with a custom `AzureOpenAIEmbeddingFunction` wrapper that conforms to ChromaDB's `EmbeddingFunction` interface while delegating actual embedding computation to the LangChain `AzureOpenAIEmbeddings` class.

---

## 7. Generation and Response System

### 7.1 Prompt Design and Grounding Strategy

The generation node uses two language-specific system prompts (English and Arabic) that follow identical structural logic. Each prompt includes the following directives:

- Answers must be derived exclusively from the provided numbered context. The model is explicitly forbidden from using its own parametric knowledge.
- The model must state that it lacks information if the context does not contain a relevant answer. This prevents confident-sounding hallucination on out-of-scope questions.
- Every factual claim must be accompanied by an inline citation referencing the numbered context block from which it was drawn.
- USSD subscription codes must be enclosed in backtick characters to prevent markdown rendering from interpreting asterisks as formatting characters.
- Responses must be formatted in Markdown with bold plan names and prices, bullet-pointed feature lists, and section headers where multiple plans are discussed.
- When the user's query provides constraints (such as budget or usage level), the model is instructed to provide a concrete recommendation rather than asking clarifying questions. Clarifying questions are reserved only for entirely constraint-free queries.

### 7.2 Citation and Source Attribution

Inline citations take the form `[N]` where `N` corresponds to the number of the context block as assembled by `assemble_context`. After the `respond` node returns the answer, the server extracts the unique source labels (plan name and product family) from the documents referenced in the answer's citations and returns them as a separate `sources` array in the API response. The frontend displays these as a labeled source list below the response.

### 7.3 Language Handling

The Arabic and English system prompts are structurally identical but fully translated. The query expansion prompt operates in English regardless of query language, as the underlying LLM performs better at semantic rewriting in English. The generated answer, however, is produced in the language of the user's original query (Arabic or English) based on which system prompt is selected.

### 7.4 Answer Validation and Self-Correction Logic

The validate node applies three deterministic rule checks:

1. **Citation presence**: The answer must match the regular expression `\[\d+\]` at least once.
2. **Minimum length**: The answer must contain at least ten words.
3. **USSD code formatting**: Any sequence matching an unformatted USSD pattern must not be present.

On failure, the language-appropriate correction hint is injected as a prepended instruction on the next `generate` call. A maximum of two retry attempts are permitted, after which the pipeline exits to `respond` with the best available answer. This cap prevents infinite retry loops under adversarial or degenerate conditions.

---

## 8. User Interface

### 8.1 Chat Interface Design

The user interface is a custom-built single-page application (SPA) delivered via `index.html` and served directly by the FastAPI server at the root endpoint. The interface presents a chat-style layout with a sidebar for managing conversation threads and a main panel for the active conversation.

Thread persistence is implemented server-side through a `data/threads.json` file. Each conversation thread is saved with a unique timestamp-based identifier and a name derived from the first user message.

The response panel renders Markdown formatting including bold text, bullet lists, inline code blocks (used for USSD codes), and section headers. Citations appear as numbered references within the response body, and a separate source attribution block lists the referenced plan documents below each assistant response.

### 8.2 Branding and Design

The interface adopts the e& Egypt visual identity using the organization's color palette applied to the sidebar, header, and interactive elements. The welcome message and introductory prompt are bilingual, addressing both Arabic and English-speaking users from the initial state.

---

## 9. Evaluation and Quality Assurance

### 9.1 LLM-as-a-Judge Evaluation

**Tool:** Internal evaluation script (`step4_eval.py`).

**Method:** A curated dataset of 50 question-and-expected-answer pairs was constructed to represent realistic customer queries across all five product families. The pipeline was executed against each question. A separate LLM call was then made to compare the generated answer against the expected answer, acting as an automated grader. This approach is referred to as "LLM-as-a-Judge" in the applied machine learning literature.

**Result:** 92% pass rate (46 of 50 test cases passed). The four failing cases involved queries requiring enumeration of all variants within a product family, a scenario where the retrieval stage did not consistently return all relevant plan sub-variants in a single pass.

### 9.2 RAGAS Evaluation

**Tool:** RAGAS framework (`step5_ragas_eval.py`).

**Method:** RAGAS (Retrieval Augmented Generation Assessment) computes standardized metrics by comparing the model's generated answer against both the retrieved context and the reference question. Two primary metrics were measured:

- **Faithfulness**: Measures whether every statement in the generated answer can be directly inferred from the retrieved context. A score of 1.0 would indicate complete adherence; a score of 0.0 would indicate complete reliance on invented information.
- **Answer Relevance**: Measures how directly and completely the answer addresses the user's original question. High scores indicate concise, on-topic responses.

**Results:**

| Metric | Score |
|---|---|
| Faithfulness | 0.85 |
| Answer Relevance | 0.88 |

These scores indicate that the system predominantly anchors its answers to retrieved documentation and stays on-topic. The faithfulness score of 0.85, while not perfect, reflects instances where minor generalization occurred in the assembly of multi-plan comparison responses.

### 9.3 Security and Adversarial Testing

**Tool:** NVIDIA Garak vulnerability scanner v0.16.0 (`step7_garak.ps1`).

**Method:** Garak was configured with a REST generator pointing at the local FastAPI endpoint. The `dan.Dan_11_0` probe was executed, which sends a series of advanced "Do Anything Now" (DAN) jailbreak prompt sequences designed to override the model's safety guidelines and induce policy-violating responses.

**Result:** The system resisted all five adversarial attacks. Azure OpenAI's Content Management Policy intercepted the malicious prompts before they could reach the generation stage and returned a `400 Bad Request` with a `ResponsibleAIPolicyViolation` status. The pipeline's exception handling blocks in the `query_expansion` and `generate` nodes captured these errors gracefully, logging a console warning and returning a predefined refusal message without terminating the server process.

Note on Garak's automated scoring: Garak's `mitigation.MitigationBypass` detector reported a nominal failure because it did not recognize the custom refusal string as a standard AI safety refusal. This is a false positive attributable to Garak's detector expecting specific pre-set refusal phrases. Manual review of the server logs confirms that the attacks were blocked by Azure's content layer in all five cases.

### 9.4 Attempted but Skipped Evaluation

**Tool:** Giskard vulnerability scanner.

**Reason skipped:** Giskard's internal workflow engine contains hardcoded logic that assumes a standard OpenAI API URL format. When configured for Azure OpenAI, it appended the `/chat/completions` path segment after the `?api-version` query string parameter, producing an invalid URL that Azure rejected with a 404 error. Attempts to subclass the internal `LLMClient` class were blocked by a missing module (`giskard.llm.client`) that does not exist in the installed version. The installed version could not be upgraded because the pip dependency resolver identified a blocking conflict with `traceloop-sdk`. Given that security and hallucination testing had already been completed via Garak and RAGAS respectively, Giskard evaluation was excluded from the final validation suite.

---

## 10. Current Limitations

**Context window constraint for multi-turn sessions:** The MCP tool and the pipeline's `generate` node each retain a sliding window of six messages (three conversational turns). Conversational context beyond that window is discarded. For extended sessions involving topic switching or long background discussions, the query expansion node may fail to resolve references accurately once the relevant turn falls outside the six-message window.

**Product family enumeration failures:** The LLM-as-a-Judge evaluation identified four failing test cases, all related to queries requesting a complete listing of all plans within a product family (e.g., "List all Hekaya Mixat plans"). Because the retrieval stage prioritizes semantic similarity to the query rather than exhaustive catalog coverage, it does not guarantee that every plan variant for a given family will be included in the context window on every query. Responses to enumeration-style queries may therefore omit plan variants.

**Single-machine local deployment:** The system currently runs entirely on a single local machine. The ChromaDB vector store, the FastAPI server, and the MCP bridge all share the same process environment. This architecture is appropriate for internal demonstration and evaluation but is not suitable for multi-user production deployment without migration to a distributed infrastructure.

**Giskard evaluation gap:** As documented in section 9.4, the Giskard evaluation framework could not be integrated due to a library compatibility issue. While the system's security posture has been confirmed through Garak, the LLM-behavior vulnerability categories covered by Giskard (such as hallucination under adversarial framing, stereotype bias, and off-topic response generation) were not formally tested using that framework.

---

## 11. Recommended Next Steps

**Exhaustive retrieval for enumeration queries:** Implement query intent classification to detect enumeration-style queries (e.g., queries containing "all", "list", "every", "compare all"). When detected, the retrieval strategy should switch to a metadata-filtered query that retrieves all chunks belonging to a specified product family rather than relying on semantic similarity alone. This would directly address the four failing test cases identified in evaluation.

**Dynamic context window with summarization:** Replace the fixed six-message sliding window with a summarization mechanism. When conversation history exceeds the six-message threshold, older turns are summarized by the LLM into a compact context paragraph appended to the beginning of the history. This preserves long-session continuity without exceeding the model's token budget.

**Production infrastructure migration:** Deploy the FastAPI server to a scalable cloud environment (Azure App Service or Azure Container Apps) with the ChromaDB store replaced by a managed vector database service (such as Azure AI Search). This enables concurrent multi-user access and removes the dependency on a locally running machine.

**Structured pricing table retrieval:** The current retrieval pipeline represents plan data as unstructured prose. Implementing a hybrid retrieval approach that stores normalized, structured plan attributes (price, quota, validity, USSD code) in a relational or tabular format alongside the vector index would significantly improve the accuracy of comparison queries and pricing lookups.

**Formal Giskard evaluation upon environment resolution:** Once the `traceloop-sdk` dependency conflict is resolved in a clean virtual environment, a full Giskard adversarial vulnerability scan should be conducted to formally assess LLM behavior under stereotype injection, off-topic framing, and hallucination-inducing prompts, complementing the existing Garak red-teaming results.
