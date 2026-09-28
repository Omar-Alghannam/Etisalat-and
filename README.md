# e& Egypt Agentic RAG Pipeline

## Project Overview
This system is an Agentic Retrieval-Augmented Generation (RAG) pipeline designed to function as an automated customer service assistant for e& Egypt. It processes user inquiries regarding telecommunication plans, including Aqwa Card, Hekaya, Emerald, and DataLine, by retrieving contextual information from a specialized knowledge base and generating accurate responses.

The architecture relies on a LangGraph state machine to orchestrate the workflow. The pipeline consists of four primary nodes. First, the `query_expansion` node analyzes the user's input alongside recent conversation history to resolve pronouns and contextual ambiguities. Second, the `retrieve` node queries a vector database and applies a cross-encoder model to re-rank the documents based on semantic relevance. Third, the `generate` node drafts a response using a large language model (LLM) constrained by the retrieved context. Finally, the `validate` node utilizes an LLM-as-a-judge mechanism to evaluate the drafted response for hallucinations or inaccuracies. If the response fails validation, the system generates a correction hint and routes back to the generation node.

The technology stack includes LangChain and LangGraph for workflow orchestration, ensuring robust state management and iterative self-correction. ChromaDB is utilized for local vector storage. The backend is served via FastAPI, providing REST endpoints. A Model Context Protocol (MCP) server acts as a bridge, allowing the pipeline to operate natively within the Claude Desktop application as an external tool. Azure OpenAI provides the foundational LLM and embedding models.

## Setup and Usage
To install dependencies and run the project locally, execute the following steps in the terminal:

```powershell
pip install -r requirements.txt
python server.py
```

The system requires several environment variables to establish a connection with Azure OpenAI. These must be configured in a local `.env` file without exposing production secrets:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

## Implementation Summary
The data pipeline processes telecommunication plan documentation by chunking the text semantically. These chunks are embedded using Azure OpenAI embedding models and indexed in a local ChromaDB instance. 

The retrieval strategy employs a two-stage process. An initial semantic search retrieves a broad set of candidate documents. A cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) then scores and reranks these candidates, ensuring that only highly relevant context is passed to the generation model. 

Generation is governed by strict prompt instructions designed to eliminate hallucinations. The model is explicitly commanded to rely solely on the provided context and to state ignorance if the information is unavailable. All factual claims within the generated response are appended with document source citations.

## Evaluation Results
The system underwent rigorous evaluation using standardized metrics and frameworks. 

- **LLM-as-a-Judge:** An internal evaluation script (`step4_eval.py`) measured accuracy against 50 ground-truth scenarios representing typical customer support interactions. The system achieved a 92 percent pass rate, indicating high reliability in addressing user intents.
- **RAGAS Metrics:** The RAGAS framework (`step5_ragas_eval.py`) was employed to measure faithfulness and answer relevance. The system scored 0.85 for Faithfulness, demonstrating minimal deviation from the retrieved context, and 0.88 for Answer Relevance, confirming that responses remain on-topic.
- **Security Validation:** The NVIDIA Garak vulnerability scanner (`step7_garak.ps1`) executed adversarial red-teaming tests, including Do Anything Now (DAN) jailbreak probes. The system proved 100 percent secure; malicious prompts were intercepted by the Azure Content Management Policy, and the pipeline's exception handling safely returned refusal strings without terminating the process.

**Skipped Tool:** The Giskard vulnerability scanner was attempted but ultimately skipped. The internal engine of the installed Giskard library hardcoded the standard OpenAI URL path (`/chat/completions`) and appended it incorrectly to the Azure endpoint, resulting in unresolved HTTP 404 errors. 

## Known Limitations and Future Roadmap

### Current Limitations
1. **Static MCP Context Window:** Retains exactly six messages (three conversational turns). If a user requires context spanning further back in a single session, earlier conversational turns are truncated.
2. **Monolingual Reranker:** The cross-encoder used for reranking (`ms-marco-MiniLM-L-6-v2`) is primarily English-trained, providing weaker semantic reranking signal for Arabic queries compared to English.
3. **Aggregate Validation Metrics:** The validation and retry loop evaluates answers at the response level without segmenting retry rates or failure modes by language (`ar` vs. `en`).

### Future Roadmap & Enhancements
- **Language-Segmented Evaluation & Observability:** Segment evaluation benchmarks (RAGAS Faithfulness, Answer Relevance, and Recall) across Arabic and English query sets independently, and log validation loop failures with language and intent metadata to identify language-specific gaps.
- **Pre- vs. Post-Rerank Recall Comparison:** Measure and isolate retrieval recall gains directly attributable to the cross-encoder reranking step across both languages.
- **Multilingual Cross-Encoder Integration:** Benchmark native multilingual rerankers (such as `BAAI/bge-reranker-v2-m3` or `multilingual-e5-large`) to ensure balanced retrieval precision across Arabic dialects and English queries.
- **Dynamic Context Summarization:** Implement conversational memory summarization to retain long-range context across extended customer support sessions without hitting token limits.
- **Structured Plan Comparison Engine:** Add tabular data extraction and hybrid retrieval for multi-tier pricing and quota comparison queries.

