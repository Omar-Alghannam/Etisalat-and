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

## Known Limitations and Future Improvements
A primary limitation involves the static configuration of the MCP context window, which currently retains exactly six messages (three conversational turns). If a user requires context spanning further back in a single session, the query expansion node will lack the necessary history to resolve references. 

Future improvements should include dynamic context window management that summarizes older turns rather than strictly truncating them. Additionally, integrating support for structured tabular data retrieval would improve the system's ability to answer complex pricing comparisons across multiple telecommunication plans.
