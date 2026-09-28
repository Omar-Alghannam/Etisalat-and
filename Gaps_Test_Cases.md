# Gap Analysis Test Cases

Now that we have upgraded the system with Memory, Subheader Chunking, and Cross-Lingual Fallback, we need to test the *edges* of these new features. These test cases are specifically designed to try and "break" the current architecture to reveal gaps.

---

## 1. Testing Memory Limits & Pronoun Confusion
*Goal: See if the LLM gets confused when switching contexts rapidly in the chat history.*

**Test Case 1.1: The "It" Trap**
*   **Message 1:** "Tell me about Hekaya Mixat 52."
*   **Message 2:** "Now tell me about Aqwa Card 19."
*   **Message 3:** "Does *it* include WhatsApp?"
*   **Expected Output:** The bot should know "it" refers to Aqwa Card 19 (the most recent context) and answer accordingly.
*   **Potential Gap:** The retriever might search ChromaDB for "Does it include WhatsApp?", which has no strong keywords, causing it to retrieve random chunks. (We didn't build a "Query Reformulator" node to rewrite "it" into "Aqwa Card 19" before searching).

## 2. Testing Subheader Chunking Isolation
*Goal: See if splitting the text into smaller subheaders stripped too much context from the chunks.*

**Test Case 2.1: The Isolated Term**
*   **Message 1:** "What happens if I don't renew the Data Line package on time?"
*   **Expected Output:** Retrieves the specific subheader chunk about Data Line renewal rules and answers accurately.
*   **Potential Gap:** Because we split subheaders into their own chunks, the actual text of the chunk might just say *"If not renewed on time, speed drops."* without explicitly saying *"Data Line"* in the paragraph text. If the embedding model doesn't link it strongly enough to the user's query, it might fail to retrieve it.

## 3. Testing Cross-Lingual Fallback
*Goal: We pull 8 main chunks and 2 fallback chunks from the opposite language. Is 2 enough?*

**Test Case 3.1: Heavy Fallback Reliance**
*   **Message 1:** (In Arabic) "أخبرني بكل تفاصيل باقة Emerald 1000" (Tell me all details of Emerald 1000).
*   **Expected Output:** It should pull the details from the English database and translate the answer to Arabic.
*   **Potential Gap:** Since Emerald 1000 only exists in the English PDF, the Arabic retrieval will fail, and it relies entirely on the 2 fallback English chunks. If the Emerald 1000 plan is very long, 2 chunks might not be enough to hold "all details."

## 4. Testing Multi-Constraint Retrieval
*Goal: Push the top-K limit (NUM_RESULTS = 8).*

**Test Case 4.1: The Broad Comparison**
*   **Message 1:** "List all the plans that cost exactly 100 EGP across all products."
*   **Expected Output:** Finds and lists every 100 EGP plan.
*   **Potential Gap:** ChromaDB retrieves the 8 *most similar* chunks. A query about "100 EGP" might retrieve 8 chunks from the same product family and miss a 100 EGP plan in a completely different document because it was chunk #9 in the similarity rankings.

---
### How to Use This Document:
Run these exactly as written in your Streamlit app. If it fails any of these, it means we have found a "Gap" and we can build a specific LangGraph node to fix it (e.g., adding a Query Rewrite node for Test 1.1).
