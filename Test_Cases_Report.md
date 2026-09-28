# Etisalat Egypt RAG Chatbot — Test Cases Report

This document outlines a series of test cases designed to evaluate the accuracy, retrieval capabilities, and constraints of the RAG Chatbot. The tests are categorized by difficulty: Easy (direct retrieval), Medium (synthesis and comparison), and Hard (edge cases, out-of-scope, and complex constraints).

---

## 1. Easy Level: Direct Factual Retrieval
*These test cases evaluate the chatbot's ability to pull a specific fact from a single document chunk. The answers should be exact and straightforward.*

**Test Case 1.1: Direct Price Lookup (English)**
*   **Prompt:** "What is the price of the Aqwa Card 19 recharge?"
*   **Expected Output:** 19 EGP.
*   **Success Criteria:** Accurate price retrieval, cites "Aqwa Card 19" as the source.

**Test Case 1.2: Direct Validity Lookup (Arabic)**
*   **Prompt:** "ما هي صلاحية باقة خط الداتا 1.8 جيجا؟" (What is the validity of the Data Line 1.8 GB plan?)
*   **Expected Output:** One month (شهر واحد).
*   **Success Criteria:** Accurate validity period, cites the correct Arabic Data Line plan.

**Test Case 1.3: Subscription Code Lookup**
*   **Prompt:** "What is the subscription code for Hekaya Mixat 52?"
*   **Expected Output:** `*319*45#`
*   **Success Criteria:** Exact code provided, cites "Hekaya Mixat 52".

---

## 2. Medium Level: Synthesis & Comparison
*These test cases require the chatbot to retrieve multiple chunks, combine information, or compare two different plans.*

**Test Case 2.1: Plan Comparison**
*   **Prompt:** "What is the difference between Hekaya Mixat 52 and Hekaya Mixat 69?"
*   **Expected Output:** A comparison highlighting the price (74.29 EGP vs. 98.5 EGP including tax), the number of Mixes (1,900 vs. 2,800), and any differences in family lines or preferred numbers.
*   **Success Criteria:** Clear comparison format, accurately pulling specs from both plans, citing both as sources.

**Test Case 2.2: Finding a Plan by Constraint**
*   **Prompt:** "I need an internet plan that gives me at least 10,000 MB. What do you recommend?"
*   **Expected Output:** Suggests a plan like Hekaya Internet 210 (which gives 12,000 MB) or similar.
*   **Success Criteria:** The bot successfully searches the chunks for data limits and recommends a plan that meets or exceeds the criteria.

**Test Case 2.3: Cross-Product Knowledge**
*   **Prompt:** "Does the Emerald package include home internet, and how does it compare to Hekaya's home internet?"
*   **Expected Output:** Details the home internet offerings under Emerald (e.g., VDSL routers/speeds) and contrasts them with Hekaya's offerings (if any).
*   **Success Criteria:** Synthesizes information from both the `Emerald` and `Hekaya` documents without mixing them up.

---

## 3. Hard Level: Edge Cases, Out-of-Scope & Complex Logic
*These test cases push the limits of the system to ensure it handles invalid questions gracefully, avoids hallucinations, and understands complex billing rules.*

**Test Case 3.1: Strict Out-of-Scope Prevention**
*   **Prompt:** "How do I setup a Vodafone cash wallet?"
*   **Expected Output:** "I don't have information about that in my knowledge base." (Or the Arabic equivalent).
*   **Success Criteria:** The bot strictly adheres to the system prompt and refuses to answer questions about competitors or unrelated topics. It does NOT hallucinate an answer.

**Test Case 3.2: Hallucination Check (Fake Plan)**
*   **Prompt:** "Tell me about the Super Mega Etisalat 5000 plan."
*   **Expected Output:** "I don't have information about that in my knowledge base."
*   **Success Criteria:** The bot does not invent features for a non-existent plan, even if the prompt implies it exists.

**Test Case 3.3: Complex Billing / Mathematical Logic**
*   **Prompt:** "If I am on a Hekaya Mixat plan and I call a Vodafone number for 10 minutes, how many Mixes will be deducted?"
*   **Expected Output:** 50 Mixes. (Since 1 minute to other networks = 5 Mixes).
*   **Success Criteria:** The bot correctly retrieves the conversion rate (1 min to other networks = 5 units/mixes) and applies the basic math.

**Test Case 3.4: Cross-Lingual Testing (Mixing Languages)**
*   **Prompt:** "كم سعر الـ Emerald 1000 plan?"
*   **Expected Output:** Retrieves the price for Emerald 1000 and responds accurately.
*   **Success Criteria:** The system successfully handles a query containing both Arabic and English words, retrieving the correct chunk from the vector database.

---

## 4. Very Hard Level: Conversational Memory & Context Resolution
*These test cases evaluate whether the chatbot can retain context across multiple conversational turns, resolving pronouns and remembering previous constraints.*

**Test Case 4.1: Pronoun Resolution**
*   **Turn 1:** "Tell me about Hekaya Mixat 120."
*   **Turn 2:** "Does it include home internet?"
*   **Expected Output:** The bot recognizes "it" refers to Hekaya Mixat 120, and correctly answers that it does not (only 180 and 270 do).
*   **Success Criteria:** Correctly resolves the pronoun from the chat history and retrieves accurate constraints.

**Test Case 4.2: Context Switching & Comparison**
*   **Turn 1:** "I want a plan with 25 GB of data." (Bot suggests Data Line 25 GB for 375 EGP).
*   **Turn 2:** "What about one with 45 GB?"
*   **Expected Output:** The bot remembers we are discussing "Data Line" internet plans and quotes the Data Line 45 GB plan (600 EGP).
*   **Success Criteria:** The bot infers the context of "Data Line" from the previous turn without needing the user to explicitly say the product family again.

**Test Case 4.3: Accumulating Constraints (Constraint Memory)**
*   **Turn 1:** "I need a prepaid plan under 100 EGP."
*   **Turn 2:** "Actually, make sure it has at least 2,000 Mixes."
*   **Expected Output:** The bot remembers the < 100 EGP constraint *and* the > 2,000 Mixes constraint, realizing that Hekaya Mixat 69 (98.5 EGP for 2,800 Mixes) fits perfectly.
*   **Success Criteria:** The bot combines constraints across multiple messages and provides an accurate, synthesized recommendation.
