import json
import os
import math
import sys
import warnings
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from unittest.mock import MagicMock

# --- Compatibility shim ---
# ragas internally imports ChatVertexAI from langchain_community which was removed.
# We don't use VertexAI at all, so we mock it to prevent the crash.
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())

# Suppress RAGAS deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- RAGAS v1.0 new API ---
try:
    # RAGAS >= 0.2 / v1.0 API
    from ragas import EvaluationDataset, evaluate, SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_NEW_API = True
    print("✅ Using RAGAS v1.0 API")
except ImportError:
    # RAGAS < 0.2 legacy API
    from ragas import evaluate
    from ragas.metrics import faithfulness as Faithfulness, answer_relevancy as AnswerRelevancy
    from datasets import Dataset
    RAGAS_NEW_API = False
    print("✅ Using RAGAS legacy API")


def main():
    load_dotenv()

    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", AZURE_ENDPOINT)
    AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", AZURE_API_KEY)
    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    # Build the LangChain LLM
    langchain_llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        azure_deployment=AZURE_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
        temperature=0.0,
    )

    # Build the LangChain Embeddings
    from langchain_openai import AzureOpenAIEmbeddings
    langchain_embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_key=AZURE_EMBEDDING_API_KEY,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_API_VERSION,
    )

    EVAL_DATASET_PATH = "data/eval_dataset_v2.json"
    API_URL = "http://127.0.0.1:8000/api/chat"

    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"❌ Cannot find {EVAL_DATASET_PATH}")
        return

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset_json = raw if isinstance(raw, list) else raw.get("test_cases", [])

    # Skip out-of-scope tests — they deliberately say "I don't know"
    # which would score 0 faithfulness and skew the results unfairly
    SKIP_CATEGORIES = {"out_of_scope", "adversarial_grounding"}
    filtered = [t for t in dataset_json if t.get("category", "") not in SKIP_CATEGORIES]

    print(f"\n🚀 Collecting {len(filtered)} answers (skipping {len(dataset_json) - len(filtered)} out-of-scope)...")
    print(f"📡 API: {API_URL}\n")

    rows = []  # list of dicts: question, answer, contexts

    for i, test in enumerate(filtered):
        query = test.get("question", test.get("query", ""))
        print(f"  [{i+1}/{len(filtered)}] {query[:60]}...")
        success = False
        for attempt in range(3):  # retry up to 3 times
            try:
                res = requests.post(API_URL, json={"query": query, "chat_history": []}, timeout=60)
                if res.status_code == 200:
                    res_data = res.json()
                    answer = res_data.get("answer", "")
                    contexts = res_data.get("contexts", [])
                    if not contexts:
                        contexts = ["No context retrieved."]
                    rows.append({"question": query, "answer": answer, "contexts": contexts})
                    success = True
                    break
                else:
                    print(f"    ⚠️ API Error {res.status_code}, attempt {attempt+1}/3")
            except requests.exceptions.ReadTimeout:
                print(f"    ⚠️ Timeout on attempt {attempt+1}/3, retrying...")
            except Exception as e:
                print(f"    ❌ Request failed: {e}")
                break
        if not success:
            print(f"    ⏭️  Skipping this query after 3 failed attempts")

    if not rows:
        print("❌ No samples collected. Is server.py running?")
        return

    print(f"\n⏳ Running RAGAS on {len(rows)} samples...")
    print("   Metrics: Faithfulness (hallucinations) & Answer Relevancy (helpfulness)\n")

    if RAGAS_NEW_API:
        # v1.0 API: wrap LLM and embeddings
        wrapped_llm = LangchainLLMWrapper(langchain_llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        m_faith = Faithfulness(llm=wrapped_llm)
        m_rel = AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings)

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
            )
            for r in rows
        ]
        eval_dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=eval_dataset, metrics=[m_faith, m_rel])
    else:
        # Legacy API
        from datasets import Dataset
        ds = Dataset.from_dict({
            "question": [r["question"] for r in rows],
            "answer":   [r["answer"]   for r in rows],
            "contexts": [r["contexts"] for r in rows],
        })
        result = evaluate(ds, metrics=[Faithfulness, AnswerRelevancy], llm=langchain_llm, embeddings=langchain_embeddings)

    print("\n✅ RAGAS Evaluation Complete!")
    print(result)

    # --- Save ---
    df = result.to_pandas()
    df.to_csv("ragas_results.csv", index=False)

    def safe_avg(raw):
        """Average a list of floats, ignoring NaN. Falls back to scalar float."""
        if isinstance(raw, list):
            valid = [x for x in raw if isinstance(x, float) and not math.isnan(x)]
            return float(sum(valid) / len(valid)) if valid else float('nan')
        return float(raw)

    faith_val = safe_avg(result["faithfulness"])
    grade_f = "🟢 Excellent" if faith_val >= 0.8 else ("🟡 Good" if faith_val >= 0.6 else "🔴 Needs Work")

    rel_val = safe_avg(result["answer_relevancy"])
    grade_r = "🟢 Excellent" if rel_val >= 0.8 else ("🟡 Good" if rel_val >= 0.6 else "🔴 Needs Work")

    with open("ragas_report.md", "w", encoding="utf-8") as f:
        f.write("# RAGAS Evaluation Report\n\n")
        f.write(f"**Test Cases:** {len(rows)} (factual, comparison, recommendation, cross-lingual)\n\n")
        f.write("| Metric | Score | Grade |\n|---|---|---|\n")
        f.write(f"| Faithfulness | {faith_val:.4f} | {grade_f} |\n")
        f.write(f"| Answer Relevancy | {rel_val:.4f} | {grade_r} |\n\n")
        f.write("> **Faithfulness** = fraction of statements directly supported by context. 1.0 = no hallucinations.\n")
        f.write("> **Answer Relevancy** = how well the answer addresses the user's specific question. 1.0 = perfectly on topic.\n\n")
        f.write("## Per-Question Breakdown\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            f.write(df.to_string(index=False))

    print(f"\n📄 Report saved to ragas_report.md and ragas_results.csv")
    print(f"   Faithfulness:     {faith_val:.4f}  {grade_f}")
    print(f"   Answer Relevancy: {rel_val:.4f}  {grade_r}")


if __name__ == "__main__":
    main()
