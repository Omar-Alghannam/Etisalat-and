import json
import os
import requests
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load env variables for Azure OpenAI (used for the LLM-as-a-Judge)
load_dotenv()
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")

# Initialize the Judge LLM
judge_llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)

EVAL_DATASET_PATH = "data/eval_dataset_v2.json"
API_URL = "http://127.0.0.1:8000/api/chat"

JUDGE_PROMPT = """You are an impartial evaluator grading a customer service AI.
You will be provided with a Question, the AI's Actual Answer, and the Expected Criteria.

Your job is to determine if the Actual Answer meets the Expected Criteria.
- If the AI says "I don't have information about that", and the criteria expects it to refuse, that is a PASS.
- If the AI provides the requested information accurately, that is a PASS.
- If the AI hallucinates, misses the core info, or fails the criteria, that is a FAIL.

Respond ONLY with a JSON object in this format:
{
    "grade": "PASS" or "FAIL",
    "reason": "One short sentence explaining why."
}
"""

def evaluate_pipeline():
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"❌ Cannot find {EVAL_DATASET_PATH}")
        return

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Auto-detect v1 (list) vs v2 (dict with test_cases key)
    if isinstance(raw, list):
        dataset = raw
        use_v2 = False
    else:
        dataset = raw.get("test_cases", [])
        use_v2 = True

    print(f"🚀 Starting Evaluation of {len(dataset)} test cases...")
    print(f"📡 Testing against API: {API_URL}")
    if use_v2:
        print(f"📋 Dataset: {raw.get('test_suite', 'v2')}")
    print("=" * 60)

    results = []
    passed = 0

    for test in dataset:
        test_id = test["id"]
        # v2 uses 'question', v1 uses 'query'
        query = test.get("question", test.get("query", ""))
        expected = test["expected"]
        category = test.get("category", test.get("type", "unknown"))
        language = test.get("language", "en")

        print(f"\n⏳ Running T{str(test_id).zfill(3)} [{category}|{language}]: {query[:70]}...")

        # 1. Get Answer from our RAG Pipeline API
        try:
            res = requests.post(API_URL, json={"query": query, "chat_history": []}, timeout=30)
            if res.status_code != 200:
                print(f"   ❌ API Error: {res.status_code}")
                actual_answer = f"API Error {res.status_code}"
            else:
                actual_answer = res.json().get("answer", "")
        except requests.exceptions.RequestException:
            print(f"   ❌ Connection Error. Is server.py running?")
            return

        # 2. Grade Answer using LLM-as-a-Judge
        eval_query = (
            f"QUESTION: {query}\n"
            f"EXPECTED CRITERIA: {expected}\n\n"
            f"ACTUAL ANSWER:\n{actual_answer}"
        )

        try:
            judge_response = judge_llm.invoke([
                SystemMessage(content=JUDGE_PROMPT),
                HumanMessage(content=eval_query)
            ])
            raw_eval = judge_response.content.strip().replace("```json", "").replace("```", "")
            eval_data = json.loads(raw_eval)
        except Exception as e:
            print(f"   ❌ Judge Error: {e}")
            eval_data = {"grade": "FAIL", "reason": "Judge LLM failed to parse."}

        grade = eval_data.get("grade", "FAIL")
        reason = eval_data.get("reason", "")

        if grade == "PASS":
            passed += 1
            print(f"   ✅ PASS: {reason}")
        else:
            print(f"   ❌ FAIL: {reason}")

        results.append({
            "id": test_id,
            "query": query,
            "category": category,
            "language": language,
            "expected": expected,
            "actual": actual_answer,
            "grade": grade,
            "reason": reason
        })

    # 3. Generate Markdown Report
    report_path = "eval_report.md"
    score = (passed / len(dataset)) * 100

    # Group results by category
    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"pass": 0, "fail": 0})
        if r["grade"] == "PASS":
            by_category[cat]["pass"] += 1
        else:
            by_category[cat]["fail"] += 1

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"**Overall Score:** {passed}/{len(dataset)} ({score:.1f}%)\n\n")

        # Category breakdown table
        f.write("## Score by Category\n\n")
        f.write("| Category | Pass | Fail | Score |\n")
        f.write("|---|---|---|---|\n")
        for cat, counts in sorted(by_category.items()):
            total_cat = counts["pass"] + counts["fail"]
            cat_pct = (counts["pass"] / total_cat) * 100
            f.write(f"| {cat} | ✅ {counts['pass']} | ❌ {counts['fail']} | {cat_pct:.0f}% |\n")

        # Summary table
        f.write("\n## Summary Table\n\n")
        f.write("| ID | Category | Lang | Grade | Reason |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            icon = "✅" if r["grade"] == "PASS" else "❌"
            f.write(f"| T{str(r['id']).zfill(3)} | {r['category']} | {r['language']} | {icon} {r['grade']} | {r['reason']} |\n")

        f.write("\n## Detailed Breakdown\n")
        for r in results:
            f.write(f"\n### T{str(r['id']).zfill(3)} — {r['category']} ({r['grade']})\n")
            f.write(f"- **Language:** {r['language']}\n")
            f.write(f"- **Query:** {r['query']}\n")
            f.write(f"- **Expected:** {r['expected']}\n")
            f.write(f"- **Actual:** {r['actual'].replace(chr(10), ' ')}\n")
            f.write(f"- **Judge Reason:** {r['reason']}\n")

    print("=" * 60)
    print(f"🎯 Final Score: {passed}/{len(dataset)} ({score:.1f}%)")
    print(f"📄 Report saved to: {report_path}")


if __name__ == "__main__":
    evaluate_pipeline()
