import os
import time
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import giskard
from giskard.scan import vulnerability_scan

from step3_graph import run_query

# ==============================================================================
# GISKARD 3.x LLM-AS-A-JUDGE CONFIGURATION
# ==============================================================================
# 1. Giskard 3.x reads standard LiteLLM environment variables
os.environ["AZURE_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
os.environ["AZURE_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
os.environ["AZURE_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini")

# 2. Native Azure initialization
giskard.llm.set_llm_api("azure")
giskard.llm.set_default_models(model=f"azure/{deployment}")
# ==============================================================================

# ==============================================================================

def agent_wrapper(query: str, **kwargs) -> str:
    """Wrapper function that Giskard calls to test our RAG pipeline."""
    # 🛡️ RATE LIMIT PROTECTION: Sleep 2 seconds between every request
    time.sleep(2) 
    try:
        result = run_query(query)
        return result["answer"]
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    print("=" * 60)
    print("🛡️ Step 6 — Giskard Vulnerability Scan (Azure Edition)")
    print("=" * 60)

    print("🚀 Starting automated vulnerability scan. This will take several minutes.")
    print("Note: A 2-second delay is added per request to respect Azure API rate limits.")
    
    report = await vulnerability_scan(
        target=agent_wrapper,
        description="A customer support chatbot for e& Egypt telecom that answers user questions about prepaid plans (Aqwa Card, Ahlan by the Second), Data Line packages, Emerald plans, and Hekaya packages.",
        target_mode="singleturn",
        languages=["en", "ar"]
    )
    
    # Save results
    report.to_html("giskard_report.html")
    print("\n✅ Scan Complete! Report saved to 'giskard_report.html'")

if __name__ == "__main__":
    asyncio.run(main())
