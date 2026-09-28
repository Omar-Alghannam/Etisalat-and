"""
Model Context Protocol (MCP) Server for e& Egypt RAG Chatbot.
Uses HTTP to call the already-running FastAPI server at localhost:8000.
This avoids dependency conflicts between Python environments.
"""

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Etisalat Egypt Support")

API_URL = "http://127.0.0.1:8000/api/chat"

# Global memory store for multi-turn conversations during an MCP session
_chat_history = []

@mcp.tool()
def ask_etisalat_support(query: str) -> str:
    """
    Ask the Etisalat Egypt (e&) AI Assistant a question about prepaid plans,
    DataLine, Hekaya Mixat, Emerald, or other telecommunications services.
    Requires the FastAPI server to be running at localhost:8000.

    Args:
        query: The specific question to ask the telecom knowledge base.
    """
    import sys
    global _chat_history
    print(f"📡 MCP Tool Called: {query}", file=sys.stderr)

    try:
        response = requests.post(
            API_URL,
            json={"query": query, "chat_history": _chat_history[-6:]}, # Pass the last 6 messages (3 turns)
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        print(f"📥 API Response keys: {list(data.keys())}", file=sys.stderr)

        # Server returns "answer" field
        answer = data.get("answer") or data.get("response") or ""
        sources = data.get("sources", [])

        if not answer:
            # Return raw data for debugging if answer is missing
            return f"Tool reached the server but got an empty answer. Raw response: {data}"

        # Save to memory
        _chat_history.append({"role": "user", "content": query})
        _chat_history.append({"role": "assistant", "content": answer})

        output = answer
        if sources:
            source_labels = [s if isinstance(s, str) else s.get("label", str(s)) for s in sources]
            output += f"\n\n---\n**Sources:** {', '.join(source_labels)}"
        return output

    except requests.exceptions.ConnectionError:
        return "❌ Error: The Etisalat chatbot server is not running. Please start it with: python server.py"
    except Exception as e:
        return f"❌ Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
