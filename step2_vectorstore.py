"""
Step 2: Vector Store — Load chunks into ChromaDB with Azure OpenAI embeddings.

Creates two collections:
  - kb_en: English chunks
  - kb_ar: Arabic chunks

Embedding model: Azure OpenAI (text-embedding-ada-002)

Usage:
    python step2_vectorstore.py
"""

import os
import json
import chromadb
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings


# ---------------------------------------------------------------------------
# Config & Azure OpenAI Setup
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", AZURE_ENDPOINT)
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", AZURE_API_KEY)
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")


from chromadb.api.types import EmbeddingFunction

class AzureOpenAIEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function wrapping AzureOpenAIEmbeddings."""
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
            api_key=AZURE_EMBEDDING_API_KEY,
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
            api_version=AZURE_API_VERSION,
        )

    def name(self) -> str:
        return "azure_openai_embeddings"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(input)

    def embed_query(self, input: list[str] | str) -> list[list[float]]:
        if isinstance(input, str):
            return [self.embeddings.embed_query(input)]
        return self.embeddings.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(input)


# ---------------------------------------------------------------------------
# Load chunks from JSON
# ---------------------------------------------------------------------------

def load_chunks(filepath: str) -> list[dict]:
    """Load chunks from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Create a unique ID for each chunk
# ---------------------------------------------------------------------------

def make_chunk_id(chunk: dict, index: int) -> str:
    """
    Create a unique ID for a chunk.
    Format: {product_family}_{chunk_index}_{index}
    """
    meta = chunk["metadata"]
    family = meta["product_family"].replace(" ", "_").replace("/", "_")
    return f"{family}_{meta['chunk_index']}_{index}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Set up the embedding function
    print(f"🔧 Loading Azure OpenAI Embeddings: {AZURE_EMBEDDING_DEPLOYMENT}")
    embedding_fn = AzureOpenAIEmbeddingFunction()

    # 2. Create ChromaDB persistent client
    print(f"\n📁 Creating ChromaDB store at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 3. Load English chunks and create collection
    en_path = os.path.join(DATA_DIR, "chunks_en.json")
    ar_path = os.path.join(DATA_DIR, "chunks_ar.json")

    if not os.path.exists(en_path) or not os.path.exists(ar_path):
        print("❌ Chunk files not found! Run step1_parse.py first.")
        return

    en_chunks = load_chunks(en_path)
    ar_chunks = load_chunks(ar_path)

    # 4. Create (or reset) collections
    # delete existing collections if they exist, so we start fresh
    for name in ["kb_en", "kb_ar"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # collection doesn't exist yet

    kb_en = client.create_collection(
        name="kb_en",
        embedding_function=embedding_fn,
        metadata={"description": "English knowledge base chunks"},
    )

    kb_ar = client.create_collection(
        name="kb_ar",
        embedding_function=embedding_fn,
        metadata={"description": "Arabic knowledge base chunks"},
    )

    # 5. Add English chunks
    print(f"\n📥 Adding {len(en_chunks)} English chunks to 'kb_en'...")
    # ChromaDB accepts batches — let's add in batches of 50
    batch_size = 50
    for start in range(0, len(en_chunks), batch_size):
        batch = en_chunks[start : start + batch_size]
        kb_en.add(
            ids=[make_chunk_id(c, start + i) for i, c in enumerate(batch)],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"   ✅ Added batch {start // batch_size + 1} ({len(batch)} chunks)")

    # 6. Add Arabic chunks
    print(f"\n📥 Adding {len(ar_chunks)} Arabic chunks to 'kb_ar'...")
    for start in range(0, len(ar_chunks), batch_size):
        batch = ar_chunks[start : start + batch_size]
        kb_ar.add(
            ids=[make_chunk_id(c, start + i) for i, c in enumerate(batch)],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"   ✅ Added batch {start // batch_size + 1} ({len(batch)} chunks)")

    # 7. Verify — run a quick test query on each collection
    print("\n" + "=" * 60)
    print("🧪 Verification — test queries")
    print("=" * 60)

    # English test
    en_results = kb_en.query(
        query_texts=["What is the price of Aqwa Card 19?"],
        n_results=3,
    )
    print("\n🔍 English query: 'What is the price of Aqwa Card 19?'")
    print(f"   Top {len(en_results['documents'][0])} results:")
    for i, (doc, meta) in enumerate(
        zip(en_results["documents"][0], en_results["metadatas"][0])
    ):
        print(f"   [{i+1}] {meta['plan_name']} ({meta['product_family']})")
        print(f"       {doc[:100]}...")

    # Arabic test
    ar_results = kb_ar.query(
        query_texts=["ما هو سعر أقوى كارت 19؟"],
        n_results=3,
    )
    print("\n🔍 Arabic query: 'ما هو سعر أقوى كارت 19؟'")
    print(f"   Top {len(ar_results['documents'][0])} results:")
    for i, (doc, meta) in enumerate(
        zip(ar_results["documents"][0], ar_results["metadatas"][0])
    ):
        print(f"   [{i+1}] {meta['plan_name']} ({meta['product_family']})")
        print(f"       {doc[:100]}...")

    print(f"\n✅ Step 2 complete!")
    print(f"   kb_en: {kb_en.count()} documents")
    print(f"   kb_ar: {kb_ar.count()} documents")
    print(f"   Stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
