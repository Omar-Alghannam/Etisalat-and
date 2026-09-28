import os
import chromadb

CHROMA_DIR = r"c:\E&_Agentic\chroma_db"

def main():
    if not os.path.exists(CHROMA_DIR):
        print(f"Chroma DB directory not found: {CHROMA_DIR}")
        return

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    collections = ['kb_en', 'kb_ar']
    
    total_problem_chunks = 0
    
    for coll_name in collections:
        try:
            collection = chroma_client.get_collection(name=coll_name)
        except Exception as e:
            print(f"Could not load collection {coll_name}: {e}")
            continue
            
        data = collection.get(include=["metadatas"])
        metadatas = data["metadatas"]
        
        problem_chunks = 0
        examples = []
        
        for meta in metadatas:
            if not meta:
                continue
            plan_name = meta.get("plan_name", "")
            if isinstance(plan_name, str):
                # A heuristic for a sentence: long string, or ends with period, or has "does not include"
                words = plan_name.split()
                if len(words) > 7 or "does not include" in plan_name.lower() or plan_name.endswith("."):
                    problem_chunks += 1
                    if len(examples) < 5:
                        examples.append(plan_name)
                        
        print(f"Collection: {coll_name}")
        print(f"  Total chunks: {len(metadatas)}")
        print(f"  Problem chunks (plan_name is a sentence): {problem_chunks}")
        if examples:
            print("  Examples:")
            for ex in examples:
                print(f"    - {ex}")
        print("-" * 40)
        
        total_problem_chunks += problem_chunks
        
    print(f"Total problem chunks across collections: {total_problem_chunks}")

if __name__ == "__main__":
    main()
