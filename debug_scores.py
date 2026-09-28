from agent.planner import retrieve

for query in [
    "paper about improving RAG reliability with an evaluator",
    "framework that corrects bad retrieval results",
]:
    print(f"=== Query: {query} ===")
    candidates = retrieve(query, n_results=5)
    for c in candidates:
        print(f"{c.score:.3f} | {c.path} | {c.metadata.get('match_type')}")
    print()