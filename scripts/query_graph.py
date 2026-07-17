"""Query an already-built HippoRAG 2 graph: retrieval + LLM-generated answer.

Reuses the Together wiring from run_hipporag_index.py, but skips hipporag.index()
-- HippoRAG loads the persisted graph/embeddings from save_dir's working
directory (keyed on llm+embed model names) the moment retrieve()/rag_qa() is
called, so this only works after that save_dir has already been indexed.

Usage:
    python scripts/query_graph.py --save-dir outputs/hipporag_claims \\
        "Does the furin cleavage site favor the lab-leak hypothesis?"

    python scripts/query_graph.py --save-dir outputs/hipporag_claims \\
        --queries-file questions.txt
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling module

from run_hipporag_index import DEFAULT_LLM, DEFAULT_EMBED, build_hipporag  # noqa: E402

DEFAULT_QUERIES = [
    "What is the final verdict on whether COVID-19 originated from a lab leak or zoonosis?",
    "What role does the furin cleavage site play in the argument?",
    "What prior probabilities did the debaters assign to each hypothesis?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="*", help="query strings (default: a few sample questions)")
    ap.add_argument("--queries-file", help="path to a text file, one query per line")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--save-dir", required=True, help="e.g. outputs/hipporag_claims")
    ap.add_argument("--synonymy-threshold", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.queries_file:
        queries = [line.strip() for line in Path(args.queries_file).read_text().splitlines() if line.strip()]
    elif args.queries:
        queries = args.queries
    else:
        queries = DEFAULT_QUERIES

    hipporag, embed_provider = build_hipporag(
        args.save_dir, args.llm, args.embed, args.synonymy_threshold)
    print(f"LLM:         {args.llm} @ Together")
    print(f"Embeddings:  {args.embed} @ {embed_provider}")
    print(f"Save dir:    {args.save_dir}")
    print(f"Queries:     {len(queries)}\n")

    query_solutions, response_messages, metadata = hipporag.rag_qa(queries=queries)

    for sol in query_solutions:
        print("=" * 80)
        print(f"Q: {sol.question}")
        print(f"\nA: {sol.answer}\n")
        print(f"Top {min(args.top_k, len(sol.docs))} retrieved:")
        for i in range(min(args.top_k, len(sol.docs))):
            score = sol.doc_scores[i] if sol.doc_scores is not None else None
            score_str = f"{score:.4f}" if score is not None else "?"
            snippet = sol.docs[i].replace("\n", " ")[:150]
            print(f"  [{score_str}] {snippet}")
        print()

    out_path = Path(args.save_dir) / "query_results.json"
    out_path.write_text(json.dumps([s.to_dict() for s in query_solutions], indent=2))
    print(f"exported -> {out_path}")


if __name__ == "__main__":
    main()
