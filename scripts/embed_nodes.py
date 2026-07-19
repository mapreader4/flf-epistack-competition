"""Step B — embed the nodes (design.md step 5).

Runs `intfloat/multilingual-e5-large-instruct` (1024-dim, the only serverless
embedding model on our Together account) over each node's `canonical_text` and writes:
  - `artifacts/epistemic/embeddings.npy`         float32 [N × 1024], row i = node_ids[i]
  - `artifacts/epistemic/embeddings_index.json`  {model, dim, node_ids}

e5-instruct is asymmetric-trained but we embed every node with the SAME instruction
template, so cosine between two node vectors is symmetric and meaningful. NOTE the
compressed cosine range (unrelated ~0.77, median ~0.84, synonyms ≥0.96): downstream
the funnel ranks by cosine (top-k), it never thresholds on an absolute value.

Mirrors `type_claims.py`: Together/OpenAI client, incremental cache keyed on
(model, node_id) so re-runs are free, token budget with partial-save.

Usage:
    python scripts/embed_nodes.py                 # all nodes
    python scripts/embed_nodes.py --limit 20      # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import read_nodes  # noqa: E402

load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_EMBED = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large-instruct")
DIM = 1024
# e5-instruct wants an instruction; use one task string for every node so the space
# is internally consistent (we compare nodes to nodes, not query to doc).
INSTRUCT = "Instruct: Retrieve statements that are epistemically related (support, attack, or restate) to the given statement.\nQuery: "


def build_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, max_retries=3, timeout=300)


def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=[INSTRUCT + t for t in texts])
    # OpenAI-compatible responses may not preserve input order guarantees; sort by index.
    rows = sorted(resp.data, key=lambda d: d.index)
    return [r.embedding for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_EMBED)
    ap.add_argument("--nodes", default=str(ROOT / "artifacts" / "epistemic" / "nodes.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "epistemic" / "embeddings.npy"))
    ap.add_argument("--index", default=str(ROOT / "artifacts" / "epistemic" / "embeddings_index.json"))
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "epistemic" / "embed_cache.json"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    nodes = read_nodes(args.nodes)
    if args.limit:
        nodes = nodes[:args.limit]
    node_ids = [n.node_id for n in nodes]

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[float]] = {}
    if cache_path.exists() and not args.no_cache:
        cache = json.loads(cache_path.read_text())

    print(f"Model:  {args.model} @ Together ({DIM}-dim)")
    print(f"Nodes:  {len(nodes)}  (batch {args.batch_size})\n")

    client = build_client()
    todo = [n for n in nodes if f"{args.model}|{n.node_id}" not in cache]
    print(f"  {len(nodes) - len(todo)} cached, {len(todo)} to embed")

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        vecs = embed_batch(client, args.model, [n.canonical_text for n in batch])
        for n, v in zip(batch, vecs):
            cache[f"{args.model}|{n.node_id}"] = v
        cache_path.write_text(json.dumps(cache))
        print(f"  embedded {min(i + args.batch_size, len(todo))}/{len(todo)}")

    # Assemble the matrix in node order (row-aligned to index.json).
    mat = np.array([cache[f"{args.model}|{nid}"] for nid in node_ids], dtype=np.float32)
    assert mat.shape == (len(node_ids), DIM), mat.shape
    np.save(args.out, mat)
    Path(args.index).write_text(json.dumps(
        {"model": args.model, "dim": DIM, "node_ids": node_ids}, indent=2))

    # sanity: cosine stats on a sample so the compressed range is visible
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sample = norm[:min(200, len(norm))]
    cos = sample @ sample.T
    off = cos[~np.eye(len(sample), dtype=bool)]
    print(f"\nwrote {mat.shape} -> {args.out}")
    print(f"  index -> {args.index}")
    print(f"  cosine (sample off-diagonal): min {off.min():.3f}  "
          f"median {np.median(off):.3f}  max {off.max():.3f}")


if __name__ == "__main__":
    main()
