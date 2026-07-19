"""Step 9 helper — question → entry node (design.md query layer).

Maps a natural-language question to the graph node a query should START from, by
embedding the question with the same e5 template the nodes were embedded with and
taking the nearest question/hypothesis-layer node(s). This is the missing piece that
lets Query 1 (evidence-for) and Query 3 (weakest-link) begin at the RIGHT node instead
of a hand-picked one.

NOTE: not every query maps to a single node. Query 2 ("where do sources disagree?") is
a whole-graph scan, not a lookup — the caller should skip resolution for it.

    resolve("Did COVID start at the Huanan market?") -> [(node_id, score, text), ...]
    python scripts/resolve_question.py "your question here"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import read_nodes, layer_of          # noqa: E402
from embed_nodes import build_client, embed_batch, DEFAULT_EMBED  # reuse the SAME template


def resolve(question: str, top_n: int = 3, layers: tuple[str, ...] = ("question",),
            w_support: float = 0.4, shortlist: int = 12,
            nodes_path: str | None = None, emb_path: str | None = None,
            index_path: str | None = None, cards_path: str | None = None
            ) -> list[tuple[str, float, str]]:
    """Return the top_n entry-node candidates. Cosine picks a topical shortlist, then
    we re-rank blending cosine with support-richness (# incoming support cards,
    normalized) so a well-connected hypothesis beats a topically-similar orphan —
    exactly the n-00024(5-cards) vs n-00502(14-cards) failure. Deterministic."""
    base = ROOT / "artifacts" / "epistemic"
    nodes = read_nodes(nodes_path or base / "nodes.jsonl")
    byid = {n.node_id: n for n in nodes}
    mat = np.load(emb_path or base / "embeddings.npy")
    idx = json.loads(Path(index_path or base / "embeddings_index.json").read_text())
    rowof = {nid: i for i, nid in enumerate(idx["node_ids"])}
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    # support-card count per node (for richness weighting)
    supct: dict[str, int] = {}
    cpath = Path(cards_path or base / "cards.jsonl")
    if cpath.exists():
        for line in cpath.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                if c.get("kind") == "RA":
                    supct[c["target"]] = supct.get(c["target"], 0) + 1

    cand = [n.node_id for n in nodes if layer_of(n.type) in layers]
    qvec = np.asarray(embed_batch(build_client(), DEFAULT_EMBED, [question])[0], dtype=np.float32)
    qn = qvec / (np.linalg.norm(qvec) + 1e-9)
    by_cos = sorted(((nid, float(qn @ norm[rowof[nid]])) for nid in cand), key=lambda t: -t[1])
    pool = by_cos[:shortlist]
    cap = max((supct.get(nid, 0) for nid, _ in pool), default=0) or 1
    blended = sorted(pool, key=lambda t: -(t[1] + w_support * (supct.get(t[0], 0) / cap)))
    return [(nid, s, byid[nid].canonical_text) for nid, s in blended[:top_n]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--layers", default="question",
                    help="comma-separated layers to search (default: question)")
    args = ap.parse_args()
    hits = resolve(args.question, top_n=args.top_n, layers=tuple(args.layers.split(",")))
    print(f'Q: {args.question}\n')
    for nid, score, text in hits:
        print(f"  {nid}  cos={score:.3f}  {text[:80]}")


if __name__ == "__main__":
    main()
