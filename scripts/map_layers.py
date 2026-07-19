"""Step A — stamp the 3-layer grouping onto every node (design.md step 2).

Adds `meta["layer"]` ∈ {data, argument, question} to each node in `nodes.jsonl`,
derived from its fine-grained `type` via `epistemic_store.layer_of`. The 8 fine types
stay untouched; `layer` is the coarse grouping the pairing funnel blocks on. No LLM.

Idempotent: re-running just refreshes the stamp. `layer` is derived, so the funnel can
also recompute it on the fly (`layer_of(node.type)`) if a re-typing pass drops the
stamp — this script is a convenience, not a source of truth.

Usage:
    python scripts/map_layers.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import read_nodes, write_jsonl, layer_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default=str(ROOT / "artifacts" / "epistemic" / "nodes.jsonl"))
    args = ap.parse_args()

    nodes = read_nodes(args.nodes)
    layers = Counter()
    for n in nodes:
        n.meta["layer"] = layer_of(n.type)
        layers[n.meta["layer"]] += 1

    write_jsonl(args.nodes, nodes)
    print(f"stamped meta.layer on {len(nodes)} nodes -> {args.nodes}")
    print("  layer distribution:", dict(layers))
    print("  by type:", {l: dict(Counter(n.type for n in nodes if n.meta["layer"] == l))
                         for l in ("data", "argument", "question")})


if __name__ == "__main__":
    main()
