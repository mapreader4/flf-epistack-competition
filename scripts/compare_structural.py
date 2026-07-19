"""Structural comparison — test the system on what the graph is FOR, not on
single-document evidence retrieval (the naive baseline's home turf).

Two tests where structure beats "read everything":

  WEAKEST-LINK: "Which single piece of evidence, if removed, most changes the
    conclusion?" The graph runs ablate() — remove each card, re-score, rank by |Δ| —
    deterministic and instant. The baseline has to reason it out from scratch every
    time, with no way to actually measure the counterfactual.

  AUDIT / DETERMINISM: ask the same question twice. The graph is byte-identical every
    run (pure arithmetic). We measure whether the baseline reproduces itself.

Multi-document reasoning — the graph's biggest edge — is NOT testable here: we only
have one document. That test needs 3-5 sources whose combined text exceeds one context
window; see the note printed at the end.

    python scripts/compare_structural.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import read_nodes, read_cards          # noqa: E402
from score_dfquad import ablate                             # noqa: E402
from compare_vs_baseline import ask, DOC, MODEL             # noqa: E402 (reuse ask+doc)
from query_epistemic import build_client                   # noqa: E402
import resolve_question                                     # noqa: E402

ART = ROOT / "artifacts" / "epistemic"
HYP_Q = "The COVID-19 outbreak originated via zoonotic spillover at the Huanan Seafood Market."
WL_Q = ("In this document's argument about COVID-19 origins, which SINGLE piece of "
        "evidence, if removed, would change the conclusion the most? Name it specifically.")


def graph_weakest_link(nodes, cards, target):
    byid = {n.node_id: n for n in nodes}
    cbyid = {c.card_id: c for c in cards}
    ranking = ablate(nodes, cards, target)           # {card_id: |Δ|}, sorted desc
    out = []
    for cid, delta in list(ranking.items())[:3]:
        c = cbyid[cid]
        prem = byid[c.premises[0]]
        out.append({"card_id": cid, "delta": round(delta, 4), "kind": c.kind,
                    "premise": prem.canonical_text,
                    "section": (prem.provenance[0].section_number if prem.provenance else None)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=str(ART / "query_examples" / "comparison_structural.md"))
    args = ap.parse_args()

    nodes = read_nodes(ART / "nodes.jsonl")
    cards = read_cards(ART / "cards.jsonl")

    target, tcos, ttext = resolve_question.resolve(HYP_Q, top_n=1)[0]
    print(f"entry node: {target} (cos {tcos:.3f}) — {ttext[:70]}")

    # --- WEAKEST-LINK ---
    g1 = graph_weakest_link(nodes, cards, target)
    g2 = graph_weakest_link(nodes, cards, target)          # determinism check
    graph_deterministic = (g1 == g2)

    client = build_client()
    doc = DOC.read_text()
    sysmsg = "You are given a document. Answer concisely, name the specific evidence."
    b1 = ask(client, args.model, sysmsg, f"DOCUMENT:\n{doc}\n\n{WL_Q}", 500)
    b2 = ask(client, args.model, sysmsg, f"DOCUMENT:\n{doc}\n\n{WL_Q}", 500)
    baseline_reproducible = (b1.strip() == b2.strip())

    md = [f"# Structural comparison (what the graph is FOR)\n",
          f"*Model: {args.model}. Entry node resolved from the question: `{target}`.*\n",
          f"## Test 1 — Weakest link (ablation vs from-scratch reasoning)\n",
          f"**Question:** {WL_Q}\n",
          f"### Graph (deterministic ablation)\n",
          "The graph removes each card, re-scores, ranks by |Δ|. Top load-bearing cards:\n"]
    for e in g1:
        md.append(f"- `{e['card_id']}` Δ={e['delta']} ({e['kind']}) §{e['section']}: {e['premise'][:90]}")
    md += [f"\n### Baseline (reasons from the full document each time)\n{b1}\n",
           f"## Test 2 — Audit / determinism (same question, twice)\n",
           f"- **Graph identical across two runs:** {graph_deterministic}  "
           f"(pure arithmetic — always reproducible, and every number traces to a card→premise→span)",
           f"- **Baseline identical across two runs:** {baseline_reproducible}  "
           f"(if false, the 'answer' isn't stable; if true, still un-auditable — you can't ask *why* it ranked one piece over another)",
           f"\n## Not testable here — multi-document reasoning\n",
           "The graph's biggest edge (reason across sources whose combined text exceeds one "
           "context window) needs 3-5 documents. We have one. That is the benchmark to build next."]
    Path(args.out).write_text("\n".join(md))

    print(f"\n=== WEAKEST-LINK — graph (deterministic={graph_deterministic}) ===")
    for e in g1:
        print(f"  {e['card_id']} Δ={e['delta']} §{e['section']}: {e['premise'][:70]}")
    print(f"\n=== WEAKEST-LINK — baseline (reproducible across 2 runs={baseline_reproducible}) ===")
    print(b1[:700])
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
