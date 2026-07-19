"""Step 8 — DF-QuAD scoring over the card graph (design.md lines 171-241).

Pure arithmetic, no LLM. Reads typed nodes (`nodes.jsonl`) + reified relation cards
(`cards.jsonl`) and computes a strength in [0,1] for every node, then writes
`scores.jsonl` (contract 5). Implements the Discontinuity-Free QuAD semantics
(Rago et al. 2016) exactly as pinned in RECON §5 (conflict C5):

    F(S)  = 1 - Π(1 - xᵢ)                      combine a set of strengths
    va    = F(strengths of the ATTACK cards on a node)
    vs    = F(strengths of the SUPPORT cards on a node)
    σ     = v0 - v0·(va - vs)      if va ≥ vs   (attack ≥ support: pushed down)
          = v0 + (1 - v0)·(vs - va) otherwise    (support wins: pushed up)
    card strength = weight × min(strength of its premise nodes)   ← joint = min

`v0` is the per-node base score (uniform `--base`, default 0.5; a node may override
via `payload["base"]`). Nodes are scored in dependency order; genuine cycles are
condensed into SCCs and iterated with damping 0.5 for ≤`--max-iter` passes.

Two public functions the query lane (C) imports:
    score_graph(nodes, cards, roles=None, base=.5, damping=.5, max_iter=100)
        -> {node_id: ScoreRecord}
    ablate(nodes, cards, target, **kw) -> {card_id: |Δ strength of target|}
        (Query 3 "weakest link": the card whose removal moves `target` most.)

Leakage guard (the doc contains its own verdict): if `roles` is given, no `active`
card may touch a `conclusory` node — those are the judge's answer key and must stay
out of scoring. `roles` maps node_id -> role via each node's provenance claim_id.

Usage:
    python scripts/score_dfquad.py --selftest          # fixture regression gate
    python scripts/score_dfquad.py                      # score the real graph
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import (  # noqa: E402
    Node, Card, read_nodes, read_cards, write_jsonl,
)

FX = ROOT / "artifacts" / "epistemic" / "fixtures"


@dataclass
class ScoreRecord:
    node_id: str
    strength: float
    base: float
    va: float                       # combined attacker pressure F(attack strengths)
    vs: float                       # combined supporter pressure F(support strengths)
    in_supporters: list[str] = field(default_factory=list)  # RA card_ids into this node
    in_attackers: list[str] = field(default_factory=list)   # CA card_ids into this node
    scc_id: int = -1                # SCC index (nodes sharing one are mutually cyclic)
    iterations: int = 1             # passes to converge (1 for acyclic nodes)


class LeakageError(AssertionError):
    """A conclusory (answer-key) node reached the scoring graph."""


def _F(strengths: list[float]) -> float:
    """DF-QuAD base combination: 1 - Π(1 - xᵢ). F(∅)=0."""
    prod = 1.0
    for x in strengths:
        prod *= (1.0 - x)
    return 1.0 - prod


def _base_of(node: Node, base: float) -> float:
    b = node.payload.get("base")
    return float(b) if isinstance(b, (int, float)) else base


def score_graph(nodes: list[Node], cards: list[Card], roles: dict[str, str] | None = None,
                base: float = 0.5, damping: float = 0.5, max_iter: int = 100
                ) -> dict[str, ScoreRecord]:
    active = [c for c in cards if getattr(c, "active", True)]
    node_ids = [n.node_id for n in nodes]
    idset = set(node_ids)

    # --- leakage guard (defense-in-depth over the funnel's source exclusion) ------
    if roles:
        for c in active:
            for nid in [*c.premises, c.target]:
                if roles.get(nid) == "conclusory":
                    raise LeakageError(
                        f"card {c.card_id} touches conclusory node {nid}; "
                        f"conclusory claims are the answer key and must not be scored")

    supporters: dict[str, list[Card]] = {nid: [] for nid in node_ids}
    attackers: dict[str, list[Card]] = {nid: [] for nid in node_ids}
    for c in active:
        if c.target not in idset:
            continue
        if c.kind == "RA":
            supporters[c.target].append(c)
        elif c.kind == "CA":
            attackers[c.target].append(c)
        elif c.kind == "PA":
            # PA "outweighs" = the disfavoured side is dampened. MVP models it as an
            # attack on the target (the outweighed node). Weight-modulation of other
            # cards is deferred (RECON C5 simplification).
            attackers[c.target].append(c)

    # dependency graph: premise -> target, so a target is scored after its premises.
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for c in active:
        for p in c.premises:
            if p in idset and c.target in idset:
                G.add_edge(p, c.target)

    cond = nx.condensation(G)                     # DAG of strongly-connected components
    mapping: dict[str, int] = cond.graph["mapping"]
    scc_members: dict[int, list[str]] = {}
    for nid in node_ids:
        scc_members.setdefault(mapping[nid], []).append(nid)

    base_of = {n.node_id: _base_of(n, base) for n in nodes}
    strength = dict(base_of)                       # init at base
    va = {nid: 0.0 for nid in node_ids}
    vs = {nid: 0.0 for nid in node_ids}
    iters = {nid: 1 for nid in node_ids}

    def card_strength(c: Card) -> float:
        return c.weight * min(strength[p] for p in c.premises)

    def sigma(nid: str) -> tuple[float, float, float]:
        vsn = _F([card_strength(c) for c in supporters[nid]])
        van = _F([card_strength(c) for c in attackers[nid]])
        v0 = base_of[nid]
        s = v0 - v0 * (van - vsn) if van >= vsn else v0 + (1.0 - v0) * (vsn - van)
        return s, van, vsn

    for scc in nx.topological_sort(cond):
        members = scc_members[scc]
        cyclic = len(members) > 1 or G.has_edge(members[0], members[0])
        if not cyclic:
            nid = members[0]
            strength[nid], va[nid], vs[nid] = sigma(nid)
            iters[nid] = 1
        else:
            for it in range(1, max_iter + 1):
                maxd = 0.0
                for nid in members:
                    s, van, vsn = sigma(nid)
                    newv = damping * strength[nid] + (1.0 - damping) * s
                    maxd = max(maxd, abs(newv - strength[nid]))
                    strength[nid], va[nid], vs[nid], iters[nid] = newv, van, vsn, it
                if maxd < 1e-9:
                    break

    return {
        nid: ScoreRecord(
            node_id=nid, strength=round(strength[nid], 6), base=base_of[nid],
            va=round(va[nid], 6), vs=round(vs[nid], 6),
            in_supporters=[c.card_id for c in supporters[nid]],
            in_attackers=[c.card_id for c in attackers[nid]],
            scc_id=mapping[nid], iterations=iters[nid],
        )
        for nid in node_ids
    }


def ablate(nodes: list[Node], cards: list[Card], target: str, **kw) -> dict[str, float]:
    """Query 3: remove each active card, re-score, return |Δ strength of `target`|.
    The card with the largest Δ is the load-bearing link."""
    active = [c for c in cards if getattr(c, "active", True)]
    baseline = score_graph(nodes, cards, **kw)[target].strength
    out: dict[str, float] = {}
    for i, c in enumerate(active):
        kept = [cc for j, cc in enumerate(active) if j != i]
        s = score_graph(nodes, kept, **kw)[target].strength
        out[c.card_id] = round(abs(s - baseline), 6)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def roles_from_tagged(nodes: list[Node], tagged_path: str | Path) -> dict[str, str]:
    """Map node_id -> role by joining nodes to claims_tagged.jsonl on claim_id."""
    role_by_claim: dict[str, str] = {}
    with open(tagged_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                role_by_claim[r["claim_id"]] = r.get("role")
    out: dict[str, str] = {}
    for n in nodes:
        cid = n.provenance[0].claim_id if n.provenance else None
        if cid in role_by_claim:
            out[n.node_id] = role_by_claim[cid]
    return out


def _selftest() -> int:
    """Regression gate on the drug-trial fixture. Asserts the exact DF-QuAD outputs
    AND design.md's ordinal claim: removing Card 2 (funding attack, up the chain)
    moves C5 more than removing Card 4 (the direct attack). See fixtures/README.md."""
    nodes = read_nodes(FX / "sample_nodes.jsonl")
    cards = read_cards(FX / "sample_cards.jsonl")
    sc = score_graph(nodes, cards, base=0.5)

    def approx(a, b, tol=1e-6):
        return abs(a - b) <= tol

    ok = True
    # exact values (independently hand-derived from the formula; see README)
    expect = {"C1": 0.5, "C2": 0.5, "C3": 0.5, "C4": 0.5, "C5": 0.675, "C6": 0.5}
    for nid, want in expect.items():
        got = sc[nid].strength
        flag = "ok" if approx(got, want) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  strength[{nid}] = {got:.4f}  (want {want})  {flag}")

    abl = ablate(nodes, cards, "C5", base=0.5)
    print("  ablate(C5):", {k: round(v, 4) for k, v in abl.items()})
    # design.md's load-bearing story: Card 2 outranks Card 4 for C5.
    story = abl["card-2"] > abl["card-4"]
    print(f"  design ordinal (Δcard-2 > Δcard-4): {abl['card-2']:.4f} > "
          f"{abl['card-4']:.4f}  {'ok' if story else 'FAIL'}")
    ok = ok and story
    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="run the fixture regression gate")
    ap.add_argument("--nodes", default=str(ROOT / "artifacts" / "epistemic" / "nodes.jsonl"))
    ap.add_argument("--cards", default=str(ROOT / "artifacts" / "epistemic" / "cards.jsonl"))
    ap.add_argument("--tagged", default=str(ROOT / "artifacts" / "epistemic" / "claims_tagged.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "epistemic" / "scores.jsonl"))
    ap.add_argument("--base", type=float, default=0.5)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--no-leakage-guard", action="store_true",
                    help="skip the conclusory-node assert (not recommended)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    nodes = read_nodes(args.nodes)
    cards = read_cards(args.cards)
    roles = None if args.no_leakage_guard else roles_from_tagged(nodes, args.tagged)
    scores = score_graph(nodes, cards, roles=roles, base=args.base,
                         damping=args.damping, max_iter=args.max_iter)
    n = write_jsonl(args.out, list(scores.values()))
    contested = [s for s in scores.values() if s.in_supporters and s.in_attackers]
    print(f"scored {n} nodes -> {args.out}")
    print(f"  cards: {sum(1 for c in cards if getattr(c,'active',True))} active")
    print(f"  contested nodes (both support & attack): {len(contested)}")


if __name__ == "__main__":
    main()
