"""Query the epistemic argument graph: DF-QuAD scoring + three query types.

Defaults to artifacts/epistemic/fixtures/sample_{nodes,cards,scores}.jsonl --
the same fixture data --selftest verifies against -- so this runs correctly
out of the box. Point --data-dir/--prefix at the real files once they exist.

DF-QuAD scoring. The formula below was reverse-engineered by hand against the
fixtures (every number in sample_scores.jsonl / sample_ablation.json is
reproduced exactly by --selftest), not assumed from the DF-QuAD literature.
A card with multiple premises is a JOINT relation -- all premises required,
so its strength is bounded by the weakest one:

    card_strength(card)   = weight(card) * min(strength[p] for p in premises)
    aggregate(strengths)  = 1 - product(1 - s for s in strengths)   # independent cards combine via complement-product
    vs(node)              = aggregate(strengths of active RA/support cards targeting node)
    va(node)              = aggregate(strengths of active CA/attack cards targeting node)
    strength(node)        = base + (1-base)*(vs-va)   if vs >= va
                             base - base*(va-vs)       otherwise

Cycles (a node whose score depends on itself through some chain) are handled
by grouping nodes into strongly connected components (Tarjan) and iterating
each cyclic component to a fixed point, rather than assuming the graph is a
DAG -- the fixture is acyclic (iterations=1 everywhere), so that path isn't
fixture-verified, but the real 590-node graph may not be.

Usage:
    python scripts/query_epistemic_single_multidoc.py --selftest
    python scripts/query_epistemic_single_multidoc.py evidence-for C5
    python scripts/query_epistemic_single_multidoc.py contested
    python scripts/query_epistemic_single_multidoc.py weakest-link C5
    python scripts/query_epistemic_single_multidoc.py rank-hypotheses
    python scripts/query_epistemic_single_multidoc.py quantitative-backbone
    python scripts/query_epistemic_single_multidoc.py evidence-for C5 --no-llm   # structured JSON only, no API call
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "artifacts" / "epistemic" / "fixtures"

load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")


def build_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing. Copy .env.example to .env and fill it in.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, max_retries=3, timeout=300)


def preflight(client: OpenAI, model: str) -> None:
    """Together rejects some params OpenAI accepts; fail here, not mid-query."""
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
    )
    msg = (resp.choices[0].message.content or "").strip()
    print(f"  preflight LLM ok -> {msg[:60]!r}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# DF-QuAD scoring
# ---------------------------------------------------------------------------

def card_strength(card: dict, strengths: dict[str, float]) -> float:
    premise_strengths = [strengths[p] for p in card["premises"]]
    return card["weight"] * min(premise_strengths)


def aggregate(strengths: list[float]) -> float:
    product = 1.0
    for s in strengths:
        product *= (1 - s)
    return 1 - product


def df_quad_combine(base: float, vs: float, va: float) -> float:
    if vs >= va:
        return base + (1 - base) * (vs - va)
    return base - base * (va - vs)


def compute_sccs(node_ids: list[str], dependency_edges: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's algorithm. dependency_edges[n] = premises n's score depends on.

    Because we recurse into dependencies before closing off a node's own SCC,
    the returned list is already in evaluation order: a node's SCC never
    appears before the SCCs of everything it depends on.
    """
    index_counter = [0]
    stack: list[str] = []
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    def strongconnect(node: str) -> None:
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        for successor in dependency_edges.get(node, []):
            if successor not in index:
                strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif on_stack.get(successor):
                lowlink[node] = min(lowlink[node], index[successor])

        if lowlink[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            result.append(scc)

    for node in node_ids:
        if node not in index:
            strongconnect(node)

    return result


def _compute_one(node: str, strengths: dict[str, float],
                  incoming_support: dict[str, list[dict]], incoming_attack: dict[str, list[dict]],
                  base: float) -> float:
    vs = aggregate([card_strength(c, strengths) for c in incoming_support.get(node, [])])
    va = aggregate([card_strength(c, strengths) for c in incoming_attack.get(node, [])])
    return df_quad_combine(base, vs, va)


def score_graph(nodes: list[dict], cards: list[dict], base: float = 0.5,
                max_iterations: int = 100, tol: float = 1e-9) -> dict[str, dict]:
    node_ids = [n["node_id"] for n in nodes]
    active_cards = [c for c in cards if c.get("active", True)]

    incoming_support: dict[str, list[dict]] = defaultdict(list)
    incoming_attack: dict[str, list[dict]] = defaultdict(list)
    dependency_edges: dict[str, list[str]] = defaultdict(list)
    for card in active_cards:
        bucket = incoming_support if card["kind"] == "RA" else incoming_attack
        bucket[card["target"]].append(card)
        dependency_edges[card["target"]].extend(card["premises"])

    sccs = compute_sccs(node_ids, dependency_edges)
    # scc_id: assign in reverse of evaluation order, so the final/root node(s)
    # get the lowest ids -- matches the fixture's own numbering (C5, the
    # ultimate target, is scc_id 0; leaves get the highest ids).
    scc_id_by_node: dict[str, int] = {}
    for scc_index, scc in enumerate(reversed(sccs)):
        for node in scc:
            scc_id_by_node[node] = scc_index

    strengths: dict[str, float] = {}
    result: dict[str, dict] = {}

    for scc in sccs:
        is_cyclic = len(scc) > 1 or scc[0] in dependency_edges.get(scc[0], [])

        if not is_cyclic:
            node = scc[0]
            strengths[node] = _compute_one(node, strengths, incoming_support, incoming_attack, base)
            iterations = 1
        else:
            for node in scc:
                strengths.setdefault(node, base)
            iterations = 0
            for it in range(1, max_iterations + 1):
                iterations = it
                new_values = {n: _compute_one(n, strengths, incoming_support, incoming_attack, base) for n in scc}
                max_delta = max(abs(new_values[n] - strengths[n]) for n in scc)
                strengths.update(new_values)
                if max_delta < tol:
                    break

        for node in scc:
            supporters = [c["card_id"] for c in incoming_support.get(node, [])]
            attackers = [c["card_id"] for c in incoming_attack.get(node, [])]
            vs = aggregate([card_strength(c, strengths) for c in incoming_support.get(node, [])])
            va = aggregate([card_strength(c, strengths) for c in incoming_attack.get(node, [])])
            result[node] = {
                "node_id": node,
                "strength": strengths[node],
                "base": base,
                "va": va,
                "vs": vs,
                "in_supporters": supporters,
                "in_attackers": attackers,
                "scc_id": scc_id_by_node[node],
                "iterations": iterations,
            }

    return result


def ablate(nodes: list[dict], cards: list[dict], target_id: str) -> dict[str, float]:
    """Remove each active card one at a time; report how much target_id's score moved."""
    baseline = score_graph(nodes, cards)[target_id]["strength"]
    deltas: dict[str, float] = {}
    for card in cards:
        if not card.get("active", True):
            continue
        remaining = [c for c in cards if c["card_id"] != card["card_id"]]
        new_strength = score_graph(nodes, remaining)[target_id]["strength"]
        deltas[card["card_id"]] = abs(baseline - new_strength)
    return dict(sorted(deltas.items(), key=lambda kv: -kv[1]))


def selftest() -> None:
    nodes = load_jsonl(FIXTURES_DIR / "sample_nodes.jsonl")
    cards = load_jsonl(FIXTURES_DIR / "sample_cards.jsonl")
    expected_scores = {r["node_id"]: r for r in load_jsonl(FIXTURES_DIR / "sample_scores.jsonl")}
    expected_ablation = json.loads((FIXTURES_DIR / "sample_ablation.json").read_text())

    ok = True
    result = score_graph(nodes, cards)

    for node_id, expected in expected_scores.items():
        actual = result[node_id]
        for field in ("strength", "base", "va", "vs"):
            if abs(actual[field] - expected[field]) > 1e-9:
                print(f"MISMATCH {node_id}.{field}: expected {expected[field]}, got {actual[field]}")
                ok = False
        if actual["in_supporters"] != expected["in_supporters"]:
            print(f"MISMATCH {node_id}.in_supporters: expected {expected['in_supporters']}, got {actual['in_supporters']}")
            ok = False
        if actual["in_attackers"] != expected["in_attackers"]:
            print(f"MISMATCH {node_id}.in_attackers: expected {expected['in_attackers']}, got {actual['in_attackers']}")
            ok = False
    print(f"scores: checked {len(expected_scores)} nodes")

    for target_id, expected_deltas in expected_ablation.items():
        actual_deltas = ablate(nodes, cards, target_id)
        for card_id, expected_delta in expected_deltas.items():
            actual_delta = actual_deltas.get(card_id)
            if actual_delta is None or abs(actual_delta - expected_delta) > 1e-9:
                print(f"MISMATCH ablate({target_id})[{card_id}]: expected {expected_delta}, got {actual_delta}")
                ok = False
    print(f"ablation: checked {sum(len(v) for v in expected_ablation.values())} deltas across {len(expected_ablation)} targets")

    if not ok:
        sys.exit("\nSELFTEST FAILED")
    print("\nSELFTEST PASSED")


# ---------------------------------------------------------------------------
# Document resolution -- three-tier fallback, most specific first:
#   1. provenance[].document -- explicit, set by extraction once ingestion is
#      document-aware (epistemic_store.Provenance.document). Not backfilled
#      onto the existing single-document corpus, so this is empty today.
#   2. provenance[].chunk_id present -- real grounding, but from before
#      per-document stamping existed. Only one real document (eric_decision.pdf)
#      exists in the corpus today, so that's a safe default -- NOT a general
#      multi-document assumption. Once a second real document is ingested with
#      chunk_id but no explicit `document`, this tier stops being safe and the
#      extraction step must start populating tier 1 instead.
#   3. meta.source / meta.fixture -- the hand-authored fixtures' own
#      convention (no chunk_id at all).
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENT = "eric_decision.pdf"


def resolve_document(node: dict) -> str:
    for p in node.get("provenance") or []:
        if p.get("document"):
            return p["document"]
    for p in node.get("provenance") or []:
        if p.get("chunk_id"):
            return DEFAULT_DOCUMENT
    meta = node.get("meta") or {}
    return meta.get("source") or meta.get("fixture") or "unknown"


def card_documents(card: dict, nodes_by_id: dict[str, dict]) -> list[str]:
    return sorted({resolve_document(nodes_by_id[p]) for p in card["premises"] if p in nodes_by_id})


# ---------------------------------------------------------------------------
# Query 1: "What's the evidence for X?"
# ---------------------------------------------------------------------------
#evidence_for() (291-346): recursively follows each node's in_supporters back through the premise chain, sorts by strength at each level, and switches to grouping-by-document if the chain spans more than one source.
def evidence_for(node_id: str, nodes_by_id: dict[str, dict], scores_by_id: dict[str, dict],
                  cards_by_id: dict[str, dict], max_depth: int = 6) -> dict:
    def walk(nid: str, depth: int) -> list[dict]:
        supporters = scores_by_id.get(nid, {}).get("in_supporters", [])
        entries = []
        for card_id in supporters:
            card = cards_by_id[card_id]
            for premise_id in card["premises"]:
                premise = nodes_by_id[premise_id]
                premise_score = scores_by_id.get(premise_id, {})
                entry = {
                    "card_id": card_id,
                    "node_id": premise_id,
                    "text": premise["canonical_text"],
                    "strength": premise_score.get("strength"),
                    "document": resolve_document(premise),
                    "provenance": premise.get("provenance"),
                    "children": walk(premise_id, depth + 1) if depth < max_depth else [],
                }
                entries.append(entry)
        entries.sort(key=lambda e: -(e["strength"] or 0))
        return entries

    root = nodes_by_id[node_id]
    tree = {
        "node_id": node_id,
        "text": root["canonical_text"],
        "strength": scores_by_id.get(node_id, {}).get("strength"),
        "children": walk(node_id, 0),
    }

    docs: set[str] = set()

    def collect(entry: dict) -> None:
        docs.add(entry["document"])
        for c in entry["children"]:
            collect(c)

    for c in tree["children"]:
        collect(c)

    if len(docs) <= 1:
        return {"mode": "chain", **tree}

    grouped: dict[str, list[dict]] = defaultdict(list)

    def flatten(entry: dict) -> None:
        grouped[entry["document"]].append({k: v for k, v in entry.items() if k != "children"})
        for c in entry["children"]:
            flatten(c)

    for c in tree["children"]:
        flatten(c)

    return {"mode": "grouped_by_document", "node_id": node_id, "text": tree["text"],
            "strength": tree["strength"], "by_document": dict(grouped)}


# ---------------------------------------------------------------------------
# Query 2: "Where do sources disagree?"
# ---------------------------------------------------------------------------
#contested() (353-378): scans every node for ones with both supporters and attackers, reports which side is winning.
def contested(nodes_by_id: dict[str, dict], scores_by_id: dict[str, dict],
              cards_by_id: dict[str, dict]) -> list[dict]:
    results = []
    for node_id, score in scores_by_id.items():
        supporters = score.get("in_supporters", [])
        attackers = score.get("in_attackers", [])
        if not (supporters and attackers):
            continue
        node = nodes_by_id[node_id]
        vs, va = score["vs"], score["va"]
        winner = "support" if vs > va else ("attack" if va > vs else "tie")
        support_docs = sorted({d for c in supporters for d in card_documents(cards_by_id[c], nodes_by_id)})
        attack_docs = sorted({d for c in attackers for d in card_documents(cards_by_id[c], nodes_by_id)})
        results.append({
            "node_id": node_id,
            "text": node["canonical_text"],
            "vs": vs,
            "va": va,
            "winner": winner,
            "support_documents": support_docs,
            "attack_documents": attack_docs,
            "supporters": supporters,
            "attackers": attackers,
        })
    results.sort(key=lambda r: -abs(r["vs"] - r["va"]))
    return results


# ---------------------------------------------------------------------------
# Query 3: "What's the weakest link?"
# ---------------------------------------------------------------------------
#weakest_link() (385-401): thin wrapper around ablate() that resolves the top few card IDs back to actual node text for display.
def weakest_link(node_id: str, nodes: list[dict], cards: list[dict],
                  nodes_by_id: dict[str, dict], cards_by_id: dict[str, dict],
                  top_n: int = 5) -> list[dict]:
    deltas = ablate(nodes, cards, node_id)
    results = []
    for card_id, delta in list(deltas.items())[:top_n]:
        card = cards_by_id[card_id]
        premises = [nodes_by_id[p] for p in card["premises"]]
        results.append({
            "card_id": card_id,
            "delta": delta,
            "kind": card["kind"],
            "premises": [{"node_id": p["node_id"], "text": p["canonical_text"],
                          "document": resolve_document(p)} for p in premises],
            "target": card["target"],
        })
    return results


# ---------------------------------------------------------------------------
# Query 3b: weakest-link, scoped to one hypothesis/claim's dependency chain
# ---------------------------------------------------------------------------
# ablate() (and therefore weakest_link()) tests EVERY card in the whole graph
# for removal, each time doing a full score_graph() rescore, purely to see
# how one target node's strength moved -- O(total cards in the corpus), most
# of which are provably irrelevant. A card can only move target_id's strength
# if it lies in target_id's backward dependency closure (support+attack,
# recursively) -- DF-QuAD strength only ever flows backward from a card's
# premises to its target, never forward, so a card outside that closure has
# delta=0 by construction. _dependency_card_ids() collects exactly that
# closure; ablate_scoped()/weakest_link_scoped() restrict the existing,
# unmodified ablate() to it -- same numbers, O(closure size) score_graph()
# calls instead of O(whole corpus).

def _dependency_card_ids(node_id: str, scores_by_id: dict[str, dict],
                         cards_by_id: dict[str, dict]) -> set[str]:
    card_ids: set[str] = set()
    seen_nodes: set[str] = set()

    def walk(nid: str) -> None:
        if nid in seen_nodes:
            return
        seen_nodes.add(nid)
        score = scores_by_id.get(nid, {})
        for card_id in score.get("in_supporters", []) + score.get("in_attackers", []):
            card_ids.add(card_id)
            for premise_id in cards_by_id[card_id]["premises"]:
                walk(premise_id)

    walk(node_id)
    return card_ids


def ablate_scoped(nodes: list[dict], cards: list[dict], target_id: str,
                  scores_by_id: dict[str, dict], cards_by_id: dict[str, dict]) -> dict[str, float]:
    relevant = _dependency_card_ids(target_id, scores_by_id, cards_by_id)
    scoped_cards = [c for c in cards if c["card_id"] in relevant]
    return ablate(nodes, scoped_cards, target_id)


def weakest_link_scoped(node_id: str, nodes: list[dict], cards: list[dict],
                        nodes_by_id: dict[str, dict], cards_by_id: dict[str, dict],
                        scores_by_id: dict[str, dict], top_n: int = 5) -> list[dict]:
    deltas = ablate_scoped(nodes, cards, node_id, scores_by_id, cards_by_id)
    results = []
    for card_id, delta in list(deltas.items())[:top_n]:
        card = cards_by_id[card_id]
        premises = [nodes_by_id[p] for p in card["premises"]]
        results.append({
            "card_id": card_id,
            "delta": delta,
            "kind": card["kind"],
            "premises": [{"node_id": p["node_id"], "text": p["canonical_text"],
                          "document": resolve_document(p)} for p in premises],
            "target": card["target"],
        })
    return results


# ---------------------------------------------------------------------------
# Query 4: "Which hypothesis does the evidence currently favor?"
# ---------------------------------------------------------------------------
# rank_hypotheses(): every hypothesis-type node, sorted by final DF-QuAD
# strength. Unlike contested() (any node with both sides) or weakest_link()
# (impact of removing one card), this answers "of the competing top-level
# explanations, which one does the graph currently favor overall" -- a plain
# sort over already-computed scores, no new traversal.

def rank_hypotheses(nodes_by_id: dict[str, dict], scores_by_id: dict[str, dict]) -> list[dict]:
    results = []
    for node_id, node in nodes_by_id.items():
        if node.get("type") != "hypothesis":
            continue
        score = scores_by_id.get(node_id, {})
        results.append({
            "node_id": node_id,
            "text": node["canonical_text"],
            "strength": score.get("strength"),
            "vs": score.get("vs"),
            "va": score.get("va"),
            "n_supporters": len(score.get("in_supporters", [])),
            "n_attackers": len(score.get("in_attackers", [])),
            "document": resolve_document(node),
        })
    results.sort(key=lambda r: -(r["strength"] or 0))
    return results


# ---------------------------------------------------------------------------
# Query 5: "What's the quantitative backbone?"
# ---------------------------------------------------------------------------
# quantitative_backbone(): the estimate-type nodes (type_claims.py's ~8%
# quantitative core -- the claims whose CONTENT is a number: a prior,
# probability, or Bayes factor) plus any node that carries a `number` in its
# payload even if typed otherwise. For each, also resolves its direct outgoing
# cards (this node as a PREMISE) so the numeric skeleton's connections to the
# qualitative argument around it are visible, not just the bare numbers.

def _is_quantitative(node: dict) -> bool:
    return node.get("type") == "estimate" or bool((node.get("payload") or {}).get("number"))


def quantitative_backbone(nodes_by_id: dict[str, dict], scores_by_id: dict[str, dict],
                          cards_by_id: dict[str, dict]) -> list[dict]:
    outgoing: dict[str, list[dict]] = defaultdict(list)
    for card in cards_by_id.values():
        for premise_id in card["premises"]:
            outgoing[premise_id].append(card)

    results = []
    for node_id, node in nodes_by_id.items():
        if not _is_quantitative(node):
            continue
        payload = node.get("payload") or {}
        number = payload.get("number")
        score = scores_by_id.get(node_id, {})
        feeds_into = []
        for card in outgoing.get(node_id, []):
            target = nodes_by_id.get(card["target"], {})
            feeds_into.append({
                "card_id": card["card_id"],
                "relation": "supports" if card["kind"] == "RA" else "attacks",
                "target_node_id": card["target"],
                "target_text": target.get("canonical_text"),
            })
        results.append({
            "node_id": node_id,
            "text": node["canonical_text"],
            "number": number,
            "attribution": payload.get("attribution"),
            "strength": score.get("strength"),
            "document": resolve_document(node),
            "feeds_into": feeds_into,
        })
    results.sort(key=lambda r: -(r["strength"] or 0))
    return results


# ---------------------------------------------------------------------------
# Query 6+: "Collect the <X> that supports and negates a hypothesis"
# ---------------------------------------------------------------------------
# _collect_polarized(): the shared walker behind quantitative_evidence_for()
# and the assumption/analysis/limitation queries below. Walks BOTH the
# support chain and the attack chain of one target, filtering to nodes where
# `matches(node)` is true, and tracks polarity as it goes: something that
# SUPPORTS an attacker of the hypothesis still NEGATES the hypothesis
# overall, so polarity flips every time the walk crosses a CA (attack) card,
# and stays the same across an RA (support) card.

#For each of the node's in_supporters (RA cards),'
#it recurses into that card's premises with the same polarity (line 533) — going through a support link doesn't change whether you're on the "for" or "against" side.
#For each of the node's in_attackers (CA cards), it recurses into that card's premises with flipped polarity (line 536) — this is the key trick: something that attacks 
#an attacker of the hypothesis is actually helping it, and something that supports an attacker is still hurting it.

def _collect_polarized(node_id: str, matches, nodes_by_id: dict[str, dict],
                       scores_by_id: dict[str, dict], cards_by_id: dict[str, dict],
                       max_depth: int = 6) -> tuple[list[dict], list[dict]]:
    supporting: list[dict] = []
    negating: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def walk(nid: str, polarity: int, depth: int) -> None:
        if depth > max_depth or (nid, polarity) in seen:
            return
        seen.add((nid, polarity))

        node = nodes_by_id.get(nid)
        if node and nid != node_id and matches(node):
            entry = {
                "node_id": nid,
                "text": node["canonical_text"],
                "type": node.get("type"),
                "strength": scores_by_id.get(nid, {}).get("strength"),
                "document": resolve_document(node),
            }
            (supporting if polarity > 0 else negating).append(entry)

        score = scores_by_id.get(nid, {})
        for card_id in score.get("in_supporters", []):
            for premise_id in cards_by_id[card_id]["premises"]:
                walk(premise_id, polarity, depth + 1)          # support: polarity unchanged
        for card_id in score.get("in_attackers", []):
            for premise_id in cards_by_id[card_id]["premises"]:
                walk(premise_id, -polarity, depth + 1)         # attack: polarity flips

    walk(node_id, +1, 0)
    supporting.sort(key=lambda e: -(e["strength"] or 0))
    negating.sort(key=lambda e: -(e["strength"] or 0))
    return supporting, negating


def quantitative_evidence_for(node_id: str, nodes_by_id: dict[str, dict],
                              scores_by_id: dict[str, dict], cards_by_id: dict[str, dict],
                              max_depth: int = 6) -> dict:
    supporting, negating = _collect_polarized(
        node_id, _is_quantitative, nodes_by_id, scores_by_id, cards_by_id, max_depth)

    def enrich(entries: list[dict]) -> list[dict]:
        out = []
        for e in entries:
            payload = nodes_by_id[e["node_id"]].get("payload") or {}
            out.append({**e, "number": payload.get("number"), "attribution": payload.get("attribution")})
        return out

    root = nodes_by_id[node_id]
    return {
        "node_id": node_id,
        "text": root["canonical_text"],
        "strength": scores_by_id.get(node_id, {}).get("strength"),
        "supporting_quantitative": enrich(supporting),
        "negating_quantitative": enrich(negating),
    }


# ---------------------------------------------------------------------------
# Query 7: "What assumptions does this hypothesis rely on?"
# ---------------------------------------------------------------------------
# `assumption` is a real NODE_TYPE (75 nodes in the current corpus) -- an
# explicit premise taken as given, that nothing else argues FOR but that
# other claims quietly depend on. Same polarity-aware walk as the
# quantitative query: an assumption underlying an ATTACKER of the hypothesis
# still counts as negating it.
#
# Queries 8-11 follow the identical shape for the rest of the real NODE_TYPES
# (claim, background, rebuttal, verdict) -- all built on the one generic
# `_type_query_for` below rather than repeating this function five more times.
#
# SCHEMA NOTE on "analysis"/"limitation": neither is in epistemic_store.
# NODE_TYPES for the CURRENT corpus. Both existed in an earlier iteration of
# this pipeline (the extraction-variants scripts, before type_claims.py
# consolidated everything into today's 8-role taxonomy: hypothesis/evidence/
# assumption/estimate/rebuttal/verdict/background/claim) and got absorbed
# into other roles. Calling `_type_query_for(..., "analysis", ...)` is
# correct and ready to use, but will always return empty against the real
# nodes.jsonl today -- not a bug, just a currently-empty node type.

_TYPE_QUERY_KEYS = {
    "assumption": "assumptions", "analysis": "analysis", "limitation": "limitations",
    "claim": "claims", "background": "background", "rebuttal": "rebuttals",
    "verdict": "verdicts",
}


def _type_query_for(node_id: str, node_type: str, nodes_by_id: dict[str, dict],
                    scores_by_id: dict[str, dict], cards_by_id: dict[str, dict],
                    max_depth: int = 6) -> dict:
    supporting, negating = _collect_polarized(
        node_id, lambda n: n.get("type") == node_type,
        nodes_by_id, scores_by_id, cards_by_id, max_depth)
    key = _TYPE_QUERY_KEYS.get(node_type, node_type)
    root = nodes_by_id[node_id]
    return {
        "node_id": node_id,
        "text": root["canonical_text"],
        "strength": scores_by_id.get(node_id, {}).get("strength"),
        f"supporting_{key}": supporting,
        f"negating_{key}": negating,
    }


def assumptions_for(node_id, nodes_by_id, scores_by_id, cards_by_id, max_depth=6):
    return _type_query_for(node_id, "assumption", nodes_by_id, scores_by_id, cards_by_id, max_depth)


def analysis_for(node_id, nodes_by_id, scores_by_id, cards_by_id, max_depth=6):
    return _type_query_for(node_id, "analysis", nodes_by_id, scores_by_id, cards_by_id, max_depth)


def limitations_for(node_id, nodes_by_id, scores_by_id, cards_by_id, max_depth=6):
    return _type_query_for(node_id, "limitation", nodes_by_id, scores_by_id, cards_by_id, max_depth)


def claims_for(node_id, nodes_by_id, scores_by_id, cards_by_id, max_depth=6):
    return _type_query_for(node_id, "claim", nodes_by_id, scores_by_id, cards_by_id, max_depth)


def background_for(node_id, nodes_by_id, scores_by_id, cards_by_id, max_depth=6):
    return _type_query_for(node_id, "background", nodes_by_id, scores_by_id, cards_by_id, max_depth)


def rebuttals_for(node_id, nodes_by_id, scores_by_id, cards_by_id, max_depth=6):
    return _type_query_for(node_id, "rebuttal", nodes_by_id, scores_by_id, cards_by_id, max_depth)


def verdicts_for(node_id, nodes_by_id, scores_by_id, cards_by_id, max_depth=6):
    return _type_query_for(node_id, "verdict", nodes_by_id, scores_by_id, cards_by_id, max_depth)


# ---------------------------------------------------------------------------
# LLM prose layer
# ---------------------------------------------------------------------------

def generate_prose(client, model: str, query_label: str, structured_result) -> str:
    prompt = (
        "You answer questions about an argumentation graph built from source documents. "
        "You are given a structured result (nodes, strengths, provenance) below. "
        "Write a short, readable prose answer, citing section numbers and quoting "
        "verbatim text where the structured result provides them. Do not state "
        "anything not present in the structured result.\n\n"
        f"Query: {query_label}\n\n"
        f"Structured result:\n{json.dumps(structured_result, indent=2)}"
    )
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--data-dir", default=str(FIXTURES_DIR))
    parent.add_argument("--prefix", default="sample_")
    parent.add_argument("--no-llm", action="store_true", help="print structured JSON only, skip the LLM prose call")
    parent.add_argument("--model", default=DEFAULT_MODEL)
    parent.add_argument("--top-n", type=int, default=5)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="verify DF-QuAD scoring against artifacts/epistemic/fixtures")
    sub = ap.add_subparsers(dest="command")

    ev = sub.add_parser("evidence-for", parents=[parent], help='Query 1: "What is the evidence for X?"')
    ev.add_argument("node_id")

    sub.add_parser("contested", parents=[parent], help='Query 2: "Where do sources disagree?"')

    wl = sub.add_parser("weakest-link", parents=[parent], help='Query 3: "What is the weakest link?"')
    wl.add_argument("node_id")

    wls = sub.add_parser("weakest-link-scoped", parents=[parent],
                         help='Query 3b: same result as weakest-link, restricted to the target\'s '
                              'own dependency chain (much faster on a large corpus)')
    wls.add_argument("node_id")

    sub.add_parser("rank-hypotheses", parents=[parent],
                   help='Query 4: "Which hypothesis does the evidence favor?"')

    sub.add_parser("quantitative-backbone", parents=[parent],
                   help='Query 5: "What is the quantitative backbone?"')

    qe = sub.add_parser("quantitative-evidence-for", parents=[parent],
                       help='Query 6: "Quantitative evidence that supports/negates a hypothesis"')
    qe.add_argument("node_id")

    af = sub.add_parser("assumptions-for", parents=[parent],
                        help='Query 7: "What assumptions does this rely on?"')
    af.add_argument("node_id")

    an = sub.add_parser("analysis-for", parents=[parent],
                        help='Query 8: "analysis" nodes bearing on this (empty on current corpus -- not a NODE_TYPE yet)')
    an.add_argument("node_id")

    lf = sub.add_parser("limitations-for", parents=[parent],
                        help='Query 9: "limitation" nodes bearing on this (empty on current corpus -- not a NODE_TYPE yet)')
    lf.add_argument("node_id")

    cf = sub.add_parser("claims-for", parents=[parent], help='Query 10: claim-type nodes bearing on this')
    cf.add_argument("node_id")

    bf = sub.add_parser("background-for", parents=[parent], help='Query 11: background-type nodes bearing on this')
    bf.add_argument("node_id")

    rf = sub.add_parser("rebuttals-for", parents=[parent], help='Query 12: rebuttal-type nodes bearing on this')
    rf.add_argument("node_id")

    vf = sub.add_parser("verdicts-for", parents=[parent], help='Query 13: verdict-type nodes bearing on this')
    vf.add_argument("node_id")

    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.command:
        ap.print_help()
        return

    data_dir = Path(args.data_dir)
    nodes = load_jsonl(data_dir / f"{args.prefix}nodes.jsonl")
    cards = load_jsonl(data_dir / f"{args.prefix}cards.jsonl")
    scores = load_jsonl(data_dir / f"{args.prefix}scores.jsonl")

    nodes_by_id = {n["node_id"]: n for n in nodes}
    cards_by_id = {c["card_id"]: c for c in cards}
    scores_by_id = {s["node_id"]: s for s in scores}

    if args.command == "evidence-for":
        result = evidence_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What is the evidence for "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "contested":
        result = contested(nodes_by_id, scores_by_id, cards_by_id)
        label = "Where do sources disagree?"
    elif args.command == "rank-hypotheses":
        result = rank_hypotheses(nodes_by_id, scores_by_id)
        label = "Which hypothesis does the evidence currently favor?"
    elif args.command == "quantitative-backbone":
        result = quantitative_backbone(nodes_by_id, scores_by_id, cards_by_id)
        label = "What is the quantitative backbone of this argument (the claims that carry an actual number)?"
    elif args.command == "quantitative-evidence-for":
        result = quantitative_evidence_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = (f'What quantitative evidence supports and negates '
                 f'"{nodes_by_id[args.node_id]["canonical_text"]}"?')
    elif args.command == "assumptions-for":
        result = assumptions_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What assumptions does "{nodes_by_id[args.node_id]["canonical_text"]}" rely on?'
    elif args.command == "analysis-for":
        result = analysis_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What analysis bears on "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "limitations-for":
        result = limitations_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What limitations bear on "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "claims-for":
        result = claims_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What claims bear on "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "background-for":
        result = background_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What background facts bear on "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "rebuttals-for":
        result = rebuttals_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What rebuttals bear on "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "verdicts-for":
        result = verdicts_for(args.node_id, nodes_by_id, scores_by_id, cards_by_id)
        label = f'What verdicts bear on "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "weakest-link":
        result = weakest_link(args.node_id, nodes, cards, nodes_by_id, cards_by_id, top_n=args.top_n)
        label = f'What is the weakest link in the case for "{nodes_by_id[args.node_id]["canonical_text"]}"?'
    elif args.command == "weakest-link-scoped":
        result = weakest_link_scoped(args.node_id, nodes, cards, nodes_by_id, cards_by_id,
                                     scores_by_id, top_n=args.top_n)
        label = f'What is the weakest link in the case for "{nodes_by_id[args.node_id]["canonical_text"]}"?'

    print(json.dumps(result, indent=2))

    if not args.no_llm:
        client = build_client()
        preflight(client, args.model)
        prose = generate_prose(client, args.model, label, result)
        print("\n=== Answer ===\n")
        print(prose)


if __name__ == "__main__":
    main()
