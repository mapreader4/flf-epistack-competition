"""Epistemic overlay store — the *container* for the Structure layer.

A thin, id-keyed typed graph that sits on top of the existing ingestion artifacts
(`artifacts/claims/claims.json`, `data/chunks.json`, `data/sections.json`, and the
HippoRAG KG). It is deliberately domain-agnostic: node/edge *types* name epistemic
roles (hypothesis, evidence, supports, attacks, weighs-on), never topic entities,
so the same schema describes the COVID, LHC, and egg case studies unchanged.

Design commitments (see CONTEXT.md §5 and the 2026-07-16 meeting notes):

  1. `node_id` is a STABLE synthetic id, never a hash of the text. The text can be
     re-phrased or canonicalized without changing identity. A separate
     `fingerprint` (normalized-text hash) is what we use to *detect* duplicates —
     identity and dedup are two different jobs (meeting-note item 5.iii).

  2. `provenance` is a LIST. One node can be grounded in many sources — this is how
     "same claim, different form" is represented natively (one proposition,
     multiple provenance links) instead of by merging at a similarity threshold.

  3. Edges are TYPED and DIRECTED, and carry an optional `payload` (a Bayes factor /
     sign / scheme). This is the headline upgrade over HippoRAG, which erases the
     predicate and collapses every relation into one unlabeled undirected edge.

  4. Every node and edge carries an extraction `tier` so inferred structure never
     masquerades as extracted fact:
        T1 = grounded on a byte-exact claim span (the trust floor)
        T2 = parsed from a chunk (e.g. an aggregation number not atomized into a claim)
        T3 = model-inferred (a proposed type, relation, or direction)

Selective quantification (decision 2026-07-16-e): `payload["number"]` / `log_bf`
stay null on the ~92% qualitative nodes and are populated only on the ~8% that
carry an actual number. Nothing here forces a probability onto a plain claim.

This module is the schema + I/O + validation only. It contains NO LLM calls — the
engines that *populate* the store (claim typing, pairwise linking, sub-question
grouping) live in their own scripts and import from here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Controlled vocabularies — epistemic roles, domain-agnostic by construction.
# Extend these deliberately; an engine emitting a type outside the set is a bug we
# want the validator to catch, not silently absorb.
# ---------------------------------------------------------------------------

NODE_TYPES = {
    "hypothesis",   # a top-level competing explanation being weighed (zoonosis, lab-leak)
    "evidence",     # an observation / finding / study result that bears on a hypothesis
    "assumption",   # an explicit premise taken as given (incl. independence assumptions)
    "estimate",     # a claim whose content IS a number: a prior, sub-prior, likelihood, or Bayes factor
    "rebuttal",     # a claim asserted to counter/attack another claim (e.g. a Rootclaim rebuttal)
    "verdict",      # a conclusion / decision / posterior statement
    "background",   # a plain contextual fact carrying no epistemic load in this argument
    "claim",        # fallback: an asserted proposition that fits none of the above cleanly
    "subquestion",  # a discourse node: a sub-question the debate answers (discourse tier)
    "correlation_group",  # a bundle of items that share a source/common cause (for discounting)
}

EDGE_TYPES = {
    "supports",            # src raises the plausibility of dst
    "attacks",             # src lowers the plausibility of dst (generic; refine to the three below)
    "rebuts",              # attacks the CONCLUSION of dst
    "undercuts",           # attacks the INFERENCE/applicability linking dst's premises to its claim
    "undermines",          # attacks a PREMISE of dst
    "weighs_on",           # signed likelihood-ratio edge: evidence -> hypothesis, carries log-BF (the crown jewel)
    "answers",             # claim -> subquestion it addresses (discourse tier)
    "decomposes_into",     # a prior/estimate -> its multiplicative sub-estimates (product chains)
    "variant_of",          # "similar but not identical" — a typed variant link (never a silent merge)
    "in_correlation_group",# item -> correlation_group it belongs to
    "part_of",             # structural containment (e.g. claim -> section), if we lift the doc tree
}

TIERS = {"T1", "T2", "T3"}

# Node types that legitimately overlap a number. `estimate` is number-primary;
# `evidence` may carry a Bayes factor too. Used only for light sanity reporting.
QUANTITATIVE_ROLES = {"estimate", "evidence"}

# The 3-layer vocabulary from design.md (step 2). The 8 fine-grained NODE_TYPES stay;
# `layer` is a coarser grouping the pairing funnel blocks on (data mostly targets
# arguments, arguments mostly target questions). Derived, never a source of truth.
LAYERS = ("data", "argument", "question")
LAYER_OF = {
    "evidence": "data", "estimate": "data", "background": "data",
    "correlation_group": "data",
    "claim": "argument", "rebuttal": "argument", "assumption": "argument",
    "hypothesis": "question", "verdict": "question", "subquestion": "question",
}


def layer_of(node_type: str) -> str:
    """Map a fine-grained NODE_TYPE to its 3-layer group (design.md step 2)."""
    return LAYER_OF.get(node_type, "argument")  # unknown → argument (safe middle)

# Card kinds — a Card is a *reified relation*: one-or-more premise nodes feeding a
# single target node under one label (AIF/IAT discipline). Kept separate from
# EDGE_TYPES because a card is n-ary (joint premises) and carries a weight, which a
# plain src->dst Edge cannot express. Steps 6-8 of design.md build over these.
#   RA = "reason for"  (support)  — premises raise the plausibility of the target
#   CA = "conflict"    (attack)   — premises lower the plausibility of the target
#   PA = "preference"  (outweighs)— one thing is weightier than another (design's PA)
CARD_KINDS = {"RA", "CA", "PA"}


# ---------------------------------------------------------------------------
# Fingerprint — for dedup, NOT identity.
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """Canonical form for duplicate DETECTION. NFKC-fold, lowercase, strip
    punctuation, collapse whitespace. Deliberately lossy — two phrasings of the
    same proposition should collide here, which is exactly what makes it useful for
    catching duplicates without merging them (they keep distinct node_ids and both
    provenance links)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def fingerprint(text: str) -> str:
    """A short stable hash of the normalized text. Equal fingerprint == strong
    duplicate candidate (still confirmed by an engine, never auto-merged)."""
    return "fp-" + hashlib.sha1(normalize_text(text).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    """One grounding of a node in the source corpus. Mirrors a claims.json record's
    locating fields so a node joins back onto the exact sentence it came from."""
    claim_id: str | None = None
    chunk_id: str | None = None
    section_number: str | None = None
    quote: str | None = None
    span: list[int] | None = None
    tier: str = "T1"
    # Which source document this grounding came from. Optional and unset for the
    # current single-document corpus (chunk_id alone is unambiguous there); once
    # ingestion spans multiple real documents, extraction should stamp this so a
    # node's provenance is self-describing instead of relying on there being only
    # one document to guess from.
    document: str | None = None

    def validate(self) -> list[str]:
        errs = []
        if self.tier not in TIERS:
            errs.append(f"provenance.tier {self.tier!r} not in {sorted(TIERS)}")
        return errs


@dataclass
class Node:
    node_id: str
    type: str
    canonical_text: str
    fingerprint: str
    provenance: list[Provenance] = field(default_factory=list)
    # type-specific, mostly empty. Common keys: number (verbatim token e.g. "1/50"),
    # log_bf (float), attribution (who asserts it), stance ("+"/"-"), scheme.
    payload: dict[str, Any] = field(default_factory=dict)
    tier: str = "T1"
    # provenance for the *typing decision* itself, kept separate from node grounding.
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errs = []
        if not self.node_id:
            errs.append("node missing node_id")
        if self.type not in NODE_TYPES:
            errs.append(f"node {self.node_id}: type {self.type!r} not in {sorted(NODE_TYPES)}")
        if self.tier not in TIERS:
            errs.append(f"node {self.node_id}: tier {self.tier!r} not in {sorted(TIERS)}")
        if not self.canonical_text:
            errs.append(f"node {self.node_id}: empty canonical_text")
        for p in self.provenance:
            errs += p.validate()
        return errs


@dataclass
class Edge:
    edge_id: str
    type: str
    src: str
    dst: str
    directed: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)  # {"stated_in": chunk_id} or {"inferred": true}
    tier: str = "T3"  # relations default to inferred until grounded

    def validate(self) -> list[str]:
        errs = []
        if self.type not in EDGE_TYPES:
            errs.append(f"edge {self.edge_id}: type {self.type!r} not in {sorted(EDGE_TYPES)}")
        if self.tier not in TIERS:
            errs.append(f"edge {self.edge_id}: tier {self.tier!r} not in {sorted(TIERS)}")
        if not self.src or not self.dst:
            errs.append(f"edge {self.edge_id}: missing src/dst")
        return errs


@dataclass
class Card:
    """A reified relation: one-or-more premise nodes → one target node, under a single
    label (kind) with a strength weight. This is design.md's "card". Unlike an Edge,
    a card is n-ary — the joint premises only matter together (design's C2+C3 → C4
    attack) — and it carries a `weight` the DF-QuAD scorer consumes. Premises/target
    are node_ids, kept inline (a card is not two edges). `active` is the soft-delete
    flag that `event_store.fold` toggles via card_deactivated.

    provenance is a dict, mirroring Edge's shape, e.g.
      {"labeler_model": ..., "relation_label": "support", "raw_confidence": 0.8,
       "llm_call_event": "ev-000123", "prompt_version": "label-v1"}."""
    card_id: str
    kind: str                    # one of CARD_KINDS: RA (support) | CA (attack) | PA (outweighs)
    weight: float                # relation strength in [0,1] (from the labeler's confidence)
    premises: list[str] = field(default_factory=list)   # node_ids feeding the card
    target: str = ""             # node_id the card outputs to
    active: bool = True          # soft-delete flag (card_deactivated flips it)
    provenance: dict[str, Any] = field(default_factory=dict)
    tier: str = "T3"             # relations are model-inferred by default

    def validate(self) -> list[str]:
        errs = []
        if not self.card_id:
            errs.append("card missing card_id")
        if self.kind not in CARD_KINDS:
            errs.append(f"card {self.card_id}: kind {self.kind!r} not in {sorted(CARD_KINDS)}")
        if not isinstance(self.weight, (int, float)) or not (0.0 <= self.weight <= 1.0):
            errs.append(f"card {self.card_id}: weight {self.weight!r} not in [0,1]")
        if not self.premises:
            errs.append(f"card {self.card_id}: no premises")
        if not self.target:
            errs.append(f"card {self.card_id}: missing target")
        if self.tier not in TIERS:
            errs.append(f"card {self.card_id}: tier {self.tier!r} not in {sorted(TIERS)}")
        return errs


# ---------------------------------------------------------------------------
# Id scheme — stable and synthetic. Given a running counter, ids are assigned in
# a deterministic order so a re-run reproduces the same ids as long as the input
# order is stable. Ids never depend on the node text (commitment #1).
# ---------------------------------------------------------------------------

def node_id(i: int) -> str:
    return f"n-{i:05d}"


def edge_id(i: int) -> str:
    return f"e-{i:05d}"


def card_id(i: int) -> str:
    return f"card-{i:05d}"


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _to_jsonable(rec) -> dict:
    d = asdict(rec)
    return d


def write_jsonl(path: str | Path, records: Iterable) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(_to_jsonable(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_nodes(path: str | Path) -> list[Node]:
    out = []
    for d in _read_jsonl(path):
        prov = [Provenance(**p) for p in d.pop("provenance", [])]
        out.append(Node(provenance=prov, **d))
    return out


def read_edges(path: str | Path) -> list[Edge]:
    return [Edge(**d) for d in _read_jsonl(path)]


def read_cards(path: str | Path) -> list[Card]:
    return [Card(**d) for d in _read_jsonl(path)]


def _read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Validation over a whole store — catches dangling edges, dup ids, bad types.
# ---------------------------------------------------------------------------

# write_cards is just write_jsonl (asdict handles the dataclass); named for symmetry
# with read_cards so producers say what they mean.
write_cards = write_jsonl


def validate_store(nodes: list[Node], edges: list[Edge],
                   cards: list["Card"] | None = None) -> list[str]:
    errs: list[str] = []
    ids: set[str] = set()
    for n in nodes:
        errs += n.validate()
        if n.node_id in ids:
            errs.append(f"duplicate node_id {n.node_id}")
        ids.add(n.node_id)
    for e in edges:
        errs += e.validate()
        if e.src not in ids:
            errs.append(f"edge {e.edge_id}: dangling src {e.src}")
        if e.dst not in ids:
            errs.append(f"edge {e.edge_id}: dangling dst {e.dst}")
    card_ids: set[str] = set()
    for c in cards or []:
        errs += c.validate()
        if c.card_id in card_ids:
            errs.append(f"duplicate card_id {c.card_id}")
        card_ids.add(c.card_id)
        for pid in c.premises:
            if pid not in ids:
                errs.append(f"card {c.card_id}: dangling premise {pid}")
        if c.target and c.target not in ids:
            errs.append(f"card {c.card_id}: dangling target {c.target}")
    return errs


if __name__ == "__main__":
    # Tiny self-test so the schema is runnable in isolation.
    n0 = Node(node_id=node_id(0), type="hypothesis",
              canonical_text="The virus emerged via zoonotic spillover at a market.",
              fingerprint=fingerprint("zoonotic spillover at a market"), tier="T1")
    n1 = Node(node_id=node_id(1), type="evidence",
              canonical_text="The furin cleavage site roughly doubles the odds of lab-leak.",
              fingerprint=fingerprint("furin site doubles odds of lab leak"),
              payload={"number": "2x", "log_bf": None}, tier="T1")
    e0 = Edge(edge_id=edge_id(0), type="weighs_on", src=n1.node_id, dst=n0.node_id,
              payload={"sign": "-", "log_bf": None}, tier="T3")
    c0 = Card(card_id=card_id(0), kind="CA", weight=0.7,
              premises=[n1.node_id], target=n0.node_id,
              provenance={"relation_label": "attack", "labeler_model": "self-test"})
    errs = validate_store([n0, n1], [e0], [c0])
    print("self-test errors:", errs or "none")
    print("fingerprint stability:",
          fingerprint("The FURIN site.") == fingerprint("the furin site"))
