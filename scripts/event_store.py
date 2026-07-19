"""Append-only event log for the epistemic store (guide §3.2).

The store never overwrites state. Every mutation — a claim ingested, a role tagged, a
card added/deactivated, a score computed, a verdict emitted — is appended as one event
carrying its `cause` and, for model-produced events, the verbatim `model_raw_output`.
The current graph is a pure **fold** over the log; `replay()` re-derives identical
state without re-calling any model. This is the substrate that makes a run reproducible
(guide §3.2: "a repeated fold must reproduce identical state").

Relationship to `epistemic_store.py`: that module is the *snapshot* schema (Node /
Edge / Provenance dataclasses + JSONL I/O + validate). This module is the *log* whose
fold produces such a snapshot. They compose; neither replaces the other.

Event line (guide §3.2):
    {ts, event_id, event_type, payload, cause, model, model_raw_output}

Run this file directly for a self-test (fold determinism + a tiny worked log).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent

# Controlled event vocabulary. An engine emitting a type outside this set is a bug we
# want to catch, not silently absorb (mirrors epistemic_store's NODE/EDGE_TYPES stance).
EVENT_TYPES = {
    "claim_added",        # a source claim enters the store (from claims.json)
    "claim_tagged",       # a role tag attached to a claim (evidentiary|conclusory|procedural)
    "attribution_added",  # a source_attribution attached to a claim (guide §3.5)
    "llm_call",           # audit record: one LLM call, verbatim raw output (guide §3.2)
    "card_added",         # a reified relation card (RA|CA|PA) — later phases
    "card_deactivated",   # soft-delete of a card
    "score_computed",     # a DF-QuAD scoring pass result — later phases
    "verdict_emitted",    # a D4 verdict — later phases
}

CAUSE_SENTINELS = {"ingest", "human"}  # other causes are a producing event_id


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    event_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    cause: str = "ingest"                 # event_id | "ingest" | "human"
    model: str | None = None
    model_raw_output: str | None = None
    ts: str | None = None

    def validate(self) -> list[str]:
        errs = []
        if self.event_type not in EVENT_TYPES:
            errs.append(f"{self.event_id}: event_type {self.event_type!r} not in {sorted(EVENT_TYPES)}")
        if not self.event_id:
            errs.append("event missing event_id")
        return errs


def event_id(seq: int) -> str:
    """Stable synthetic id in append order (mirrors epistemic_store.node_id)."""
    return f"ev-{seq:06d}"


# ---------------------------------------------------------------------------
# JSONL I/O — append-only. write_events is used to build a fresh log in order;
# append_events adds to an existing one. Neither rewrites prior lines.
# ---------------------------------------------------------------------------

def _read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_events(path: str | Path) -> list[Event]:
    return [Event(**d) for d in _read_jsonl(path)]


def append_events(path: str | Path, events: Iterable[Event]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as f:  # append — never truncate
        for e in events:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
            n += 1
    return n


def next_seq(path: str | Path) -> int:
    """Count existing events so new ids continue the sequence (append-only growth)."""
    return sum(1 for _ in _read_jsonl(path))


# ---------------------------------------------------------------------------
# Fold — the log projected to current state. Deterministic: depends only on the
# ordered payloads, never on wall-clock `ts`. fold(events) == fold(events).
# ---------------------------------------------------------------------------

def fold(events: Iterable[Event]) -> dict[str, Any]:
    state: dict[str, Any] = {"claims": {}, "cards": {}, "scores": {}, "verdicts": {}}
    claims = state["claims"]
    cards = state["cards"]

    for e in events:
        p = e.payload
        t = e.event_type

        if t == "claim_added":
            cid = p["claim_id"]
            # base record verbatim; role fields fill in via claim_tagged
            claims[cid] = {**p, "role_tags": []}

        elif t == "claim_tagged":
            cid = p["claim_id"]
            c = claims.setdefault(cid, {"claim_id": cid, "role_tags": []})
            # keep EVERY labeler's vote (model x prompt); the operative role is chosen
            # later by resolve_roles(). This is what lets us run several models and
            # "keep both results" while designating one as primary.
            c["role_tags"].append({
                "role": p["role"], "model": e.model,
                "prompt_version": p.get("prompt_version"),
                "labeler_pass": p.get("labeler_pass"),
                "confidence": p.get("confidence"), "justification": p.get("justification"),
            })

        elif t == "attribution_added":
            cid = p["claim_id"]
            c = claims.setdefault(cid, {"claim_id": cid, "role_tags": []})
            c["source_attribution"] = p["source_attribution"]

        elif t == "card_added":
            cards[p["card_id"]] = {**p, "active": True}
        elif t == "card_deactivated":
            if p["card_id"] in cards:
                cards[p["card_id"]]["active"] = False

        elif t == "score_computed":
            state["scores"][p.get("node_id", p.get("id"))] = p
        elif t == "verdict_emitted":
            state["verdicts"][p["verdict_id"]] = p

        elif t == "llm_call":
            pass  # audit-only; raw output preserved on the event, not folded into state

    return state


def replay(path: str | Path) -> dict[str, Any]:
    return fold(read_events(path))


def resolve_roles(state: dict[str, Any], primary_model: str) -> dict[str, Any]:
    """Set the operative role on each claim from `primary_model`'s vote (preferring a
    'primary'-pass tag if several), and record every model's vote so cross-model
    agreement is computable. Keeps all labelers' results; only the *operative* choice
    is the primary model's. Idempotent and deterministic (no wall-clock)."""
    for c in state["claims"].values():
        tags = c.get("role_tags") or []
        if not tags:
            continue
        real = [t for t in tags if t["role"] != "UNPARSED"]
        prim = [t for t in real if t.get("model") == primary_model]
        chosen = (next((t for t in prim if t.get("labeler_pass") == "primary"), None)
                  or (prim[-1] if prim else (real[-1] if real else tags[-1])))
        c["role"] = chosen["role"]
        c["role_confidence"] = chosen.get("confidence")
        c["role_justification"] = chosen.get("justification")
        c["role_model"] = chosen.get("model")
        # last NON-UNPARSED vote per model (a compact cross-model view)
        by_model: dict[str, str] = {}
        for t in tags:
            if t["role"] != "UNPARSED" or t.get("model") not in by_model:
                by_model[t.get("model")] = t["role"]
        c["role_by_model"] = by_model
        c["role_agreement"] = len({r for r in by_model.values() if r != "UNPARSED"}) <= 1
    return state


# ---------------------------------------------------------------------------
# Self-test: a tiny worked log + the fold-determinism invariant the guide requires.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evs = [
        Event(event_id(0), "claim_added",
              {"claim_id": "claim-x", "claim_text": "The FCS was present.", "section_number": "5.7"},
              cause="ingest", ts=utc_now_iso()),
        Event(event_id(1), "llm_call",
              {"prompt_version": "role-v1", "batch": ["claim-x"]},
              cause="ingest", model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
              model_raw_output='[{"claim_id":"claim-x","role":"evidentiary"}]', ts=utc_now_iso()),
        Event(event_id(2), "claim_tagged",
              {"claim_id": "claim-x", "role": "evidentiary", "labeler_pass": "primary",
               "confidence": 0.9, "justification": "reports an observed genomic feature"},
              cause=event_id(1), model="meta-llama/Llama-3.3-70B-Instruct-Turbo", ts=utc_now_iso()),
        Event(event_id(3), "claim_tagged",
              {"claim_id": "claim-x", "role": "evidentiary", "labeler_pass": "secondary",
               "confidence": 0.85, "justification": "a reported feature, not a finding"},
              cause="ingest", model="meta-llama/Llama-3.3-70B-Instruct-Turbo", ts=utc_now_iso()),
    ]
    s1 = fold(evs)
    s2 = fold(evs)
    same = json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    print("fold determinism:", "OK" if same else "FAIL")
    resolve_roles(s1, primary_model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    c = s1["claims"]["claim-x"]
    print("operative role:", c.get("role"), "| by_model:", c.get("role_by_model"),
          "| agreement:", c.get("role_agreement"), "| n_tags:", len(c["role_tags"]))
    print("validate:", [err for e in evs for err in e.validate()] or "none")
