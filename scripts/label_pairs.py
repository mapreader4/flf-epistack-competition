"""Step D — label candidate pairs into cards (design.md step 7, lines 96-99).

Reads `candidate_pairs.json` (the funnel's <=1000 survivors) and asks gpt-oss, for
each ordered pair (A, B): does A **support**, **attack**, or have **no relation** to
B? Each support/attack becomes a reified `Card` (RA/CA) with A as premise and B as
target. No-relation pairs are dropped. Output: `artifacts/epistemic/cards.jsonl`
(contract 4), plus optional `card_added` + `llm_call` events (`--emit-events`).

Mirrors `type_claims.py`: Together/OpenAI client, temperature 0, `[i]`-indexed
batches, code-fence/truncation-tolerant JSON parser, incremental cache keyed on
(prompt_version, model, pair_ids), token budget with partial-save. Per project policy
the model is **gpt-oss-120b** with `max_tokens=4000` (reasoning model — truncates
otherwise). MVP builds single-premise cards; joint cards (clustering co-targeting CAs)
and PA/outweighs are a later pass.

Usage:
    python scripts/label_pairs.py                    # label all candidate pairs
    python scripts/label_pairs.py --limit 40         # smoke test
    python scripts/label_pairs.py --emit-events      # also append to events.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import (  # noqa: E402
    Card, read_nodes, write_cards, validate_store, card_id,
)
from event_store import Event, event_id, append_events, next_seq  # noqa: E402

load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"     # project default (see CLAUDE.md model policy)
PROMPT_VERSION = "label-v1"
RATE_PER_MTOK = 0.30
KIND_OF = {"support": "RA", "attack": "CA"}
WEIGHT_OF = {"high": 0.85, "med": 0.6, "low": 0.35}

SYSTEM_PROMPT = (
    "You are a precise argument-mining assistant. You are given ordered pairs of "
    "atomic statements (A, B). For each pair you decide the DIALECTICAL relation of A "
    "to B: does A support B, attack B, or neither? You judge the argumentative "
    "relation, not whether either statement is true, and you never add information. "
    "The labels are domain-general and must not depend on the topic."
)

USER_TEMPLATE = """\
For EACH numbered pair below, decide how statement A relates to statement B:
  - "support": A, if accepted, raises the plausibility of B (evidence for it, a reason to believe it).
  - "attack":  A, if accepted, lowers the plausibility of B (evidence against it, a rebuttal, a contradiction).
  - "none":    A and B are about different things, or A has no clear bearing on B.

Be conservative: choose "none" unless there is a genuine evidential or dialectical link.

Return for each pair:
  - "i": the pair index (integer, copied from input)
  - "label": "support" | "attack" | "none"
  - "subtype": for an attack only, one of "rebuts" (attacks B's conclusion),
    "undercuts" (attacks the inference to B), "undermines" (attacks a premise of B);
    otherwise null.
  - "confidence": "high" | "med" | "low"

Return ONLY JSON of the exact form:
{{"labels": [{{"i": 0, "label": "none", "subtype": null, "confidence": "high"}}]}}

PAIRS:
{pairs}
"""


def build_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, max_retries=3, timeout=300)


def batch_prompt(batch: list[dict], text: dict[str, str]) -> str:
    lines = []
    for i, p in enumerate(batch):
        lines.append(f"[{i}] A: {text[p['src']]}\n    B: {text[p['dst']]}")
    return USER_TEMPLATE.format(pairs="\n".join(lines))


def call_llm(client: OpenAI, model: str, batch: list[dict], text: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": batch_prompt(batch, text)},
    ]
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=4000, messages=messages)
    u = resp.usage
    return {"content": resp.choices[0].message.content or "",
            "usage": {"total_tokens": getattr(u, "total_tokens", 0) or 0}}


def parse_labels(text: str, n: int) -> list[dict]:
    """Tolerant of code fences / stray prose / truncation. Missing indices default to
    'none' so nothing is silently mislabelled as a relation."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    data = None
    try:
        data = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1:
            frag = t[s:e + 1]
            try:
                data = json.loads(frag)
            except (json.JSONDecodeError, ValueError):
                # salvage truncated array: close at last complete object
                cut = frag.rfind("}")
                if cut != -1:
                    try:
                        data = json.loads(frag[:cut + 1] + "]}")
                    except (json.JSONDecodeError, ValueError):
                        data = None
    rows = data.get("labels") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    by_i = {r["i"]: r for r in rows if isinstance(r, dict) and isinstance(r.get("i"), int)}

    out = []
    for i in range(n):
        r = by_i.get(i, {})
        label = r.get("label") if r.get("label") in {"support", "attack", "none"} else "none"
        conf = r.get("confidence") if r.get("confidence") in WEIGHT_OF else "low"
        sub = r.get("subtype") if r.get("subtype") in {"rebuts", "undercuts", "undermines"} else None
        out.append({"label": label, "subtype": sub, "confidence": conf, "recovered": i in by_i})
    return out


def label_pairs(pairs: list[dict], text: dict[str, str], *, model: str = DEFAULT_MODEL,
                cache_path: str | Path | None = None, batch_size: int = 20,
                max_tokens_budget: int = 1_500_000, no_cache: bool = False,
                client: "OpenAI | None" = None, start_index: int = 0,
                emit_events: bool = False, events_path: str | Path | None = None):
    """Label candidate pairs into Cards. Returns (cards, events, stats).

    Extracted so the whole build can run from one command: pairing_funnel.py calls
    this directly instead of shelling out to this script. `start_index` continues the
    card-id sequence when appending to an existing store (incremental ingest)."""
    if client is None:
        client = build_client()
    cache_path = Path(cache_path or (ROOT / "outputs" / "epistemic" / "label_cache.json"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not no_cache else {}

    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    labels: list[dict] = []
    total_tokens = new_tokens = 0
    aborted = False
    for bi, batch in enumerate(batches):
        key = f"{PROMPT_VERSION}|{model}|" + "|".join(f"{p['src']}~{p['dst']}" for p in batch)
        if key in cache and not no_cache:
            entry, tag = cache[key], "cached"
        else:
            if new_tokens >= max_tokens_budget:
                print(f"  [batch {bi}] BUDGET REACHED ({new_tokens:,} tok) — stopping, saving progress")
                aborted = True
                break
            entry = call_llm(client, model, batch, text)
            cache[key] = entry
            cache_path.write_text(json.dumps(cache, indent=2))
            new_tokens += entry["usage"]["total_tokens"]
            tag = f"{entry['usage']['total_tokens']:,} tok"
        total_tokens += entry["usage"]["total_tokens"]
        rows = parse_labels(entry["content"], len(batch))
        for p, r in zip(batch, rows):
            labels.append({**p, **r, "raw_output": entry["content"] if emit_events else None})
        n_rel = sum(r["label"] != "none" for r in rows)
        print(f"  [batch {bi:>2}] {n_rel}/{len(batch)} relations ({tag}; cum {total_tokens:,} tok)")

    cards: list[Card] = []
    events: list[Event] = []
    seq = next_seq(events_path) if (emit_events and events_path) else 0
    for lab in labels:
        if lab["label"] == "none":
            continue
        cid = card_id(start_index + len(cards))
        prov = {"labeler_model": model, "relation_label": lab["label"],
                "subtype": lab["subtype"], "raw_confidence": lab["confidence"],
                "channel": lab["channel"], "prompt_version": PROMPT_VERSION}
        if emit_events:
            call_ev = Event(event_id=event_id(seq), event_type="llm_call",
                            payload={"pair": f"{lab['src']}~{lab['dst']}", "prompt_version": PROMPT_VERSION},
                            cause="ingest", model=model, model_raw_output=lab.get("raw_output"))
            seq += 1
            prov["llm_call_event"] = call_ev.event_id
            events.append(call_ev)
        card = Card(card_id=cid, kind=KIND_OF[lab["label"]],
                    weight=WEIGHT_OF[lab["confidence"]], premises=[lab["src"]],
                    target=lab["dst"], provenance=prov, tier="T3")
        cards.append(card)
        if emit_events:
            events.append(Event(event_id=event_id(seq), event_type="card_added",
                                payload={k: v for k, v in card.__dict__.items()},
                                cause="ingest", model=model))
            seq += 1

    stats = {"pairs_in": len(pairs), "labels": len(labels), "cards_out": len(cards),
             "tokens": {"total": total_tokens, "new": new_tokens}, "aborted": aborted}
    return cards, events, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pairs", default=str(ROOT / "artifacts" / "epistemic" / "candidate_pairs.json"))
    ap.add_argument("--nodes", default=str(ROOT / "artifacts" / "epistemic" / "nodes.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "epistemic" / "cards.jsonl"))
    ap.add_argument("--events", default=str(ROOT / "artifacts" / "epistemic" / "events.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "epistemic" / "label_cache.json"))
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--emit-events", action="store_true", help="append card_added + llm_call to events.jsonl")
    ap.add_argument("--max-tokens-budget", type=int, default=1_500_000)
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())["pairs"]
    if args.limit:
        pairs = pairs[:args.limit]
    nodes = read_nodes(args.nodes)
    text = {n.node_id: n.canonical_text for n in nodes}

    print(f"Model:  {args.model} @ Together  (max_tokens=4000)")
    print(f"Pairs:  {len(pairs)}  (batch {args.batch_size})\n")

    cards, events, stats = label_pairs(
        pairs, text, model=args.model, cache_path=args.cache, batch_size=args.batch_size,
        max_tokens_budget=args.max_tokens_budget, no_cache=args.no_cache,
        emit_events=args.emit_events, events_path=args.events)
    total_tokens = stats["tokens"]["total"]
    new_tokens = stats["tokens"]["new"]
    aborted = stats["aborted"]

    errs = validate_store(nodes, [], cards)
    if errs:
        print("\n  VALIDATION ERRORS:", errs[:10])
    n_written = write_cards(args.out, cards)
    if args.emit_events and events:
        append_events(args.events, events)

    from collections import Counter
    by_kind = Counter(c.kind for c in cards)
    by_channel = Counter(c.provenance["channel"] for c in cards)
    summary = {"model": args.model, "prompt_version": PROMPT_VERSION,
               "pairs_in": len(pairs), "cards_out": n_written,
               "by_kind": dict(by_kind), "by_channel": dict(by_channel),
               "relation_rate": round(n_written / max(1, stats["labels"]), 3),
               "tokens": {"total": total_tokens, "new": new_tokens},
               "est_cost_usd": round(total_tokens / 1e6 * RATE_PER_MTOK, 4),
               "validation_errors": len(errs), "aborted_on_budget": aborted,
               "events_emitted": len(events) if args.emit_events else 0,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (Path(args.out).parent / "cards_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Card labeling summary ===")
    print(f"  cards written : {n_written} -> {args.out}")
    print(f"  by kind       : {dict(by_kind)}  (RA=support, CA=attack)")
    print(f"  by channel    : {dict(by_channel)}")
    print(f"  relation rate : {summary['relation_rate']}  (fraction of pairs that were a relation)")
    print(f"  est cost      : ${summary['est_cost_usd']}")
    if aborted:
        print("  NOTE: aborted on token budget; cards.jsonl is partial.")


if __name__ == "__main__":
    main()
