"""Step 4 — source attribution (design.md step 4; guide §3.5).

For each atomic claim, identify WHO asserts or is credited with it — the voice behind
it — so corroboration can later discount correlated sources (ten reprints of one press
release are not ten independent evidences). One gpt-oss pass, domain-general labels.

Source taxonomy (topic-independent):
  - adjudicator  : the document's own author/decision-maker asserting their analysis
                   (overlaps the `conclusory` role; the judge's own finding)
  - party        : an advocate / side in a dispute asserting a position
                   (e.g. "Rootclaim argues …", "the defense claims …")
  - cited_source : an external study / report / dataset / proposal the claim credits
                   (e.g. "a 2020 study", "the DEFUSE proposal")
  - witness      : a specific person's testimony or first-hand observation
  - unattributed : stated as plain fact/background, no source named

Output: artifacts/epistemic/attributions.jsonl — one record per claim
  {claim_id, source_type, source_id, verbatim, source_tier, confidence}
source_tier = T1 when the claim text names the source verbatim, else T3 (inferred).
With --emit-events, also appends `attribution_added` events (event_store folds them to
claim["source_attribution"]).

Mirrors tag_roles.py / type_claims.py: Together client, temp 0, max_tokens=4000,
batched, incremental cache, tolerant JSON parse.

    python scripts/attribute_sources.py --claims artifacts/claims_improved_prompt/claims.json --emit-events
    python scripts/attribute_sources.py --limit 10        # smoke test
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

from event_store import Event, event_id, append_events, next_seq  # noqa: E402

load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "attr-v2"   # v2: default to the document author, not "unattributed"
# Grounded in the document itself: it names its author ("Eric Stansifer") and states
# "I am one of two judges" adjudicating the COVID-19 origins debate. So the default
# source for any un-credited claim is this specific judge in this specific case.
DEFAULT_AUTHOR = "Eric Stansifer (adjudicating judge, COVID-19 origins debate)"
SOURCE_TYPES = {"adjudicator", "party", "cited_source", "witness", "unattributed"}

SYSTEM_PROMPT = (
    "You are a precise argument-mining assistant. For each already-extracted atomic "
    "claim you identify WHO is the source/voice behind it — who asserts it or is "
    "credited with it. The document has a KNOWN author who is asserting everything not "
    "explicitly credited to someone else, so a claim is never truly source-less: at "
    "worst it is the author speaking. You never judge whether a claim is true and never "
    "add information. The labels are domain-general and must not depend on the topic."
)

USER_TEMPLATE = """\
This document is authored by: {author} (the adjudicator). Anything the author states —
including impersonal facts, findings, and judgements ("the odds are ~1/50", "this makes
X less likely") — is asserted BY the author unless the claim explicitly credits someone
else. So the DEFAULT source is the author; never use "unattributed" for a claim the
author is simply stating.

For EACH numbered claim, identify its source:
- "adjudicator": the author ({author}) stating their own analysis, finding, judgement,
  or a plain fact they assert. THIS IS THE DEFAULT — use it whenever no other source is
  explicitly credited, even for impersonal statements.
- "party": an advocate or side in a dispute asserting a position ("Rootclaim argues", "the challenger claims").
- "cited_source": an external study/report/dataset/proposal/institution the claim credits ("a 2020 study", "the DEFUSE proposal", "the WHO report").
- "witness": a specific person's testimony or first-hand observation.
- "unattributed": ONLY if the claim quotes an unnamed speaker whose identity is genuinely impossible to tell (rare).

Return for each claim:
  - "i": the claim's index (integer, copied from input)
  - "source_type": one label from the list above
  - "source_id": a SHORT normalized name for the source. For the author use "{author}".
    Otherwise the credited name ("Rootclaim", "DEFUSE proposal", "WHO"). Reuse the same
    spelling across claims.
  - "verbatim": the exact phrase in the claim that names an EXPLICIT source, or null if
    the attribution is to the author by default (no phrase in the text).
  - "confidence": "high" | "med" | "low"

Return ONLY JSON of the exact form:
{{"attributions": [{{"i": 0, "source_type": "adjudicator", "source_id": "{author}", "verbatim": null, "confidence": "high"}}]}}

CLAIMS:
{claims}
"""


def build_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, max_retries=3, timeout=300)


def batch_prompt(batch: list[dict], author: str) -> str:
    lines = [f"[{i}] {c['claim_text']}" for i, c in enumerate(batch)]
    return USER_TEMPLATE.format(claims="\n".join(lines), author=author)


def call_llm(client: OpenAI, model: str, batch: list[dict], author: str) -> dict:
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=4000,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": batch_prompt(batch, author)}])
    u = resp.usage
    return {"content": resp.choices[0].message.content or "",
            "usage": {"total_tokens": getattr(u, "total_tokens", 0) or 0}}


def parse_attrs(text: str, n: int) -> list[dict]:
    """Tolerant of code fences / stray prose / truncation. Missing indices default to
    unattributed so nothing is silently mislabelled."""
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
                cut = frag.rfind("}")
                if cut != -1:
                    try:
                        data = json.loads(frag[:cut + 1] + "]}")
                    except (json.JSONDecodeError, ValueError):
                        data = None
    rows = data.get("attributions") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    by_i = {r["i"]: r for r in (rows or []) if isinstance(r, dict) and isinstance(r.get("i"), int)}

    out = []
    for i in range(n):
        r = by_i.get(i, {})
        st = r.get("source_type") if r.get("source_type") in SOURCE_TYPES else "unattributed"
        sid = (r.get("source_id") or "").strip()
        vb = r.get("verbatim")
        vb = vb.strip() if isinstance(vb, str) and vb.strip() else None
        conf = r.get("confidence") if r.get("confidence") in {"high", "med", "low"} else "low"
        out.append({"source_type": st, "source_id": sid, "verbatim": vb,
                    "source_tier": "T1" if vb else "T3", "confidence": conf,
                    "recovered": i in by_i})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--author", default=DEFAULT_AUTHOR,
                    help="the document's author — the default source for any claim that "
                         "doesn't credit someone else (per-document; keeps this general)")
    ap.add_argument("--claims", default=str(ROOT / "artifacts" / "claims_improved_prompt" / "claims.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "epistemic" / "attributions.jsonl"))
    ap.add_argument("--events", default=str(ROOT / "artifacts" / "epistemic" / "events.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "epistemic" / "attr_cache.json"))
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--emit-events", action="store_true", help="append attribution_added events")
    ap.add_argument("--max-tokens-budget", type=int, default=2_000_000)
    args = ap.parse_args()

    claims = json.load(open(args.claims))
    if args.limit:
        claims = claims[:args.limit]
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() and not args.no_cache else {}

    print(f"Model:  {args.model} @ Together  | claims: {len(claims)} (batch {args.batch_size})\n")
    client = build_client()
    batches = [claims[i:i + args.batch_size] for i in range(0, len(claims), args.batch_size)]
    labels: list[dict] = []
    total_tokens = new_tokens = 0
    aborted = False

    for bi, batch in enumerate(batches):
        key = f"{PROMPT_VERSION}|{args.author}|{args.model}|" + "|".join(c["claim_id"] for c in batch)
        if key in cache and not args.no_cache:
            entry, tag = cache[key], "cached"
        else:
            if new_tokens >= args.max_tokens_budget:
                print(f"  [batch {bi}] BUDGET REACHED — stopping, saving progress")
                aborted = True
                break
            entry = call_llm(client, args.model, batch, args.author)
            cache[key] = entry
            cache_path.write_text(json.dumps(cache, indent=2))
            new_tokens += entry["usage"]["total_tokens"]
            tag = f"{entry['usage']['total_tokens']:,} tok"
        total_tokens += entry["usage"]["total_tokens"]
        rows = parse_attrs(entry["content"], len(batch))
        labels.extend(rows)
        n_attr = sum(r["source_type"] != "unattributed" for r in rows)
        print(f"  [batch {bi:>2}] {n_attr}/{len(batch)} attributed ({tag}; cum {total_tokens:,} tok)")

    # write snapshot + optional events
    from collections import Counter
    records, events = [], []
    seq = next_seq(args.events) if args.emit_events else 0
    for c, lab in zip(claims[:len(labels)], labels):
        rec = {"claim_id": c["claim_id"], "source_type": lab["source_type"],
               "source_id": lab["source_id"], "verbatim": lab["verbatim"],
               "source_tier": lab["source_tier"], "confidence": lab["confidence"]}
        records.append(rec)
        if args.emit_events:
            events.append(Event(event_id(seq), "attribution_added",
                                {"claim_id": c["claim_id"], "source_attribution": rec},
                                cause="ingest", model=args.model))
            seq += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.emit_events and events:
        append_events(args.events, events)

    by_type = Counter(r["source_type"] for r in records)
    top_src = Counter(r["source_id"] for r in records if r["source_id"]).most_common(8)
    tier = Counter(r["source_tier"] for r in records)
    summary = {"model": args.model, "prompt_version": PROMPT_VERSION,
               "claims_in": len(claims), "attributed": len(records),
               "by_source_type": dict(by_type), "top_sources": dict(top_src),
               "by_tier": dict(tier), "tokens": total_tokens,
               "events_emitted": len(events) if args.emit_events else 0,
               "aborted_on_budget": aborted, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (Path(args.out).parent / "attributions_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Source attribution summary ===")
    print(f"  records      : {len(records)} -> {args.out}")
    print(f"  by type      : {dict(by_type)}")
    print(f"  top sources  : {dict(top_src)}")
    print(f"  by tier      : {dict(tier)}  (T1 = source named verbatim)")
    if aborted:
        print("  NOTE: aborted on budget; output is partial.")


if __name__ == "__main__":
    main()
