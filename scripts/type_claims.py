"""Engine A — claim typing. Populate the epistemic store's NODE layer.

Reads the 590 provenance-anchored claims (`artifacts/claims/claims.json`) and, with
one GPT-OSS pass, assigns each an epistemic *role* (hypothesis / evidence /
assumption / estimate / rebuttal / verdict / background / claim) plus two cheap
fields that are already latent in the claim text (CONTEXT.md §4):

  - `number`      — a verbatim numeric/probability token if the claim states one
                    ("1/50", "20x", "8%"), else null. This is what mechanically
                    separates the ~8% quantitative core from the ~92% qualitative
                    bulk (decision 2026-07-16-e), without inventing any numbers.
  - `attribution` — who the claim says asserts it ("Rootclaim", "the judge", "a
                    study"), else null.

Output is `artifacts/epistemic/nodes.jsonl` in the container schema
(`scripts/epistemic_store.py`). The claim text and its provenance are tier **T1**
(byte-exact, reused verbatim); the *typing decision* is model-inferred and recorded
as tier **T3** in `meta`, so we never let an inferred label masquerade as extracted
fact. This is engine A of three (typing / pairwise-linking / sub-question grouping);
it writes only nodes — no edges yet.

Mirrors scripts/extract_claims.py conventions: Together/OpenAI client, temperature
0, incremental JSON cache keyed on (batch, prompt version, model), a token budget.

Usage:
    python scripts/type_claims.py                 # all 590 claims
    python scripts/type_claims.py --limit 40      # smoke test on the first 40
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
    Node, Provenance, node_id, fingerprint, write_jsonl, read_nodes,
    validate_store, NODE_TYPES,
)

load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "type-v1"
RATE_PER_MTOK = 0.30  # rough gpt-oss-120b blended $/1M tok; friendly estimate only

SYSTEM_PROMPT = (
    "You are a precise argument-mining assistant. You classify already-extracted, "
    "atomic claims by their EPISTEMIC ROLE in an argument. You never evaluate "
    "whether a claim is true, and you never add information; you only label the "
    "role each claim plays and copy out any number or attribution it already "
    "contains. The labels are domain-general and must not depend on the topic."
)

# Domain-agnostic role definitions. Kept verbatim-aligned with epistemic_store.NODE_TYPES.
ROLE_GUIDE = """\
ROLES (choose exactly one per claim):
- hypothesis: a top-level competing explanation being weighed (e.g. "the virus arose by zoonotic spillover").
- evidence: an observation, finding, study result, or fact offered as bearing on a hypothesis.
- estimate: a claim whose CONTENT IS A NUMBER — a probability, prior, rate, or likelihood ratio / Bayes factor (e.g. "the probability is 1/50").
- assumption: an explicit premise taken as given, including independence or modelling assumptions.
- rebuttal: a claim asserted specifically to COUNTER or ATTACK another claim or position.
- verdict: a conclusion, decision, ruling, or posterior judgement.
- background: a plain contextual fact that carries no argumentative load here (definitions, taxonomy, dates).
- claim: fallback — an asserted proposition that fits none of the above cleanly.

Guidance: prefer `estimate` over `evidence` ONLY when the number is the point of the
claim. A finding that happens to mention a number is still `evidence`. Reserve
`hypothesis` for the small number of top-level rival explanations, not every claim."""

USER_TEMPLATE = """\
{guide}

For EACH numbered claim below, return:
  - "i": the claim's index (integer, copied from the input)
  - "type": one role from the list above
  - "number": the FIRST verbatim numeric/probability token the claim states ("1/50",
    "20x", "8%", "roughly doubles"), or null if it states no number. Copy it
    verbatim; do not compute or normalize.
  - "attribution": who the claim says is asserting it ("Rootclaim", "the judge",
    "a 2020 study"), or null if the claim is unattributed.
  - "confidence": your confidence in the "type" label — "high", "med", or "low".

Return ONLY JSON, no prose, of the exact form:
{{"types": [{{"i": 0, "type": "...", "number": null, "attribution": null, "confidence": "high"}}]}}

CLAIMS:
{claims}
"""


def build_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, max_retries=3, timeout=300)


def batch_prompt(batch: list[dict]) -> str:
    lines = [f"[{i}] {c['claim_text']}" for i, c in enumerate(batch)]
    return USER_TEMPLATE.format(guide=ROLE_GUIDE, claims="\n".join(lines))


def call_llm(client: OpenAI, model: str, batch: list[dict]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": batch_prompt(batch)},
    ]
    resp = client.chat.completions.create(model=model, temperature=0, messages=messages)
    u = resp.usage
    return {
        "content": resp.choices[0].message.content or "",
        "usage": {"total_tokens": getattr(u, "total_tokens", 0) or 0},
    }


def parse_types(text: str, n: int) -> list[dict]:
    """Pull the per-claim label objects out, tolerant of code fences / stray prose.
    Returns a list aligned to input indices (length n); missing entries default to
    an unclassified `claim` at low confidence so nothing is silently dropped."""
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
            try:
                data = json.loads(t[s:e + 1])
            except (json.JSONDecodeError, ValueError):
                data = None
    rows = []
    if isinstance(data, dict) and isinstance(data.get("types"), list):
        rows = data["types"]
    elif isinstance(data, list):
        rows = data

    by_i: dict[int, dict] = {}
    for r in rows:
        if isinstance(r, dict) and isinstance(r.get("i"), int):
            by_i[r["i"]] = r

    out = []
    for i in range(n):
        r = by_i.get(i, {})
        typ = r.get("type")
        if typ not in NODE_TYPES:
            typ, conf = "claim", "low"
        else:
            conf = r.get("confidence") if r.get("confidence") in {"high", "med", "low"} else "low"
        num = r.get("number")
        num = num.strip() if isinstance(num, str) and num.strip() else None
        attr = r.get("attribution")
        attr = attr.strip() if isinstance(attr, str) and attr.strip() else None
        out.append({"type": typ, "number": num, "attribution": attr,
                    "confidence": conf, "recovered": i in by_i})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--claims", default=str(ROOT / "artifacts" / "claims" / "claims.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "epistemic" / "nodes.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "epistemic" / "type_cache.json"))
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="only type the first N claims (0 = all)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-tokens-budget", type=int, default=400_000)
    args = ap.parse_args()

    claims = json.loads(Path(args.claims).read_text())
    if args.limit:
        claims = claims[:args.limit]
    out_path = Path(args.out)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model:   {args.model} @ Together")
    print(f"Claims:  {len(claims)}  (batch {args.batch_size})")
    print(f"Out:     {out_path}\n")

    client = build_client()

    cache: dict[str, dict] = {}
    if cache_path.exists() and not args.no_cache:
        cache = json.loads(cache_path.read_text())

    batches = [claims[i:i + args.batch_size] for i in range(0, len(claims), args.batch_size)]
    labels: list[dict] = []
    total_tokens = new_tokens = 0
    aborted = False

    for bi, batch in enumerate(batches):
        # cache key is the batch's claim_ids + prompt version + model: stable across
        # re-runs, invalidates if the claim set or prompt changes.
        key = f"{PROMPT_VERSION}|{args.model}|{'|'.join(c['claim_id'] for c in batch)}"
        if key in cache and not args.no_cache:
            entry = cache[key]
            tag = "cached"
        else:
            if new_tokens >= args.max_tokens_budget:
                print(f"  [batch {bi}] BUDGET REACHED ({new_tokens:,} tok) — stopping, saving progress")
                aborted = True
                break
            entry = call_llm(client, args.model, batch)
            cache[key] = entry
            cache_path.write_text(json.dumps(cache, indent=2))
            new_tokens += entry["usage"]["total_tokens"]
            tag = f"{entry['usage']['total_tokens']:,} tok"
        total_tokens += entry["usage"]["total_tokens"]

        rows = parse_types(entry["content"], len(batch))
        labels.extend(rows)
        n_rec = sum(r["recovered"] for r in rows)
        print(f"  [batch {bi:>2}] {n_rec}/{len(batch)} labelled ({tag}; "
              f"cum {total_tokens:,} tok ~= ${total_tokens/1e6*RATE_PER_MTOK:.4f})")

    # Assemble nodes. Ids are sequential in claim order → stable across re-runs.
    nodes: list[Node] = []
    typed_claims = claims[:len(labels)]
    for i, (c, lab) in enumerate(zip(typed_claims, labels)):
        prov = Provenance(
            claim_id=c["claim_id"], chunk_id=c.get("chunk_id"),
            section_number=c.get("section_number"), quote=c.get("quote"),
            span=c.get("span"), tier="T1",
        )
        payload = {}
        if lab["number"]:
            payload["number"] = lab["number"]
        if lab["attribution"]:
            payload["attribution"] = lab["attribution"]
        nodes.append(Node(
            node_id=node_id(i), type=lab["type"], canonical_text=c["claim_text"],
            fingerprint=fingerprint(c["claim_text"]), provenance=[prov],
            payload=payload, tier="T1",
            meta={"type_tier": "T3", "type_source": args.model,
                  "type_confidence": lab["confidence"],
                  "is_appendix": c.get("is_appendix"), "is_excluded": c.get("is_excluded")},
        ))

    errs = validate_store(nodes, [])
    if errs:
        print("\n  VALIDATION ERRORS:")
        for e in errs[:20]:
            print("   -", e)

    n_written = write_jsonl(out_path, nodes)

    # --- inspection summary: this is the point of slice A — look before building. ---
    from collections import Counter
    type_counts = Counter(n.type for n in nodes)
    with_number = [n for n in nodes if n.payload.get("number")]
    with_attr = Counter(n.payload.get("attribution") for n in nodes if n.payload.get("attribution"))
    conf_counts = Counter(n.meta["type_confidence"] for n in nodes)

    summary = {
        "model": args.model, "prompt_version": PROMPT_VERSION,
        "claims_in": len(claims), "nodes_out": n_written,
        "type_distribution": dict(type_counts.most_common()),
        "quantitative_core": {
            "nodes_with_number": len(with_number),
            "pct_with_number": round(100 * len(with_number) / max(1, n_written), 1),
            "by_type": dict(Counter(n.type for n in with_number).most_common()),
        },
        "attribution_top": dict(with_attr.most_common(10)),
        "type_confidence": dict(conf_counts),
        "tokens": {"total": total_tokens, "new_this_run": new_tokens},
        "est_cost_usd": round(total_tokens / 1e6 * RATE_PER_MTOK, 4),
        "validation_errors": len(errs), "aborted_on_budget": aborted,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_path.parent / "nodes_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Node typing summary ===")
    print(f"  nodes written        : {n_written} -> {out_path}")
    print(f"  type distribution    : {summary['type_distribution']}")
    print(f"  quantitative core    : {summary['quantitative_core']['nodes_with_number']} "
          f"({summary['quantitative_core']['pct_with_number']}%) carry a number "
          f"-> {summary['quantitative_core']['by_type']}")
    print(f"  attribution (top)    : {summary['attribution_top']}")
    print(f"  type confidence      : {summary['type_confidence']}")
    print(f"  est cost             : ${summary['est_cost_usd']}")
    if aborted:
        print("\n  NOTE: aborted on token budget; nodes.jsonl is partial.")
    print(f"\n  summary -> {out_path.parent / 'nodes_summary.json'}")


if __name__ == "__main__":
    main()
