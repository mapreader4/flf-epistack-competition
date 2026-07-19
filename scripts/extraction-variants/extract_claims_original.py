"""Claim extraction with provenance (CAMS-style §3.2) for the epistemic layer.

Decompose curated sections of eric_decision.pdf into atomic, decontextualized
{claim, quote} pairs with an LLM, then resolve each quote's provenance
*deterministically* with a tiered exact-then-fuzzy string matcher: first locate
the quote in the section's raw text (tier A), then attribute the matched span to
one of the existing 195 chunks (tier B).

We never ask the model for character offsets. Instruction-tuned models miscount
positions and hallucinate spans; provenance is resolved by string matching after
the fact instead. This mirrors HippoRAG's own separation of extraction from graph
construction (see scripts/run_hipporag_index.py) and reuses the PDF/section
parsing from scripts/chunk_decision.py rather than reimplementing it.

Usage:
    python scripts/extraction-variants/extract_claims_original.py                 # curated 12-section sample
    python scripts/extraction-variants/extract_claims_original.py --sections 7,7.1 # explicit subset
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ -- import sibling module

# Reuse the PDF/section parsing verbatim rather than re-deriving it. build_body
# already runs clean() on every page, so `body` (and thus each section's raw
# slice) is ligature/NFKC-normalized; the chunk texts in data/chunks.json are the
# same body text with whitespace collapsed by pack_chunks().
from chunk_decision import (  # noqa: E402
    extract_pages,
    parse_toc,
    build_body,
    locate_headings,
)
# The tiered exact-then-fuzzy matcher is shared with chunk_decision.py (which
# uses it to resolve each chunk's own page -- see its module docstring), not
# reimplemented here.
from span_match import locate  # noqa: E402

load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

PROMPT_VERSION = "v1"

# Curated sample, chosen for downstream usefulness rather than document order:
#   7, 7.1-7.6  -> Analysis: hypotheses, priors, and the final verdict converge.
#   5.4, 5.4.1-5.4.4 -> the funding-proposal excerpt section: evidence-dense and
#                       quote-heavy (built from excerpts of a real document), a
#                       stress test for the matcher on text that is itself quotation.
DEFAULT_SECTIONS = [
    "7", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6",
    "5.4", "5.4.1", "5.4.2", "5.4.3", "5.4.4",
]

# Together's listed rate for Llama-3.3-70B-Instruct-Turbo, USD per 1M tokens,
# as of 2026-07-19 (together.ai/pricing). Only used for a friendly running
# estimate; not authoritative -- re-check if it drifts again.
RATE_PER_MTOK = 1.04


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise claim-extraction assistant for an argument-mining and "
    "provenance-indexing pipeline over a public, judicial-style decision "
    "document. Your only job is to LOCATE and ATTRIBUTE the claims the text "
    "makes, pairing each with a verbatim supporting quotation copied from the "
    "text. You do not evaluate, endorse, correct, or add information of your "
    "own; you only mine and attribute what the passage itself states."
)

USER_TEMPLATE = """\
Document: "eric_decision.pdf" -- a judge's written decision in a structured public \
debate about COVID-19 origins (zoonosis vs. lab-leak), weighing competing \
hypotheses under explicit Bayesian argument.
Section {number}: {title}

Task: Decompose the SECTION TEXT below into atomic, self-contained claims. For \
each claim, return a verbatim quotation from the section text that licenses it.

Rules:
- Each claim must be ATOMIC (a single assertion) and DECONTEXTUALIZED: resolve \
pronouns, bridging references, and elided subjects/times/locations so the claim \
stands alone, naming the specific entities involved rather than leaving them \
implicit. (Example: if the surrounding text is about the furin cleavage site and \
says "It roughly doubled the odds of the lab-leak hypothesis," extract the claim \
"The presence of the furin cleavage site in SARS-CoV-2 roughly doubles the odds \
of the lab-leak hypothesis.")
- The "quote" MUST be an exact, verbatim, contiguous substring of the section \
text -- copy it character-for-character. Do NOT paraphrase, normalize whitespace, \
fix typos, or stitch together non-contiguous fragments. Do NOT include character \
offsets or line numbers.
- Extract only claims actually asserted or reported in this section. Keep any \
attribution inside the decontextualized claim (e.g. "Rootclaim argues that ...", \
"The judge concludes that ...").
- If the section text contains no substantive claims (e.g. it is only a heading \
or a fragment), return an empty list.

Return ONLY JSON, with no prose before or after, of the exact form:
{{"claims": [{{"claim": "<decontextualized claim>", "quote": "<verbatim substring>"}}]}}

SECTION TEXT:
\"\"\"
{raw}
\"\"\"
"""


def build_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing. Copy .env.example to .env and fill it in.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, max_retries=3, timeout=300)


def preflight(client: OpenAI, model: str) -> None:
    """Together rejects some params OpenAI accepts; fail here, not 12 sections in."""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
    )
    msg = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    print(f"  preflight LLM ok -> {msg[:60]!r} "
          f"({getattr(usage, 'completion_tokens', '?')} completion tokens)")


def call_llm(client: OpenAI, model: str, number: str, title: str, raw: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(number=number, title=title, raw=raw)},
    ]
    resp = client.chat.completions.create(model=model, temperature=0, messages=messages)
    u = resp.usage
    return {
        "content": resp.choices[0].message.content or "",
        "usage": {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        },
    }


def _try_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


# Matches "claim": "<claim>", "quote": "<quote>" } using the literal field
# delimiters as anchors rather than requiring the surrounding response to be
# valid JSON. Regex backtracking makes this safe against a stray unescaped
# quote inside <claim>/<quote> (the model's most common JSON error -- it
# quotes a phrase from the source text without escaping the inner marks,
# which breaks json.loads and, since the bad character persists, every
# {...}/[...] substring re-parse too): a "\"" that isn't immediately followed
# by the next field's exact delimiter doesn't satisfy the pattern, so the
# match keeps extending past it to the real field boundary instead of
# terminating early.
_PAIR_RE = re.compile(
    r'"claim"\s*:\s*"(.*?)"\s*,\s*"quote"\s*:\s*"(.*?)"\s*\}',
    re.DOTALL,
)


def _unescape_lenient(s: str) -> str:
    """Undo the handful of JSON escapes the model actually uses, for text
    captured by regex rather than a real JSON parser. Protecting "\\\\" first
    keeps a literal backslash from being mistaken for the start of one of the
    other escapes handled below."""
    s = s.replace("\\\\", "\x00")
    s = s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    return s.replace("\x00", "\\")


def _regex_extract_pairs(text: str) -> list[dict]:
    pairs = []
    for m in _PAIR_RE.finditer(text):
        claim = _unescape_lenient(m.group(1).strip())
        quote = _unescape_lenient(m.group(2).strip())
        if claim and quote:
            pairs.append({"claim": claim, "quote": quote})
    return pairs


def parse_claims(text: str) -> tuple[list[dict], int]:
    """Tolerantly pull {claim, quote} pairs out of the model's response.

    Returns (valid_pairs, n_invalid). n_invalid counts objects that looked like
    claim records but were missing a field, so they can be logged/reported.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()

    data = _try_json(t)
    if data is None:
        for pat in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pat, t, re.DOTALL)
            if m:
                data = _try_json(m.group(0))
                if data is not None:
                    break
    if data is None:
        # Strict JSON parsing failed everywhere -- fall back to a structural
        # match on the known claim/quote field pattern instead of dropping
        # the whole response's claims silently.
        return _regex_extract_pairs(t), 0

    if isinstance(data, dict):
        if isinstance(data.get("claims"), list):
            data = data["claims"]
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if lists else [data]
    if not isinstance(data, list):
        return [], 0

    valid: list[dict] = []
    for d in data:
        if (isinstance(d, dict)
                and isinstance(d.get("claim"), str) and d["claim"].strip()
                and isinstance(d.get("quote"), str) and d["quote"].strip()):
            valid.append({"claim": d["claim"].strip(), "quote": d["quote"].strip()})
    return valid, len(data) - len(valid)


# ---------------------------------------------------------------------------
# Two-tier span resolution (deterministic)
# ---------------------------------------------------------------------------
# locate() itself lives in span_match.py (imported above) and is shared with
# chunk_decision.py's page-offset resolution -- not reimplemented here.


def resolve(pair: dict, section_number: str, section_raw: str,
            section_chunks: list[dict], threshold: float) -> dict | None:
    """Resolve one {claim, quote} pair to a chunk-relative span.

    Returns a claims.json record, or None if the quote could not be located in
    the section text at all (tier A reject -> the claim is dropped and counted).
    """
    a = locate(pair["quote"], section_raw, threshold)
    if a["tier"] == "reject":
        return None
    matched_text = section_raw[a["start"]:a["end"]]

    # Tier B: attribute the matched span to one of the section's chunks.
    best_chunk = None
    best_res = None
    best_key = None
    for c in sorted(section_chunks, key=lambda c: c["chunk_order_in_section"]):
        r = locate(matched_text, c["text"], threshold)
        if r["tier"] == "reject":
            continue
        key = (r["tier"] == "exact", r["score"])
        if best_key is None or key > best_key:
            best_chunk, best_res, best_key = c, r, key

    is_appendix = section_number[:1].isalpha()
    is_excluded = section_number.startswith("C")

    if best_chunk is None:
        chunk_id = None
        chunk_match = "split_across_chunks"
        span = None
    else:
        chunk_id = best_chunk["chunk_id"]
        chunk_match = best_res["tier"]
        span = [best_res["start"], best_res["end"]]

    fingerprint = f"{chunk_id or ''}|{pair['claim']}|{pair['quote']}"
    return {
        "claim_id": "claim-" + hashlib.md5(fingerprint.encode()).hexdigest(),
        "section_number": section_number,
        "chunk_id": chunk_id,
        "claim_text": pair["claim"],
        "quote": pair["quote"],
        "span": span,
        "section_match_tier": a["tier"],
        "section_match_score": a["score"],
        "chunk_match": chunk_match,
        "is_appendix": is_appendix,
        "is_excluded": is_excluded,
    }


# ---------------------------------------------------------------------------
# Section-text recovery
# ---------------------------------------------------------------------------

def recover_sections() -> dict[str, dict]:
    """Reproduce (title, raw text) per section, one step before chunk_decision.py
    destroys paragraph structure. Mirrors the top of its main()."""
    pages = extract_pages()
    toc = parse_toc(pages)
    body, _offsets = build_body(pages, first_body_page=4)
    sections = locate_headings(body, toc)
    out: dict[str, dict] = {}
    for sec in sections:
        raw = body[sec["start"]:sec["end"]].strip()
        out[sec["number"]] = {"title": sec["title"], "raw": raw}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # Calibrated empirically on the curated 12-section sample, not taken from the
    # paper: genuine fuzzy matches (whitespace/hyphenation/short footnote-number
    # splices) landed at 0.913-0.995 (12 cases). The one dropped claim in that run
    # scored 0.778 -- a footnote splice that inserted a full sentence (not just a
    # number) mid-quote, which is a real matcher limitation, not a fabrication. The
    # gap between 0.913 and 0.778 is wide, so 0.90 sits in a clean valley between
    # "real quote, PDF-extraction noise" and "matcher can't bridge this gap."
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="min similarity (0-1) to accept a fuzzy span match")
    ap.add_argument("--sections", default=",".join(DEFAULT_SECTIONS),
                    help="comma-separated section numbers (default: curated sample)")
    ap.add_argument("--save-dir", default=str(ROOT / "artifacts" / "claims"))
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "claims" / "llm_cache.json"))
    ap.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses")
    ap.add_argument("--max-tokens-budget", type=int, default=200_000,
                    help="abort (saving progress) once this many NEW tokens are spent")
    args = ap.parse_args()

    want = [s.strip() for s in args.sections.split(",") if s.strip()]
    save_dir = Path(args.save_dir)
    cache_path = Path(args.cache)

    print(f"Model:      {args.model} @ Together")
    print(f"Threshold:  fuzzy >= {args.threshold}")
    print(f"Sections:   {len(want)} -> {', '.join(want)}")
    print(f"Save dir:   {save_dir}")
    print(f"Budget:     {args.max_tokens_budget:,} new tokens\n")

    client = build_client()
    preflight(client, args.model)

    print("\n=== Recovering per-section raw text ===")
    sections = recover_sections()
    print(f"  recovered {len(sections)} sections from the PDF")

    chunks = json.loads((ROOT / "data" / "chunks.json").read_text())
    chunks_by_section: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_section.setdefault(c["section_number"], []).append(c)

    # LLM cache keyed on (section, prompt version, model): re-runs are free unless
    # the section text or prompt changes.
    cache: dict[str, dict] = {}
    if cache_path.exists() and not args.no_cache:
        cache = json.loads(cache_path.read_text())
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def cache_key(number: str) -> str:
        return f"{number}|{PROMPT_VERSION}|{args.model}"

    print("\n=== Extraction ===")
    raw_pairs: list[dict] = []
    invalid_total = 0
    new_tokens = 0
    total_tokens = 0
    aborted = False

    for number in want:
        sec = sections.get(number)
        if sec is None:
            print(f"  [{number}] not found in parsed sections -- skipping")
            continue
        if not sec["raw"]:
            print(f"  [{number}] empty section text -- skipping")
            continue

        key = cache_key(number)
        cached = key in cache and not args.no_cache
        if cached:
            entry = cache[key]
            total_tokens += entry["usage"]["total_tokens"]
        else:
            if new_tokens >= args.max_tokens_budget:
                print(f"  [{number}] BUDGET REACHED ({new_tokens:,} new tokens) -- "
                      f"aborting, saving progress so far")
                aborted = True
                break
            entry = call_llm(client, args.model, number, sec["title"], sec["raw"])
            cache[key] = entry
            cache_path.write_text(json.dumps(cache, indent=2))  # persist incrementally
            new_tokens += entry["usage"]["total_tokens"]
            total_tokens += entry["usage"]["total_tokens"]

        pairs, invalid = parse_claims(entry["content"])
        invalid_total += invalid
        for p in pairs:
            raw_pairs.append({"section_number": number, "claim": p["claim"], "quote": p["quote"]})

        tag = "cached" if cached else f"{entry['usage']['total_tokens']:,} tok"
        est = total_tokens / 1e6 * RATE_PER_MTOK
        print(f"  [{number}] {len(pairs):>3} claims"
              f"{f', {invalid} invalid' if invalid else ''} "
              f"({tag}; cumulative {total_tokens:,} tok ~= ${est:.4f})")

    # Persist the unresolved pairs before resolution, same role as
    # openie_results.json: lets the matcher/threshold be retuned without re-paying.
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "claims_raw.json").write_text(json.dumps(raw_pairs, indent=2))

    print("\n=== Span resolution ===")
    records: list[dict] = []
    dropped: list[dict] = []
    a_exact = a_fuzzy = 0
    b_exact = b_fuzzy = b_split = 0

    for pair in raw_pairs:
        number = pair["section_number"]
        rec = resolve(pair, number, sections[number]["raw"],
                      chunks_by_section.get(number, []), args.threshold)
        if rec is None:
            dropped.append(pair)
            continue
        records.append(rec)
        if rec["section_match_tier"] == "exact":
            a_exact += 1
        else:
            a_fuzzy += 1
        if rec["chunk_match"] == "exact":
            b_exact += 1
        elif rec["chunk_match"] == "fuzzy":
            b_fuzzy += 1
        else:
            b_split += 1

    (save_dir / "claims.json").write_text(json.dumps(records, indent=2))

    extracted = len(raw_pairs)
    n_dropped = len(dropped)
    drop_rate = round(n_dropped / extracted, 4) if extracted else 0.0
    stats = {
        "sections_requested": len(want),
        "sections_processed": len({p["section_number"] for p in raw_pairs}),
        "claims_extracted": extracted,
        "invalid_pairs": invalid_total,
        "claims_resolved": len(records),
        "claims_dropped": n_dropped,
        "drop_rate": drop_rate,
        "section_match": {"exact": a_exact, "fuzzy": a_fuzzy},
        "chunk_match": {"exact": b_exact, "fuzzy": b_fuzzy, "split_across_chunks": b_split},
        "tokens": {"total": total_tokens, "new_this_run": new_tokens},
        "est_cost_usd": round(total_tokens / 1e6 * RATE_PER_MTOK, 4),
        "aborted_on_budget": aborted,
    }
    (save_dir / "claims_stats.json").write_text(json.dumps(stats, indent=2))
    (save_dir / "dropped_pairs.json").write_text(json.dumps(dropped, indent=2))

    run_config = {
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "fuzzy_threshold": args.threshold,
        "sections": want,
        "extraction_unit": "section",
        "max_tokens_budget": args.max_tokens_budget,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reproduce": f"python scripts/extraction-variants/extract_claims_original.py --sections {args.sections} "
                     f"--threshold {args.threshold}",
        "note": "quotes are resolved deterministically post-hoc; offsets are never "
                "requested from the model. span is chunk-relative [start, end).",
    }
    (save_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("\n=== Done ===")
    for k, v in stats.items():
        print(f"  {k:>18}: {v}")
    if aborted:
        print("\n  NOTE: run aborted on token budget; outputs contain partial results.")
    print(f"\n  exported -> {save_dir}")


if __name__ == "__main__":
    main()
