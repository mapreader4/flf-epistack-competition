"""Role tagging (guide §3.1) — the leakage holdout's first line.

Every claim gets role ∈ {evidentiary, conclusory, procedural}:
  - evidentiary : input material the reasoning operates on — observations, data,
                  cited findings, and the DEBATERS' (Rootclaim / challenger) arguments
                  and estimates. Fed into hypothesis scoring.
  - conclusory  : the JUDGE'S OWN outputs about the origin question — his findings,
                  his assigned priors / Bayes factors / posteriors, his weighings, his
                  verdict. NEVER fed into scoring; these populate the answer key.
  - procedural  : housekeeping with no load on the origin question — debate format,
                  "about this document", dates, glossary, excluded-material bookkeeping,
                  and the judge's findings about the DEBATE PROCESS (fairness, good faith).

The distinction that makes or breaks the leakage guard: a number that is *measured or
reported* (from data / a study / a witness, or the debaters' own calculation) is
evidentiary; a number that is the *judge's assigned* estimate / Bayes factor /
probability about Z vs LL is conclusory. The document contains its own verdict — if the
judge's own weights leak in as evidence, DF-QuAD "recovers his ranking" by copying his
numbers (guide §3.1).

Two independent passes (primary v1, secondary v2 = paraphrased prompt, same labels)
give an inter-rater signal; disagreements are surfaced first for the human 30-claim
spot-check (guide §5 acceptance: a second model instance counts, disagreements resolved
by hand). All raw model output is logged as `llm_call` events (guide §3.2).

Writes: artifacts/epistemic/events.jsonl (append-only), claims_tagged.jsonl (folded
snapshot), role_cache.json (raw-output cache → replay without re-calling), and
role_review_sample.json (the 30 for hand review).

    .venv/bin/python scripts/tag_roles.py            # full run, 590 x 2 passes
    .venv/bin/python scripts/tag_roles.py --limit 20 # smoke test on first 20 claims
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_store import (  # noqa: E402
    Event, event_id, append_events, read_events, fold, resolve_roles, next_seq, utc_now_iso,
)

ROOT = Path(__file__).resolve().parent.parent
CLAIMS_PATH = ROOT / "artifacts" / "claims" / "claims.json"
EVENTS_PATH = ROOT / "artifacts" / "epistemic" / "events.jsonl"
FOLDED_PATH = ROOT / "artifacts" / "epistemic" / "claims_tagged.jsonl"
CACHE_PATH = ROOT / "artifacts" / "epistemic" / "role_cache.json"
REVIEW_PATH = ROOT / "artifacts" / "epistemic" / "role_review_sample.json"

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
GPTOSS_MODEL = "openai/gpt-oss-120b"                       # primary labeler (focus)
LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"    # cross-model check
DEFAULT_MODEL = LLAMA_MODEL                                # (kept for back-compat)
BATCH_SIZE = 10
ROLES = {"evidentiary", "conclusory", "procedural"}

# ---------------------------------------------------------------------------
# Prompts. Same label set; two independent cognitive routes.
# ---------------------------------------------------------------------------

_RUBRIC = """\
This corpus is ONE judge's (Eric Stansifer's) written decision adjudicating a debate on
the origin of SARS-CoV-2: natural zoonotic spillover (Z) vs. gain-of-function lab leak
(LL). Rootclaim (Saar Wilf) argued LL; the challenger (Peter Miller) argued Z.

Tag each claim with exactly one role:

evidentiary — INPUT MATERIAL the judge reasons over. Observations, measurements, data,
  timelines, genomic facts, cited studies/findings, and the DEBATERS' own arguments and
  numeric estimates (e.g. "Rootclaim calculates ...", "Pekar et al. argue two
  introductions", "the index case was a resident of the west market", "SARS-CoV-2 binds
  human ACE2 with high affinity"). If a debater or a study asserts it, it is evidentiary
  even when numeric.

conclusory — the JUDGE'S OWN output ABOUT THE ORIGIN QUESTION. His findings, credibility
  determinations, and — critically — his OWN assigned priors, Bayes factors, log-odds,
  and posteriors, and his verdict/ranking (e.g. "the judge finds Rootclaim's model
  unrealistic", "the prior probability of a pandemic from HSM is 1/32000", "the Bayes
  factor for HSM is 1/10000", "P(LL|O)=0.07529%", "physical evidence points strongly to
  zoonotic spillover"). Rule of thumb: if the sentence reports what THE JUDGE
  estimates / assigns / calculates / concludes / finds about Z vs LL, it is conclusory —
  NOT evidentiary — even though it contains a number. (Only the JUDGE's numbers are
  conclusory; the debaters' numbers are evidentiary.)

procedural — housekeeping with no bearing on the origin question: debate format and
  rules, "about this document", moderation, dates, glossary, excluded-material
  bookkeeping, AND the judge's findings about the DEBATE PROCESS itself (e.g. "the debate
  was fair", "both parties argued in good faith").

Return ONLY a JSON array, one object per claim, in input order:
[{"claim_id": "...", "role": "evidentiary|conclusory|procedural", "confidence": 0.0-1.0,
  "justification": "<=12 words"}]
No prose outside the JSON."""

_RUBRIC_V2 = """\
You are separating a judge's written decision into three kinds of statements. The
document is Eric Stansifer's adjudication of SARS-CoV-2 origin: zoonosis (Z) vs lab leak
(LL); the debaters are Rootclaim/Saar Wilf (LL) and the challenger Peter Miller (Z).

For each claim ask: WHOSE epistemic content is this, and does it bear on the origin?

  -> If it is the JUDGE'S OWN finding, weighing, assigned probability/Bayes-factor/prior/
     posterior, or verdict about Z vs LL  => "conclusory".
     (His assigned numbers count here even though they look like data. His numbers are
     conclusions; the debaters' numbers are not.)
  -> If it is something the judge WEIGHS — an observation, a measurement, a study result,
     a genomic/epidemiological fact, or an argument/estimate made by Rootclaim or the
     challenger  => "evidentiary".
  -> If it is case housekeeping (format, dates, document navigation, glossary, excluded
     material) or a finding about the DEBATE PROCESS rather than the origin  => "procedural".

Output ONLY a JSON array in input order:
[{"claim_id":"...","role":"evidentiary|conclusory|procedural","confidence":0.0-1.0,
  "justification":"<=12 words"}]"""

PROMPTS = {
    "role-v1": _RUBRIC,
    "role-v2": _RUBRIC_V2,
}


def build_client() -> OpenAI:
    load_dotenv()
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key, max_retries=3, timeout=300)


def batch_user_msg(batch: list[dict]) -> str:
    lines = ["Claims to tag:"]
    for c in batch:
        lines.append(json.dumps({"claim_id": c["claim_id"],
                                 "section": c["section_number"],
                                 "text": c["claim_text"]}, ensure_ascii=False))
    return "\n".join(lines)


def parse_roles(text: str) -> dict[str, dict]:
    """Index roles by claim_id. Tolerant of truncated arrays — a reasoning model can hit
    max_tokens mid-JSON; if the closing ] is missing, salvage by closing at the last
    complete object so claims that DID come back are still recovered."""
    if not text:
        return {}
    start = text.find("[")
    if start == -1:
        return {}
    frag = text[start:]
    arr = None
    m = re.search(r"\[.*\]", frag, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            arr = None
    if arr is None:                              # truncated — close at last complete object
        last = frag.rfind("}")
        if last != -1:
            try:
                arr = json.loads(frag[:last + 1] + "]")
            except json.JSONDecodeError:
                arr = None
    if arr is None:
        return {}
    out = {}
    for o in arr:
        cid = o.get("claim_id")
        role = (o.get("role") or "").strip().lower()
        if cid and role in ROLES:
            out[cid] = {"role": role,
                        "confidence": float(o.get("confidence", 0.5)),
                        "justification": (o.get("justification") or "")[:120]}
    return out


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=0))


def cache_key(prompt_version: str, model: str, batch: list[dict]) -> str:
    return f"{prompt_version}|{model}|" + "|".join(c["claim_id"] for c in batch)


def already_done(events: list[Event]) -> tuple[set, set, set]:
    """(claim_added ids, GOOD non-UNPARSED tag keys, UNPARSED-only keys to retry).
    Keyed by (claim_id, model, prompt_version) so a new model's pass is added alongside
    prior ones. Only GOOD counts as done; an UNPARSED-only key is retried."""
    added, good, unparsed = set(), set(), set()
    for e in events:
        if e.event_type == "claim_added":
            added.add(e.payload["claim_id"])
        elif e.event_type == "claim_tagged":
            key = (e.payload["claim_id"], e.model, e.payload.get("prompt_version"))
            (unparsed if e.payload["role"] == "UNPARSED" else good).add(key)
    return added, good, unparsed - good


def vote(c: dict, model: str, prompt: str | None = None, pass_: str | None = None):
    """Read one labeler's role for a claim. Prefers a real role over a stale UNPARSED
    tag (a retried batch leaves both an UNPARSED and a good tag in the append-only log)."""
    fallback = None
    for t in c.get("role_tags", []):
        if t.get("model") == model and (prompt is None or t.get("prompt_version") == prompt) \
                and (pass_ is None or t.get("labeler_pass") == pass_):
            if t["role"] != "UNPARSED":
                return t["role"]
            fallback = fallback or t["role"]
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=str(CLAIMS_PATH),
                    help="claims file to tag (default: the original claims.json)")
    ap.add_argument("--primary-model", default=GPTOSS_MODEL,
                    help="operative labeler (default gpt-oss-120b)")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="gpt-oss is a reasoning model — needs headroom to avoid truncation")
    ap.add_argument("--limit", type=int, default=0, help="tag only first N claims (smoke test)")
    ap.add_argument("--review-n", type=int, default=30, help="size of the human review sample")
    args = ap.parse_args()

    claims = json.load(open(args.claims))
    if args.limit:
        claims = claims[: args.limit]

    # This invocation runs ONE pass: the primary model on the v1 rubric. Prior passes
    # (e.g. the Llama v1/v2 already in the log) stay put and become cross-model checks.
    passes = [{"model": args.primary_model, "prompt": "role-v1",
               "pass": "primary", "max_tokens": args.max_tokens}]
    print(f"Claims: {len(claims)} | primary labeler: {args.primary_model} @ Together "
          f"| batch {BATCH_SIZE}")

    client = build_client()
    cache = load_cache()

    seq = next_seq(EVENTS_PATH)
    added_ids, tagged_set, retry_set = already_done(read_events(EVENTS_PATH))
    pending: list[Event] = []

    # 1) seed claim_added for any claim not yet in the log (append-only)
    for c in claims:
        if c["claim_id"] not in added_ids:
            pending.append(Event(event_id(seq), "claim_added", {
                "claim_id": c["claim_id"], "claim_text": c["claim_text"],
                "section_number": c["section_number"], "chunk_id": c["chunk_id"],
                "quote": c["quote"], "span": c["span"],
                "is_appendix": c["is_appendix"], "is_excluded": c["is_excluded"],
            }, cause="ingest", ts=utc_now_iso()))
            seq += 1
            added_ids.add(c["claim_id"])

    # 2) labeling pass(es), batched, cached; idempotent by (claim, model, prompt)
    n_calls = 0
    for ps in passes:
        model, prompt_version, labeler_pass = ps["model"], ps["prompt"], ps["pass"]
        rubric = PROMPTS[prompt_version]
        for i in range(0, len(claims), BATCH_SIZE):
            batch = claims[i: i + BATCH_SIZE]
            keys = [(c["claim_id"], model, prompt_version) for c in batch]
            if all(k in tagged_set for k in keys):
                continue
            ck = cache_key(prompt_version, model, batch)
            if any(k in retry_set for k in keys):
                cache.pop(ck, None)          # previously truncated — force a fresh call
            raw = cache.get(ck)
            if raw is None:
                resp = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=ps["max_tokens"],
                    messages=[{"role": "system", "content": rubric},
                              {"role": "user", "content": batch_user_msg(batch)}])
                raw = resp.choices[0].message.content or ""
                cache[ck] = raw
                n_calls += 1
                if n_calls % 10 == 0:
                    save_cache(cache)
                    print(f"  ... {n_calls} live calls")
            parsed = parse_roles(raw)

            call_ev = Event(event_id(seq), "llm_call", {
                "prompt_version": prompt_version, "labeler_pass": labeler_pass,
                "model": model, "batch": [c["claim_id"] for c in batch]},
                cause="ingest", model=model, model_raw_output=raw, ts=utc_now_iso())
            call_eid = call_ev.event_id
            pending.append(call_ev); seq += 1

            for c in batch:
                if (c["claim_id"], model, prompt_version) in tagged_set:
                    continue
                r = parsed.get(c["claim_id"]) or {
                    "role": "UNPARSED", "confidence": 0.0,
                    "justification": "model omitted this claim — needs review"}
                pending.append(Event(event_id(seq), "claim_tagged", {
                    "claim_id": c["claim_id"], "role": r["role"],
                    "prompt_version": prompt_version, "labeler_pass": labeler_pass,
                    "confidence": r["confidence"], "justification": r["justification"]},
                    cause=call_eid, model=model, ts=utc_now_iso()))
                seq += 1
                tagged_set.add((c["claim_id"], model, prompt_version))

    save_cache(cache)
    append_events(EVENTS_PATH, pending)
    print(f"Appended {len(pending)} events ({n_calls} live LLM calls; rest cached).")

    # 3) fold, resolve operative role from the primary model, report
    from collections import Counter
    state = fold(read_events(EVENTS_PATH))
    resolve_roles(state, primary_model=args.primary_model)
    tagged = {cid: c for cid, c in state["claims"].items() if c.get("role")}
    dist = Counter(c["role"] for c in tagged.values())

    # cross-model agreement: primary (gpt-oss v1) vs Llama primary (v1, same rubric)
    xpairs = [(vote(c, args.primary_model, prompt="role-v1"),
               vote(c, LLAMA_MODEL, pass_="primary")) for c in tagged.values()]
    xpairs = [(a, b) for a, b in xpairs if a and b]
    xagree = sum(1 for a, b in xpairs if a == b)
    print(f"\nOperative role distribution ({args.primary_model}): {dict(dist)}")
    if xpairs:
        print(f"Cross-model agreement (gpt-oss vs Llama, same rubric): "
              f"{xagree}/{len(xpairs)} = {100*xagree/len(xpairs):.1f}%")

    with FOLDED_PATH.open("w") as f:
        for c in tagged.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 4) human review sample — cross-model disagreement is the strongest leakage signal
    def priority(c):
        prim = vote(c, args.primary_model, prompt="role-v1")
        llp = vote(c, LLAMA_MODEL, pass_="primary")
        lls = vote(c, LLAMA_MODEL, pass_="secondary")
        role, sec = c.get("role"), str(c.get("section_number", ""))
        text = c.get("claim_text", "").lower()
        numericish = any(t in text for t in ["bayes", "prior", "probability", "odds",
                                             "1/", "factor", "estimate", "%"])
        conf = c.get("role_confidence") if c.get("role_confidence") is not None else 1.0
        score = 0.0
        if "UNPARSED" in (prim, llp, lls): score += 100
        if prim and llp and prim != llp: score += 60      # cross-model disagreement
        if llp and lls and llp != lls: score += 25        # prompt-robustness disagreement
        if role == "conclusory" and not sec.startswith("7"): score += 20
        if role == "conclusory": score += 8
        if numericish: score += 6
        score += (1.0 - conf) * 10
        return score

    ranked = sorted(tagged.values(), key=priority, reverse=True)
    sample = ranked[: args.review_n]
    review = [{
        "claim_id": c["claim_id"], "section": c.get("section_number"),
        "role_operative": c.get("role"),
        "gptoss_v1": vote(c, args.primary_model, prompt="role-v1"),
        "llama_v1": vote(c, LLAMA_MODEL, pass_="primary"),
        "llama_v2": vote(c, LLAMA_MODEL, pass_="secondary"),
        "cross_model_agree": vote(c, args.primary_model, prompt="role-v1") == vote(c, LLAMA_MODEL, pass_="primary"),
        "confidence": c.get("role_confidence"),
        "justification": c.get("role_justification"),
        "text": c.get("claim_text"),
    } for c in sample]
    REVIEW_PATH.write_text(json.dumps(review, ensure_ascii=False, indent=2))
    n_xdis = sum(1 for r in review if not r["cross_model_agree"])
    print(f"\nReview sample: {len(review)} claims ({n_xdis} cross-model disagreements) "
          f"-> {REVIEW_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
