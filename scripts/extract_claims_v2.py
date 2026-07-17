"""Sentence-level claim extraction v2 (multi-stage) for the epistemic layer.

This is a from-scratch redesign of scripts/extract_claims.py (v1). v1 asked one
LLM call to atomize + decontextualize + quote-locate over a whole section's raw
text at once, which produced decontextualization errors (wrong pronoun/reference
resolution) and atomization errors (merged or dropped atomic claims). v2 instead
works ONE SENTENCE AT A TIME through a staged pipeline, so each model call does
exactly one narrow job:

  Stage 1  mention detection      (1 call/sentence, bare sentence, no context)
  Stage 2a UNNAMED resolution     (pronoun/bridging refs -> a registered entity)
  Stage 2b INTERNAL_REF resolution(cross-doc refs -> a section number)
  Stage 3a decontextualize        (mechanical substitution of resolved refs)
  Stage 3b atomize                (split into single-assertion claims)

Provenance (quote + chunk-relative span) is resolved DETERMINISTICALLY after the
fact by the same tiered exact-then-fuzzy matcher v1 uses (span_match.locate), never
requested from the model. Because the extraction unit is now a single sentence,
tier-A span matching should land "exact" almost every time.

The "registry" here (NamedEntityRegistry) is deliberately NOT a general
coreference clusterer: two NAMED mentions merge ONLY when their surface text is
identical after casefold + whitespace-collapse. It exists to give Stage 2a a
short, auditable candidate list, not to do semantic entity linking.

The main loop is a single SEQUENTIAL pass in document order. This is load-bearing:
the registry only grows correctly, and Stage 2a's "same-section vs. cross-section,
recency-weighted" candidate ranking only makes sense, if sentences are visited in
order. This rules out parallelizing across sections.

Outputs go under artifacts/claims_v2/ and outputs/claims_v2/. v1's outputs
(artifacts/claims/, outputs/claims/) are never touched.

Usage:
    python scripts/extract_claims_v2.py                       # curated 12-section sample
    python scripts/extract_claims_v2.py --sections 7,7.1      # explicit subset
    python scripts/extract_claims_v2.py --only-stage 1 --limit 5   # eyeball Stage 1 prompts
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling modules

# Reuse verbatim -- do NOT reimplement these.
from chunk_decision import split_sentences, PDF  # noqa: E402
from extract_claims import recover_sections, build_client  # noqa: E402
from span_match import locate  # noqa: E402

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = os.getenv("TOGETHER_MODEL_V2", "openai/gpt-oss-120b")

# Per-stage prompt versions -- independently retunable, and part of each stage's
# cache key so a prompt tweak invalidates only that stage.
PROMPT_VERSION_S1 = "s1v1"
PROMPT_VERSION_S2 = "s2v1"          # UNNAMED resolution
PROMPT_VERSION_S2_REF = "s2refv1"   # INTERNAL_REF resolution
PROMPT_VERSION_S3A = "s3av1"        # decontextualize
PROMPT_VERSION_S3B = "s3bv1"        # atomize

# Stage 2a candidate-list escalation, decided by the PIPELINE not the model. The
# model only ever sees the candidates handed to it, so it cannot tell "referent
# truly absent" from "referent absent from this slice" -- so on "none" the
# pipeline re-ranks with a bigger list and asks again. Final tier (below) shows
# the whole eligible registry.
CANDIDATE_TIERS = [5, 20]

# Same curated 12-section sample as v1 (see extract_claims.DEFAULT_SECTIONS).
DEFAULT_SECTIONS = [
    "7", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6",
    "5.4", "5.4.1", "5.4.2", "5.4.3", "5.4.4",
]

# Rough blended (in+out) Together price per 1M tokens for gpt-oss-120b. Used only
# for a friendly running cost estimate; not authoritative.
RATE_PER_MTOK = 0.60

MENTION_TYPES = {"NAMED", "UNNAMED", "INTERNAL_REF", "EXTERNAL_REF"}

# Map internal stage codes (used in cache keys / token accounting) to the
# --only-stage labels the user passes.
STAGE_CODE_TO_LABEL = {"s1": "1", "s2u": "2a", "s2r": "2b", "s3a": "3a", "s3b": "3b"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SentenceRecord:
    global_idx: int          # monotonic across the whole document
    section_number: str
    section_title: str
    local_idx: int           # 0-based within its section
    text: str                # verbatim raw sentence; becomes the claim's quote


@dataclass
class Mention:
    mention_idx: int
    text: str
    type: str                # one of MENTION_TYPES


@dataclass
class ResolvedMention:
    mention_idx: int
    text: str
    type: str
    resolved_to: str | None
    resolution_method: str   # identity | algorithmic | llm | unresolved | external


@dataclass
class RegistryEntry:
    entity_id: str                 # "ent-00001" (or "ent-{doc_id}-00001"), assigned on first sighting
    normalized_key: str            # casefold + whitespace-collapsed surface text
    canonical_text: str            # surface text as first seen
    first_seen: tuple              # (section_number, global_idx)
    last_seen_section: str
    last_seen_global_idx: int
    last_seen_mention_idx: int     # position within its last-seen sentence (reading order)
    mention_count: int


def normalize_key(s: str) -> str:
    """Merge key for NAMED mentions: casefold + collapse whitespace.

    Same normalization idea as span_match._norm_needle (casefold instead of
    lower, no ligature clean here) -- reimplemented, not imported, since that
    helper is private to span_match.
    """
    return re.sub(r"\s+", " ", s).strip().casefold()


class NamedEntityRegistry:
    """A list of RegistryEntry plus a normalized_key -> entity_id index.

    Deliberately simple: two NAMED mentions merge ONLY on exact normalized-text
    match, never on semantic similarity. Not a general coreference clusterer.
    """

    def __init__(self, doc_id: str = "") -> None:
        # doc_id namespaces every entity_id (e.g. "ent-{doc_id}-00001") so that
        # registries from separate documents never collide on bare numbering if
        # they're later merged into one shared cross-document graph. Cross-document
        # entity MATCHING (e.g. via embeddings) is a later, separate structure-layer
        # step -- this is just collision-avoidance for identifiers.
        self.doc_id = doc_id
        self.entries: list[RegistryEntry] = []
        self.key_to_id: dict[str, str] = {}
        self.id_to_entry: dict[str, RegistryEntry] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        prefix = f"ent-{self.doc_id}-" if self.doc_id else "ent-"
        return f"{prefix}{self._counter:05d}"

    def register(self, surface_text: str, section_number: str, global_idx: int,
                 mention_idx: int) -> str:
        """Register (or bump) a NAMED mention. Returns its entity_id.

        `mention_idx` is the mention's reading-order position within its sentence;
        stored as last_seen_mention_idx so same-sentence anaphora (a pronoun that
        points at an entity earlier in the SAME sentence) can be resolved while
        cataphora (entity appears later) stays excluded. See eligible_candidates.
        """
        key = normalize_key(surface_text)
        if key in self.key_to_id:
            eid = self.key_to_id[key]
            e = self.id_to_entry[eid]
            e.last_seen_section = section_number
            e.last_seen_global_idx = global_idx
            e.last_seen_mention_idx = mention_idx
            e.mention_count += 1
            return eid
        eid = self._next_id()
        e = RegistryEntry(
            entity_id=eid,
            normalized_key=key,
            canonical_text=surface_text,
            first_seen=(section_number, global_idx),
            last_seen_section=section_number,
            last_seen_global_idx=global_idx,
            last_seen_mention_idx=mention_idx,
            mention_count=1,
        )
        self.entries.append(e)
        self.key_to_id[key] = eid
        self.id_to_entry[eid] = e
        return eid


# ---------------------------------------------------------------------------
# Sentence indexing
# ---------------------------------------------------------------------------

def index_sentences(sections: dict) -> list[SentenceRecord]:
    """Flatten recover_sections() output into per-sentence records in document order.

    Iterate the dict in INSERTION order (recover_sections walks locate_headings'
    output, which is already document order) -- do NOT re-sort by section-number
    string, since "7.10" < "7.2" lexicographically would break document order.
    """
    records: list[SentenceRecord] = []
    global_idx = 0
    for number, sec in sections.items():
        sentences = split_sentences(sec["raw"])
        for local_idx, text in enumerate(sentences):
            records.append(SentenceRecord(
                global_idx=global_idx,
                section_number=number,
                section_title=sec["title"],
                local_idx=local_idx,
                text=text,
            ))
            global_idx += 1
    return records


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise text-annotation assistant for an argument-mining and "
    "provenance-indexing pipeline over a source text. You perform narrow, "
    "mechanical annotation tasks (identifying references, resolving them, "
    "rewriting, splitting into atomic claims). You do not evaluate, endorse, "
    "correct, or add information of your own; you only annotate what the text "
    "itself states. Always reply with ONLY the requested JSON, no prose before or "
    "after."
)


def call_llm_v2(client: OpenAI, model: str, messages: list[dict],
                use_reasoning_effort: bool, reasoning_effort: str) -> dict:
    """One chat completion. Includes reasoning_effort only if preflight accepted it."""
    kwargs = dict(model=model, temperature=0, messages=messages)
    if use_reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kwargs)
    u = resp.usage
    return {
        "content": resp.choices[0].message.content or "",
        "usage": {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        },
    }


def preflight_v2(client: OpenAI, model: str) -> bool:
    """Confirm connectivity, then probe whether Together accepts reasoning_effort.

    First call mirrors v1's preflight() exactly (bare call, no extra params).
    Second call retries with reasoning_effort="low". Returns True iff that second
    call succeeded, so call_llm_v2 knows whether to include the param thereafter.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
    )
    msg = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    print(f"  preflight LLM ok -> {msg[:60]!r} "
          f"({getattr(u, 'completion_tokens', '?')} completion tokens)")

    try:
        client.chat.completions.create(
            model=model,
            temperature=0,
            reasoning_effort="low",
            messages=[{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}],
        )
        print("  preflight reasoning_effort=low ACCEPTED -> will pass it on every call")
        return True
    except Exception as exc:  # noqa: BLE001 -- any rejection means "don't send it"
        print(f"  preflight reasoning_effort=low REJECTED ({type(exc).__name__}) "
              f"-> omitting it from calls")
        return False


def _try_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_json_response(text: str):
    """Tolerantly pull a JSON object/array out of a model response."""
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
    return data


# ---------------------------------------------------------------------------
# Runtime context + cached LLM caller
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    client: OpenAI
    model: str
    use_re: bool
    reasoning_effort: str
    cache: dict
    cache_path: Path
    no_cache: bool
    stage_tokens: dict           # stage_code -> {"total": int, "new": int}
    debug_stage: str | None = None   # a --only-stage label, or None


def llm_call(ctx: Ctx, key: str, messages: list[dict], stage: str) -> tuple[dict, bool]:
    """Cached single LLM call. `stage` is a stage code (s1/s2u/s2r/s3a/s3b).

    Only calls that actually hit the LLM get a cache entry; the cache is written
    incrementally to disk after each new call (mirrors v1). Returns (entry, cached).
    """
    if not ctx.no_cache and key in ctx.cache:
        entry = ctx.cache[key]
        ctx.stage_tokens[stage]["total"] += entry["usage"]["total_tokens"]
        cached = True
    else:
        entry = call_llm_v2(ctx.client, ctx.model, messages, ctx.use_re, ctx.reasoning_effort)
        ctx.cache[key] = entry
        ctx.cache_path.write_text(json.dumps(ctx.cache, indent=2))
        ctx.stage_tokens[stage]["total"] += entry["usage"]["total_tokens"]
        ctx.stage_tokens[stage]["new"] += entry["usage"]["total_tokens"]
        cached = False

    if ctx.debug_stage and STAGE_CODE_TO_LABEL.get(stage) == ctx.debug_stage:
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        print("\n" + "-" * 78)
        print(f"[stage {ctx.debug_stage}] {'(cached)' if cached else '(fresh call)'}")
        print("--- PROMPT (user message) ---")
        print(user_msg)
        print("--- RAW MODEL OUTPUT ---")
        print(entry["content"])
    return entry, cached


# ---------------------------------------------------------------------------
# Stage 1 -- mention detection
# ---------------------------------------------------------------------------

S1_USER = """\
Identify the referential MENTIONS in the single sentence below -- spans a reader \
would need to resolve, or that point to other evidence.

Mention types:
- NAMED: the phrase's own wording already identifies something specific -- a \
name, title, technical term, or a definite description detailed enough to read \
as referring to one distinct thing by itself (e.g. "the 2019 household survey of \
400 families in rural Ohio", "the Amazon rainforest" \
).
- UNNAMED: a pronoun, demonstrative, or a bare/underspecified definite-description \
reference that presupposes something established elsewhere to be understood -- \
including a plain, undetailed definite phrase with no identifying content of its \
own (e.g. "it", "this finding", "the aforementioned proposal", "the authors", "the \
previous claim", "the committee", "the survey").
- INTERNAL_REF: a pointer to another place in THIS document (e.g. "as discussed \
in Section 5.4", "the previous paragraph", "see above").
- EXTERNAL_REF: a pointer to evidence OUTSIDE this document (e.g. "a 2020 \
published study", "an external report").

Copy each mention's text verbatim from the sentence. Return an empty list if the \
sentence has no mentions worth tagging.

Return ONLY JSON of the exact form:
{{"mentions": [{{"text": "<verbatim span>", "type": "NAMED|UNNAMED|INTERNAL_REF|EXTERNAL_REF"}}]}}

SENTENCE:
\"\"\"{sentence}\"\"\"
"""


def build_s1_messages(sent: SentenceRecord) -> list[dict]:
    # Deliberately NO section title, NO surrounding sentences -- mention detection
    # on one sentence must not pick up entity names belonging to a neighbour.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": S1_USER.format(sentence=sent.text)},
    ]


def parse_mentions(text: str) -> list[Mention]:
    data = parse_json_response(text)
    raw = []
    if isinstance(data, dict) and isinstance(data.get("mentions"), list):
        raw = data["mentions"]
    elif isinstance(data, list):
        raw = data
    mentions: list[Mention] = []
    idx = 0
    for d in raw:
        if not isinstance(d, dict):
            continue
        t = d.get("text")
        ty = d.get("type")
        if isinstance(t, str) and t.strip() and ty in MENTION_TYPES:
            mentions.append(Mention(mention_idx=idx, text=t.strip(), type=ty))
            idx += 1
    return mentions


def stage1_detect_mentions(ctx: Ctx, sent: SentenceRecord) -> list[Mention]:
    key = ck_s1(sent.section_number, sent.local_idx, ctx.model)
    entry, _ = llm_call(ctx, key, build_s1_messages(sent), "s1")
    return parse_mentions(entry["content"])


# ---------------------------------------------------------------------------
# Registry update (no LLM call)
# ---------------------------------------------------------------------------

def register_named_mentions(registry: NamedEntityRegistry, sent: SentenceRecord,
                            mentions: list[Mention]) -> dict[int, str]:
    """Register every NAMED mention; return {mention_idx -> entity_id}."""
    out: dict[int, str] = {}
    for m in mentions:
        if m.type == "NAMED":
            out[m.mention_idx] = registry.register(
                m.text, sent.section_number, sent.global_idx, m.mention_idx)
    return out


# ---------------------------------------------------------------------------
# Stage 2a -- UNNAMED resolution
# ---------------------------------------------------------------------------

def eligible_candidates(registry: NamedEntityRegistry, sent: SentenceRecord,
                        current_mention_idx: int | None = None) -> list[RegistryEntry]:
    """Entities eligible as antecedents for a reference in this sentence.

    An entry is eligible if it was last seen strictly EARLIER in the document
    (cross-sentence case), OR -- when resolving a specific mention and
    current_mention_idx is supplied -- it was last seen in THIS same sentence but
    at a strictly earlier reading-order position (same-sentence anaphora). The
    latter clause is what lets "The proposal ... it was leaked" resolve "it" to
    "the proposal"; cataphora (entity appears LATER in the sentence) stays
    ineligible because its last_seen_mention_idx is not < current_mention_idx.
    """
    out = []
    for e in registry.entries:
        if e.last_seen_global_idx < sent.global_idx:
            out.append(e)
        elif (e.last_seen_global_idx == sent.global_idx
              and current_mention_idx is not None
              and e.last_seen_mention_idx < current_mention_idx):
            out.append(e)
    return out


def rank_candidates(registry: NamedEntityRegistry, sent: SentenceRecord,
                    top_k: int, current_mention_idx: int | None = None
                    ) -> list[tuple[RegistryEntry, float]]:
    """Recency- and section-weighted candidate shortlist for the LLM.

    Exists ONLY to build a short, cheap candidate list -- it must NOT decide the
    resolution algorithmically (no confidence-margin shortcut). The LLM always
    makes the actual call. Same-sentence candidates score at recency-distance 0
    (max recency), which is the intended behavior.
    """
    pool = eligible_candidates(registry, sent, current_mention_idx)
    scored: list[tuple[RegistryEntry, float]] = []
    for e in pool:
        weight = 2.0 if e.last_seen_section == sent.section_number else 1.0
        score = weight / (1 + (sent.global_idx - e.last_seen_global_idx))
        scored.append((e, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]


S2U_USER = """\
A sentence from a source document contains a reference that needs to be resolved \
to the entity it points back to. Choose which candidate entity the reference \
refers to, or "none" if none of them fit.

REFERENCE: "{mention}"
SENTENCE: "{sentence}"

CANDIDATES (each was mentioned earlier in the document):
{candidates}

Return ONLY JSON: {{"choice": <1-based candidate number>}} or {{"choice": "none"}}.
"""


def _format_candidates(ranked: list[tuple[RegistryEntry, float]]) -> str:
    return "\n".join(
        f"{i + 1}. {e.canonical_text}  (last seen in Section {e.last_seen_section})"
        for i, (e, _score) in enumerate(ranked)
    )


def build_s2u_messages(sent: SentenceRecord, mention: Mention,
                       ranked: list[tuple[RegistryEntry, float]]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": S2U_USER.format(
            mention=mention.text, sentence=sent.text, candidates=_format_candidates(ranked))},
    ]


def resolve_unnamed_llm(ctx: Ctx, sent: SentenceRecord, mention: Mention,
                        registry: NamedEntityRegistry) -> ResolvedMention:
    """Resolve one UNNAMED mention via the LLM, escalating the candidate list.

    Signature note: the spec sketches resolve_unnamed_llm(client, model, sent,
    mention, ranked_candidates, ...); because escalation must RE-RANK with a
    bigger top_k, this takes the registry and ranks internally instead of a fixed
    ranked list. client/model live on ctx (so the shared cache is used).

    Tiers: CANDIDATE_TIERS (top-5, top-20) then the full eligible registry. On
    "none" at a tier, escalate to the next larger tier; if "none" persists once
    the list already covers the whole eligible pool, mark unresolved.
    """
    pool_size = len(eligible_candidates(registry, sent, mention.mention_idx))
    if pool_size == 0:
        # No eligible antecedent yet (nothing earlier in the document, and nothing
        # earlier in this sentence) -- nothing to resolve against.
        return ResolvedMention(mention.mention_idx, mention.text, mention.type,
                               resolved_to=None, resolution_method="unresolved")

    tiers = list(CANDIDATE_TIERS) + [None]  # None -> full eligible registry (uncapped)
    for round_idx, top_k in enumerate(tiers):
        effective_k = pool_size if top_k is None else min(top_k, pool_size)
        ranked = rank_candidates(registry, sent, effective_k, mention.mention_idx)
        key = ck_s2u(sent.section_number, sent.local_idx, mention.mention_idx, round_idx, ctx.model)
        entry, _ = llm_call(ctx, key, build_s2u_messages(sent, mention, ranked), "s2u")
        parsed = parse_json_response(entry["content"])
        choice = parsed.get("choice") if isinstance(parsed, dict) else "none"

        if isinstance(choice, bool):
            choice = "none"  # guard: JSON true/false is not a valid choice
        if isinstance(choice, int) and 1 <= choice <= len(ranked):
            picked = ranked[choice - 1][0]
            return ResolvedMention(mention.mention_idx, mention.text, mention.type,
                                   resolved_to=picked.canonical_text, resolution_method="llm")

        # choice == "none" (or unparseable): escalate only if there is a strictly
        # larger candidate list still available.
        if effective_k >= pool_size:
            break  # this tier already showed the whole eligible pool

    return ResolvedMention(mention.mention_idx, mention.text, mention.type,
                           resolved_to=None, resolution_method="unresolved")


# ---------------------------------------------------------------------------
# Stage 2b -- INTERNAL_REF resolution
# ---------------------------------------------------------------------------

_SECTION_REF_RE = re.compile(
    r"(?:section|sections|§|sec\.?)\s*([0-9]+(?:\.[0-9]+)*|[A-E](?:\.[0-9]+)*)",
    re.IGNORECASE,
)


def resolve_internal_ref_algorithmic(mention_text: str, section_index: set[str]) -> str | None:
    """If the ref names an explicit section number that exists, return it; else None."""
    m = _SECTION_REF_RE.search(mention_text)
    if not m:
        return None
    num = m.group(1)
    return num if num in section_index else None


S2R_USER = """\
A sentence from a source document contains an internal cross-reference that does \
NOT name an explicit section number. Choose which section it most likely points \
to, or "none".

REFERENCE: "{mention}"
SENTENCE: "{sentence}"

CANDIDATE SECTIONS:
{candidates}

Return ONLY JSON: {{"choice": "<section number from the list>"}} or {{"choice": "none"}}.
"""


def build_s2r_messages(sent: SentenceRecord, mention_text: str,
                       nearby_sections: list[tuple[str, str]]) -> list[dict]:
    candidates = "\n".join(f"- {num}: {title}" for num, title in nearby_sections)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": S2R_USER.format(
            mention=mention_text, sentence=sent.text, candidates=candidates)},
    ]


def resolve_internal_ref_llm(ctx: Ctx, sent: SentenceRecord, mention: Mention,
                             nearby_sections: list[tuple[str, str]]) -> ResolvedMention:
    """Resolve a vague INTERNAL_REF ("as noted above") to a nearby section via LLM."""
    valid = {num for num, _ in nearby_sections}
    key = ck_s2r(sent.section_number, sent.local_idx, mention.mention_idx, ctx.model)
    entry, _ = llm_call(ctx, key, build_s2r_messages(sent, mention.text, nearby_sections), "s2r")
    parsed = parse_json_response(entry["content"])
    choice = parsed.get("choice") if isinstance(parsed, dict) else None
    if isinstance(choice, str) and choice in valid:
        return ResolvedMention(mention.mention_idx, mention.text, mention.type,
                               resolved_to=choice, resolution_method="llm")
    return ResolvedMention(mention.mention_idx, mention.text, mention.type,
                           resolved_to=None, resolution_method="unresolved")


def nearby_sections_for(sent: SentenceRecord, section_order: list[str],
                        section_titles: dict[str, str]) -> list[tuple[str, str]]:
    """Candidate sections for a vague internal ref: current, parent, prev 2-3 (doc order)."""
    picks: list[str] = []
    cur = sent.section_number
    picks.append(cur)
    parent = ".".join(cur.split(".")[:-1]) if "." in cur else None
    if parent:
        picks.append(parent)
    try:
        idx = section_order.index(cur)
    except ValueError:
        idx = len(section_order)
    for num in section_order[max(0, idx - 3):idx]:
        picks.append(num)
    # Dedup preserving order.
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for num in picks:
        if num in seen:
            continue
        seen.add(num)
        out.append((num, section_titles.get(num, "")))
    return out


# ---------------------------------------------------------------------------
# Stage 2 dispatcher
# ---------------------------------------------------------------------------

def resolve_mentions(ctx: Ctx, sent: SentenceRecord, mentions: list[Mention],
                     registry: NamedEntityRegistry, named_map: dict[int, str],
                     section_index: set[str], section_order: list[str],
                     section_titles: dict[str, str]) -> list[ResolvedMention]:
    """Resolve every mention on a sentence to a ResolvedMention (algorithmic-first)."""
    resolved: list[ResolvedMention] = []
    for m in mentions:
        if m.type == "NAMED":
            eid = named_map.get(m.mention_idx)
            canonical = registry.id_to_entry[eid].canonical_text if eid else m.text
            resolved.append(ResolvedMention(m.mention_idx, m.text, m.type,
                                            resolved_to=canonical, resolution_method="identity"))
        elif m.type == "EXTERNAL_REF":
            resolved.append(ResolvedMention(m.mention_idx, m.text, m.type,
                                            resolved_to=None, resolution_method="external"))
        elif m.type == "UNNAMED":
            resolved.append(resolve_unnamed_llm(ctx, sent, m, registry))
        elif m.type == "INTERNAL_REF":
            algo = resolve_internal_ref_algorithmic(m.text, section_index)
            if algo is not None:
                resolved.append(ResolvedMention(m.mention_idx, m.text, m.type,
                                                resolved_to=algo, resolution_method="algorithmic"))
            else:
                nearby = nearby_sections_for(sent, section_order, section_titles)
                resolved.append(resolve_internal_ref_llm(ctx, sent, m, nearby))
        else:  # unreachable given parse_mentions validation
            resolved.append(ResolvedMention(m.mention_idx, m.text, m.type,
                                            resolved_to=None, resolution_method="unresolved"))
    return resolved


# ---------------------------------------------------------------------------
# Stage 3a -- decontextualize
# ---------------------------------------------------------------------------

S3A_USER = """\
Rewrite the sentence below as a single fluent, standalone sentence by substituting \
in the resolved references provided. Make ONLY these substitutions and whatever \
minimal grammatical adjustment they require. Do not add, remove, or reinterpret \
any other information. Preserve every attribution (who says/argues/concludes what).

ORIGINAL SENTENCE: "{sentence}"

SUBSTITUTIONS (replace each reference with what it resolves to):
{subs}

Return ONLY JSON: {{"decontextualized": "<rewritten standalone sentence>"}}.
"""


def substitution_payload(resolved: list[ResolvedMention]) -> list[dict]:
    """Substitutions to inline: resolved UNNAMED (-> canonical entity) and
    resolved INTERNAL_REF (-> section number). Sorted for a stable cache hash."""
    subs = [
        {"mention_text": r.text, "resolved_to": r.resolved_to}
        for r in resolved
        if r.type in ("UNNAMED", "INTERNAL_REF") and r.resolved_to is not None
    ]
    subs.sort(key=lambda d: (d["mention_text"], d["resolved_to"]))
    return subs


def _format_subs(subs: list[dict]) -> str:
    lines = []
    for s in subs:
        rt = s["resolved_to"]
        # INTERNAL_REF resolves to a bare section number -- label it as such.
        target = f"Section {rt}" if re.fullmatch(r"[0-9A-E]+(?:\.[0-9]+)*", str(rt)) else rt
        lines.append(f'- "{s["mention_text"]}" -> {target}')
    return "\n".join(lines)


def stage3a_decontextualize(ctx: Ctx, sent: SentenceRecord,
                            resolved: list[ResolvedMention]) -> tuple[str, str]:
    """Return (decontextualized_text, method).

    Skip shortcut: if there are no substitutions to inline (zero mentions, all
    mentions resolved to identity/external, or the only non-identity mentions were
    left unresolved), keep the sentence verbatim with no LLM call. Otherwise do a
    mechanical-rewrite call -- referent-finding already happened in Stage 2.
    """
    subs = substitution_payload(resolved)
    if not subs:
        return sent.text, "skipped_no_substitutions"

    # Cache key hashes the ACTUAL substitution payload, not just the prompt version:
    # Stage 2 output can change (e.g. ranking constants retuned) without a prompt
    # bump, and a stale Stage 3a entry would then be wrong.
    payload_md5 = hashlib.md5(json.dumps(subs, sort_keys=True).encode()).hexdigest()
    key = ck_s3a(sent.section_number, sent.local_idx, ctx.model, payload_md5)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": S3A_USER.format(sentence=sent.text, subs=_format_subs(subs))},
    ]
    entry, _ = llm_call(ctx, key, messages, "s3a")
    parsed = parse_json_response(entry["content"])
    if isinstance(parsed, dict) and isinstance(parsed.get("decontextualized"), str) \
            and parsed["decontextualized"].strip():
        return parsed["decontextualized"].strip(), "llm"
    # Fall back to the verbatim sentence if the rewrite was unusable.
    return sent.text, "llm_unparsed"


# ---------------------------------------------------------------------------
# Stage 3b -- atomize
# ---------------------------------------------------------------------------

S3B_USER = """\
Split the sentence below into ATOMIC claims -- each a single, self-contained \
assertion. If the sentence makes only one assertion, return a single-element list. \
Preserve attributions inside each claim (e.g. "The authors argue that ..."). Do not \
add information not present in the sentence, and do not include quotations or \
citations.

SENTENCE: "{sentence}"

Return ONLY JSON: {{"claims": ["<atomic claim>", "..."]}}.
"""


def parse_atomic_claims(text: str) -> list[str]:
    data = parse_json_response(text)
    raw = []
    if isinstance(data, dict) and isinstance(data.get("claims"), list):
        raw = data["claims"]
    elif isinstance(data, list):
        raw = data
    return [c.strip() for c in raw if isinstance(c, str) and c.strip()]


def stage3b_atomize(ctx: Ctx, sent: SentenceRecord, decontextualized_text: str) -> list[str]:
    text_md5 = hashlib.md5(decontextualized_text.encode()).hexdigest()
    key = ck_s3b(sent.section_number, sent.local_idx, ctx.model, text_md5)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": S3B_USER.format(sentence=decontextualized_text)},
    ]
    entry, _ = llm_call(ctx, key, messages, "s3b")
    return parse_atomic_claims(entry["content"])


# ---------------------------------------------------------------------------
# Provenance / span resolution (deterministic, two-tier -- structurally v1's resolve())
# ---------------------------------------------------------------------------

def resolve_span(sentence_text: str, section_number: str, section_raw: str,
                 section_chunks: list[dict], threshold: float) -> dict | None:
    """Place the sentence in its section (tier A), then attribute it to a chunk (tier B).

    Returns quote/span/section_match_*/chunk_match/chunk_id, or None if tier A
    rejects (the sentence -- and all its atomic claims -- is then dropped).
    """
    a = locate(sentence_text, section_raw, threshold)
    if a["tier"] == "reject":
        return None
    matched_text = section_raw[a["start"]:a["end"]]

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

    if best_chunk is None:
        chunk_id = None
        chunk_match = "split_across_chunks"
        span = None
    else:
        chunk_id = best_chunk["chunk_id"]
        chunk_match = best_res["tier"]
        span = [best_res["start"], best_res["end"]]

    return {
        "quote": sentence_text,
        "span": span,
        "chunk_id": chunk_id,
        "section_match_tier": a["tier"],
        "section_match_score": a["score"],
        "chunk_match": chunk_match,
    }


# ---------------------------------------------------------------------------
# Claim assembly
# ---------------------------------------------------------------------------

def make_claim_id(section_number: str, local_sentence_idx: int, atomic_idx: int,
                  doc_id: str = "") -> str:
    """Keyed on the section-LOCAL sentence index so ids don't shift under --sections/--limit.

    doc_id namespaces the id (e.g. "claim-{doc_id}-7.1-s003-00") so claims from
    separate documents can't collide if later merged into one shared graph.
    """
    prefix = f"claim-{doc_id}-" if doc_id else "claim-"
    return f"{prefix}{section_number}-s{local_sentence_idx:03d}-{atomic_idx:02d}"


def build_evidence_refs(resolved: list[ResolvedMention]) -> list[dict]:
    """One entry per EXTERNAL_REF / INTERNAL_REF mention on the sentence."""
    out: list[dict] = []
    for r in resolved:
        if r.type == "EXTERNAL_REF":
            out.append({"type": "external_source", "text": r.text, "resolved_to": r.resolved_to})
        elif r.type == "INTERNAL_REF":
            out.append({"type": "internal_ref", "text": r.text, "resolved_to": r.resolved_to})
    return out


def build_decontextualization_map(resolved: list[ResolvedMention]) -> list[dict]:
    """One entry per UNNAMED / INTERNAL_REF mention that was actually substituted."""
    out: list[dict] = []
    for r in resolved:
        if r.type in ("UNNAMED", "INTERNAL_REF") and r.resolved_to is not None:
            out.append({
                "mention_text": r.text,
                "mention_type": r.type,
                "resolved_to": r.resolved_to,
            })
    return out


# ---------------------------------------------------------------------------
# Cache-key builders (one outputs/claims_v2/llm_cache.json dict, five key shapes)
# ---------------------------------------------------------------------------

def ck_s1(section: str, local_idx: int, model: str) -> str:
    return f"s1|{section}|{local_idx}|{PROMPT_VERSION_S1}|{model}"


def ck_s2u(section: str, local_idx: int, mention_idx: int, round_idx: int, model: str) -> str:
    return f"s2u|{section}|{local_idx}|{mention_idx}|{round_idx}|{PROMPT_VERSION_S2}|{model}"


def ck_s2r(section: str, local_idx: int, mention_idx: int, model: str) -> str:
    return f"s2r|{section}|{local_idx}|{mention_idx}|{PROMPT_VERSION_S2_REF}|{model}"


def ck_s3a(section: str, local_idx: int, model: str, payload_md5: str) -> str:
    return f"s3a|{section}|{local_idx}|{PROMPT_VERSION_S3A}|{model}|{payload_md5}"


def ck_s3b(section: str, local_idx: int, model: str, text_md5: str) -> str:
    return f"s3b|{section}|{local_idx}|{PROMPT_VERSION_S3B}|{model}|{text_md5}"


# ---------------------------------------------------------------------------
# Per-sentence pipeline
# ---------------------------------------------------------------------------

@dataclass
class SentenceArtifacts:
    mentions: list[Mention] = field(default_factory=list)
    resolved: list[ResolvedMention] = field(default_factory=list)
    decontextualized_text: str = ""
    decontext_method: str = ""
    atomic_claims: list[str] = field(default_factory=list)


def process_sentence(ctx: Ctx, sent: SentenceRecord, registry: NamedEntityRegistry,
                     section_index: set[str], section_order: list[str],
                     section_titles: dict[str, str],
                     stop_after: str | None = None) -> SentenceArtifacts:
    """Run the staged pipeline over one sentence.

    `stop_after` (a --only-stage label) short-circuits after the named stage --
    used by debug mode so no downstream work happens. registry is mutated here
    (NAMED mentions get registered), which is why the caller MUST visit sentences
    in document order.
    """
    art = SentenceArtifacts()

    # Stage 1
    art.mentions = stage1_detect_mentions(ctx, sent)
    named_map = register_named_mentions(registry, sent, art.mentions)
    if stop_after == "1":
        return art

    # Stage 2 (2a for UNNAMED, 2b for INTERNAL_REF; NAMED=identity, EXTERNAL=external)
    art.resolved = resolve_mentions(ctx, sent, art.mentions, registry, named_map,
                                    section_index, section_order, section_titles)
    if stop_after in ("2a", "2b"):
        return art

    # Stage 3a
    art.decontextualized_text, art.decontext_method = stage3a_decontextualize(ctx, sent, art.resolved)
    if stop_after == "3a":
        return art

    # Stage 3b
    art.atomic_claims = stage3b_atomize(ctx, sent, art.decontextualized_text)
    return art


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"],
                    help="only sent if preflight confirms Together accepts it for this model")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="min similarity (0-1) to accept a fuzzy span match")
    ap.add_argument("--sections", default=",".join(DEFAULT_SECTIONS),
                    help="comma-separated section numbers (default: curated 12-section sample)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N sentences (document order) for a cheap smoke test")
    ap.add_argument("--save-dir", default=str(ROOT / "artifacts" / "claims_v2"))
    ap.add_argument("--cache", default=str(ROOT / "outputs" / "claims_v2" / "llm_cache.json"))
    ap.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses")
    ap.add_argument("--max-tokens-budget", type=int, default=200_000,
                    help="abort (saving progress) once this many NEW tokens are spent")
    ap.add_argument("--only-stage", choices=["1", "2a", "2b", "3a", "3b"], default=None,
                    help="debug: run only this stage's call over the selected sentences, "
                         "print raw prompt+output, then exit (no assembly, no output files)")
    ap.add_argument("--doc-id", default=PDF.stem,
                    help="namespace for entity_id/claim_id (e.g. 'ent-{doc_id}-00001'), so "
                         "registries/claims from separate documents don't collide if later "
                         "merged into one shared cross-document graph (default: source PDF stem)")
    args = ap.parse_args()

    want = [s.strip() for s in args.sections.split(",") if s.strip()]
    save_dir = Path(args.save_dir)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Doc ID:     {args.doc_id}")
    print(f"Model:      {args.model} @ Together")
    print(f"Reasoning:  {args.reasoning_effort} (pending preflight)")
    print(f"Threshold:  fuzzy >= {args.threshold}")
    print(f"Sections:   {len(want)} -> {', '.join(want)}")
    print(f"Limit:      {args.limit if args.limit is not None else 'none'}")
    print(f"Save dir:   {save_dir}")
    print(f"Budget:     {args.max_tokens_budget:,} new tokens")
    if args.only_stage:
        print(f"DEBUG:      --only-stage {args.only_stage} (no output files will be written)")
    print()

    client = build_client()
    use_re = preflight_v2(client, args.model)

    print("\n=== Recovering per-section raw text ===")
    sections = recover_sections()
    print(f"  recovered {len(sections)} sections from the PDF")

    all_sentences = index_sentences(sections)
    print(f"  indexed {len(all_sentences)} sentences across the whole document")

    # Document-order section list + titles (for internal-ref candidate building).
    section_order: list[str] = []
    for s in all_sentences:
        if not section_order or section_order[-1] != s.section_number:
            if s.section_number not in section_order:
                section_order.append(s.section_number)
    sections_list = json.loads((ROOT / "data" / "sections.json").read_text())
    section_index = {s["number"] for s in sections_list}
    section_titles = {s["number"]: s["title"] for s in sections_list}
    for num, sec in sections.items():  # backfill any recovered titles not in sections.json
        section_titles.setdefault(num, sec.get("title", ""))

    chunks = json.loads((ROOT / "data" / "chunks.json").read_text())
    chunks_by_section: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_section.setdefault(c["section_number"], []).append(c)

    # Select the sentences to process: filter to requested sections (document
    # order preserved), then apply --limit to the first N of those.
    want_set = set(want)
    selected = [s for s in all_sentences if s.section_number in want_set]
    if args.limit is not None:
        selected = selected[:args.limit]
    print(f"  selected {len(selected)} sentences to process "
          f"(sections {', '.join(want)}{f'; first {args.limit}' if args.limit else ''})")

    cache: dict = {}
    if cache_path.exists() and not args.no_cache:
        cache = json.loads(cache_path.read_text())

    stage_tokens = {s: {"total": 0, "new": 0} for s in ("s1", "s2u", "s2r", "s3a", "s3b")}
    ctx = Ctx(client=client, model=args.model, use_re=use_re,
              reasoning_effort=args.reasoning_effort, cache=cache, cache_path=cache_path,
              no_cache=args.no_cache, stage_tokens=stage_tokens, debug_stage=args.only_stage)

    registry = NamedEntityRegistry(doc_id=args.doc_id)

    # ------------------------------------------------------------------
    # Debug mode: --only-stage
    # ------------------------------------------------------------------
    if args.only_stage:
        print(f"\n=== DEBUG --only-stage {args.only_stage} "
              f"({len(selected)} sentences) ===")
        for sent in selected:
            print("\n" + "=" * 78)
            print(f"SENTENCE g{sent.global_idx} [{sent.section_number} #{sent.local_idx}]: "
                  f"{sent.text}")
            # process_sentence prints the target stage's prompt+output via llm_call,
            # and stops right after that stage -- no assembly, no downstream calls.
            process_sentence(ctx, sent, registry, section_index, section_order,
                             section_titles, stop_after=args.only_stage)
        print("\n\n=== DEBUG done (no output files written) ===")
        return

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    print("\n=== Pipeline (sentence by sentence, document order) ===")
    records: list[dict] = []
    trace: list[dict] = []
    dropped: list[dict] = []

    mentions_by_type = {t: 0 for t in ("NAMED", "UNNAMED", "INTERNAL_REF", "EXTERNAL_REF")}
    s2a_counts = {"llm_resolved": 0, "unresolved": 0}
    s2b_counts = {"algorithmic": 0, "llm": 0, "unresolved": 0}
    s3a_counts = {"skipped": 0, "llm": 0, "llm_unparsed": 0}
    section_match = {"exact": 0, "fuzzy": 0}
    chunk_match = {"exact": 0, "fuzzy": 0, "split_across_chunks": 0}
    atomic_total = 0
    sentences_with_claims = 0
    aborted = False

    for sent in selected:
        new_tokens = sum(v["new"] for v in stage_tokens.values())
        if new_tokens >= args.max_tokens_budget:
            print(f"  BUDGET REACHED ({new_tokens:,} new tokens) -- aborting, saving progress")
            aborted = True
            break

        art = process_sentence(ctx, sent, registry, section_index, section_order, section_titles)

        for m in art.mentions:
            mentions_by_type[m.type] += 1
        for r in art.resolved:
            if r.type == "UNNAMED":
                s2a_counts["llm_resolved" if r.resolution_method == "llm" else "unresolved"] += 1
            elif r.type == "INTERNAL_REF":
                s2b_counts[r.resolution_method if r.resolution_method in s2b_counts else "unresolved"] += 1
        if art.decontext_method == "skipped_no_substitutions":
            s3a_counts["skipped"] += 1
        elif art.decontext_method == "llm":
            s3a_counts["llm"] += 1
        elif art.decontext_method == "llm_unparsed":
            s3a_counts["llm_unparsed"] += 1

        atomic_total += len(art.atomic_claims)

        evidence_refs = build_evidence_refs(art.resolved)
        decontext_map = build_decontextualization_map(art.resolved)

        # Span is resolved once per sentence and shared across its atomic claims.
        span_info = None
        if art.atomic_claims:
            span_info = resolve_span(sent.text, sent.section_number,
                                     sections[sent.section_number]["raw"],
                                     chunks_by_section.get(sent.section_number, []),
                                     args.threshold)

        trace_entry = {
            "global_idx": sent.global_idx,
            "section_number": sent.section_number,
            "local_idx": sent.local_idx,
            "sentence_text": sent.text,
            "mentions": [asdict(m) for m in art.mentions],
            "resolutions": [asdict(r) for r in art.resolved],
            "decontextualized_text": art.decontextualized_text,
            "decontext_method": art.decontext_method,
            "atomic_claims": art.atomic_claims,
            "dropped": bool(art.atomic_claims and span_info is None),
        }
        trace.append(trace_entry)

        if art.atomic_claims and span_info is None:
            # Tier-A reject: the sentence couldn't be located in its section, so
            # every atomic claim from it is dropped together.
            dropped.append({
                "section_number": sent.section_number,
                "local_idx": sent.local_idx,
                "global_idx": sent.global_idx,
                "sentence_text": sent.text,
                "decontextualized_text": art.decontextualized_text,
                "atomic_claims": art.atomic_claims,
            })
            continue

        if not art.atomic_claims:
            continue

        sentences_with_claims += 1
        if span_info["section_match_tier"] == "exact":
            section_match["exact"] += 1
        else:
            section_match["fuzzy"] += 1

        is_appendix = sent.section_number[:1].isalpha()
        is_excluded = sent.section_number.startswith("C")

        for atomic_idx, claim_text in enumerate(art.atomic_claims):
            chunk_match[span_info["chunk_match"]] += 1
            records.append({
                "claim_id": make_claim_id(sent.section_number, sent.local_idx, atomic_idx,
                                         doc_id=args.doc_id),
                "section_number": sent.section_number,
                "chunk_id": span_info["chunk_id"],
                "claim_text": claim_text,
                "quote": span_info["quote"],
                "span": span_info["span"],
                "section_match_tier": span_info["section_match_tier"],
                "section_match_score": span_info["section_match_score"],
                "chunk_match": span_info["chunk_match"],
                "is_appendix": is_appendix,
                "is_excluded": is_excluded,
                "evidence_refs": evidence_refs,
                "decontextualization_map": decontext_map,
            })

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "claims_v2.json").write_text(json.dumps(records, indent=2))
    (save_dir / "claims_v2_raw.json").write_text(json.dumps(trace, indent=2))
    (save_dir / "dropped_pairs_v2.json").write_text(json.dumps(dropped, indent=2))
    (save_dir / "named_entity_registry.json").write_text(
        json.dumps([asdict(e) for e in registry.entries], indent=2))

    total_tokens = sum(v["total"] for v in stage_tokens.values())
    new_tokens = sum(v["new"] for v in stage_tokens.values())
    stats = {
        "sections_requested": len(want),
        "sentences_selected": len(selected),
        "sentences_processed": len(trace),
        "sentences_with_claims": sentences_with_claims,
        "mentions_by_type": mentions_by_type,
        "stage2a_unnamed": s2a_counts,
        "stage2b_internal_ref": s2b_counts,
        "stage3a_decontextualize": s3a_counts,
        "claims_extracted": atomic_total,
        "claims_resolved": len(records),
        "claims_dropped": sum(len(d["atomic_claims"]) for d in dropped),
        "sentences_dropped": len(dropped),
        "section_match": section_match,
        "chunk_match": chunk_match,
        "tokens_by_stage": stage_tokens,
        "tokens": {"total": total_tokens, "new_this_run": new_tokens},
        "est_cost_usd": round(total_tokens / 1e6 * RATE_PER_MTOK, 4),
        "registry_entities": len(registry.entries),
        "aborted_on_budget": aborted,
    }
    (save_dir / "claims_v2_stats.json").write_text(json.dumps(stats, indent=2))

    run_config = {
        "doc_id": args.doc_id,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort if use_re else None,
        "reasoning_effort_accepted": use_re,
        "prompt_versions": {
            "stage1_mentions": PROMPT_VERSION_S1,
            "stage2a_unnamed": PROMPT_VERSION_S2,
            "stage2b_internal_ref": PROMPT_VERSION_S2_REF,
            "stage3a_decontextualize": PROMPT_VERSION_S3A,
            "stage3b_atomize": PROMPT_VERSION_S3B,
        },
        "candidate_tiers": CANDIDATE_TIERS,
        "fuzzy_threshold": args.threshold,
        "sections": want,
        "limit": args.limit,
        "extraction_unit": "sentence",
        "max_tokens_budget": args.max_tokens_budget,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reproduce": f"python scripts/extract_claims_v2.py --sections {args.sections} "
                     f"--threshold {args.threshold} --model {args.model} "
                     f"--doc-id {args.doc_id}"
                     + (f" --limit {args.limit}" if args.limit is not None else ""),
        "note": "quotes and spans are resolved deterministically post-hoc (span_match), "
                "never requested from the model -- same as v1. span is chunk-relative "
                "[start, end). All atomic claims split from one sentence share that "
                "sentence's quote/span/evidence_refs/decontextualization_map.",
    }
    (save_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("\n=== Done ===")
    for k, v in stats.items():
        print(f"  {k:>22}: {v}")
    if aborted:
        print("\n  NOTE: run aborted on token budget; outputs contain partial results.")
    print(f"\n  exported -> {save_dir}")


if __name__ == "__main__":
    main()
