#!/usr/bin/env python3
"""
Extract epistemic nodes across MULTIPLE documents in one run, then deduplicate.

This is a multi-document generalization of epistemic_node_extraction_improved_
prompt.py. The extraction stage (LLM decomposition into typed epistemic nodes,
then deterministic quote->span resolution) is unchanged in spirit; what is new:

  1. Documents are specified once per run via --documents config.json (a JSON
     list, one entry per document), instead of one --document-name/--sections
     invocation per document.
  2. All seven node types run by default (--node-types, default = all), instead
     of a single required --node-type.
  3. A two-tier deduplication stage runs after extraction, modeled on section
     3.3 ("Cross-Document Clustering and Conflict Detection") of the reference
     paper -- embedding-based candidate clustering confirmed by bidirectional
     NLI entailment -- but preceded by a cheap, deterministic same-document
     span-overlap pre-filter:

       Tier 1 (span overlap): free, deterministic, needs only rapidfuzz.
         Same doc_id + same chunk_id + overlapping chunk-relative spans AND
         high node_text similarity => same fact extracted twice. The text-
         similarity guard is essential: the ATOMIC prompt rule deliberately
         splits one sentence/quote into several distinct nodes, so span
         overlap ALONE would wrongly merge legitimately-different atomic
         siblings (whether same-type or, now, different-type). Only merges
         when both span and text conditions hold.

       Tier 2 (embedding + NLI): the paper's method, run only over the pool
         remaining after Tier 1 (one representative per Tier-1 group plus every
         un-merged node). Embeds node_text via Together, takes cosine-near
         candidate pairs, and confirms each merge with bidirectional NLI
         (min-entailment above a threshold AND max-contradiction below one) --
         the guard against "30 killed" vs "47 killed" sitting close in
         embedding space yet not being equivalent.

     Both tiers use clique-not-connected-components merging (a confirmed edge
     graph's connected component is only accepted as a group if every pair in
     it is a confirmed edge; otherwise its largest fully-confirmed subset is
     taken) -- the paper avoids single linkage "for its chaining", and taking
     raw connected components of pairwise-confirmed edges IS single linkage.

Dedup spans node_type as well as document: the same fact picked up under two
different types (e.g. a number-bearing sentence extracted as both `evidence`
and `quantitative_result`) is exactly the case this is meant to catch, not a
harder problem left alone -- unlike combine_epistemic_nodes.py's plain union,
which deliberately does not attempt this. A merged group's representative row
carries `node_types` (the sorted set of types spanned by its members) and
`dedup_cross_type` (bool), alongside the existing `node_type` field (kept as
the representative's own type, for callers that expect a single string).

Both tiers additionally apply a numeric-conflict veto (`_numeric_conflict`):
if two otherwise-matching texts disagree on an embedded number, the merge is
refused even though rapidfuzz/cosine/NLI all cleared their thresholds. This
is motivated by a real, previously-documented defect in this repo (see
CLAUDE.md's "Node vs. claims pipeline comparison" section): two different
node types independently extracting the same quote have rendered one Bayes-
factor cell as 1.39 vs 0.39. Entailment models are not reliably sensitive to
a single digit changing, so merging on NLI confirmation alone would silently
pick one of two possibly-wrong numbers as "the" value instead of surfacing
the disagreement. This guard does NOT catch the subtler certainty-reversal
case from the same CLAUDE.md section (e.g. "no bats were sold" vs "not known
if any bats were sold") -- that failure mode has no reliable cheap regex
signature, so it remains a known residual risk: read `dedup_groups.json`
periodically rather than assuming every merge it contains is safe.

Distinct outputs: everything goes under artifacts/nodes_multidoc/ and
outputs/nodes_multidoc/, never artifacts/<node_type>[_improved_prompt]/ or
artifacts/nodes_combined/, so the existing single-document pipeline's outputs
are never touched. PROMPT_VERSION is a distinct namespace so a shared cache
can't replay another prompt generation's responses.

Per-document layout knobs (first_body_page / toc_page_start / toc_page_end)
default to eric_decision.pdf's specific PDF layout (4, 1, 4). Those defaults
are a property of that document's layout, NOT a universal -- a genuinely
different document will likely need per-document overrides in its config
entry, the same spirit as this repo's note that HippoRAG's synonymy threshold
is "a property of the embedding model, not the document."

Dependencies
------------
Extraction needs only what epistemic_node_extraction_improved_prompt.py needs
(openai, python-dotenv, rapidfuzz, tiktoken, pypdf). The Tier-2 dedup stage
additionally needs numpy (required for cosine) and, for NLI confirmation,
sentence_transformers + torch. If those are missing the run does NOT crash:

  - Missing numpy -> Tier 2 is skipped with a loud, actionable message naming
    the pip command; Tier 1 (which needs no numpy) still runs, and a
    nodes_combined.json is still written.
  - Missing sentence_transformers/torch -> Tier 2 degrades to cosine-only
    clustering (every cosine candidate treated as confirmed), unless
    --require-nli was passed (then it's a hard failure). --no-nli skips even
    attempting NLI, for a fast dependency-light run.

To enable the full dedup path, in the activated `hipporag` conda env run:

    pip install numpy sentence-transformers torch

Environment
-----------
.env at the project root must contain TOGETHER_API_KEY (and optionally
TOGETHER_MODEL, EMBED_MODEL).

Examples
--------
Default single-document smoke test (reproduces the old script's default
document + curated sections, but across all 7 node types, auto-combined and
deduplicated):

    python scripts/extraction-variants/epistemic_node_extraction_multidoc.py \
        --node-types limitation,assumption --sections 7,7.1

Multi-document run:

    python scripts/extraction-variants/epistemic_node_extraction_multidoc.py \
        --documents my_docs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------
# extraction-variants/ scripts live one level deeper than scripts/, so ROOT is
# three parents up, not two. Moving this file back to scripts/ would break this.

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "scripts"))

from chunk_decision import recover_document  # noqa: E402
from span_match import locate  # noqa: E402


load_dotenv(ROOT / ".env")
print("ROOT =", ROOT)
print(".env exists:", (ROOT / ".env").exists())
print("API key loaded:", bool(os.getenv("TOGETHER_API_KEY")))


# ---------------------------------------------------------------------------
# Together AI configuration
# ---------------------------------------------------------------------------

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

DEFAULT_MODEL = os.getenv(
    "TOGETHER_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
)

DEFAULT_EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "intfloat/multilingual-e5-large-instruct",
)

# Distinct cache namespace so this pipeline can share a cache file with nothing
# else and never replays another prompt generation's cached responses.
PROMPT_VERSION = "epistemic-nodes-v1-multidoc"

# Together's listed rates as of 2026-07-19 (together.ai/pricing). Only used
# for a friendly cost estimate; not authoritative -- re-check if they drift.
# These are two different models billed at two different rates (chat
# completion vs. embedding), so the combined estimate tracks each separately
# rather than applying one blended rate to every token spent.
RATE_PER_MTOK = 1.04
EMBED_RATE_PER_MTOK = 0.02

# e5-instruct wants an instruction, and the instruction changes what "close"
# means. embed_nodes.py deliberately frames retrieval as "epistemically related
# (support, attack, or restate)" -- right for the pairing funnel, wrong here:
# dedup wants EQUIVALENCE, not relatedness, so this instruction is about
# restating the same underlying fact, not merely relating to it.
DEDUP_INSTRUCT = (
    "Instruct: Retrieve statements that assert the same underlying fact or "
    "claim as the given statement.\nQuery: "
)

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


# ---------------------------------------------------------------------------
# Default single-document config (reproduces the improved-prompt defaults)
# ---------------------------------------------------------------------------

DEFAULT_SECTIONS = [
    "7",
    "7.1",
    "7.2",
    "7.3",
    "7.4",
    "7.5",
    "7.6",
    "5.4",
    "5.4.1",
    "5.4.2",
    "5.4.3",
    "5.4.4",
]

DEFAULT_DOCUMENT_NAME = "eric_decision.pdf"
DEFAULT_DOCUMENT_DESCRIPTION = (
    "a judge's written decision in a structured public debate about COVID-19 "
    "origins (zoonosis vs. lab-leak), weighing competing hypotheses under "
    "explicit Bayesian argument."
)

# Placeholder used (with a loud warning) when a config entry omits its
# description -- a vague description measurably degrades extraction quality per
# this repo's prior findings, so we never proceed silently as if a missing
# description were fine.
PLACEHOLDER_DESCRIPTION = (
    "a source document (no description provided; extraction quality is "
    "measurably degraded without a specific one)"
)

# eric_decision.pdf-calibrated layout defaults; see module docstring.
DEFAULT_FIRST_BODY_PAGE = 4
DEFAULT_TOC_PAGE_START = 1
DEFAULT_TOC_PAGE_END = 4


# ---------------------------------------------------------------------------
# Node-type configuration (unchanged from epistemic_node_extraction_improved_prompt.py)
# ---------------------------------------------------------------------------

NODE_CONFIGS: dict[str, dict[str, str]] = {
    "research_question": {
        "plural": "research_questions",
        "text_field": "research_question",
        "definition": (
            "a substantive question that organizes an investigation, debate, "
            "analysis, or comparison of possible explanations"
        ),
        "rules": """
- Extract questions explicitly asked in the section.
- You may extract an implicit research question only when the section clearly
  investigates or attempts to answer it.
- Rewrite an implicit research question as a clear, self-contained question.
- Do not extract rhetorical questions that do not organize substantive analysis.
- Do not convert every topic, heading, or factual statement into a research question.
- Preserve the scope of the question, including relevant entities, events,
  locations, and time periods.
""",
    },
    "hypothesis": {
        "plural": "hypotheses",
        "text_field": "hypothesis",
        "definition": (
            "a proposed explanation, causal account, or possible answer to a "
            "research question that can be evaluated against evidence"
        ),
        "rules": """
- Extract primary, competing, alternative, rejected, or partially accepted hypotheses.
- A hypothesis must be an explanatory or answer-like proposition.
- Do not extract ordinary factual claims as hypotheses.
- Preserve attribution when the hypothesis belongs to a specific speaker,
  organization, or side of the debate.
- Preserve uncertainty and modality.
""",
    },
    "evidence": {
        "plural": "evidence_items",
        "text_field": "evidence",
        "definition": (
            "an observation, measurement, record, reported fact, dataset result, "
            "documented event, or source datum used to assess a claim or hypothesis"
        ),
        "rules": """
- Extract concrete observations, measurements, records, reported events, and
  source facts that play an evidential role.
- Do not extract the author's interpretation of evidence; that belongs to analysis.
- Preserve attribution when the evidence is reported by another source.
- Do not treat unsupported speculation as evidence.
- Do not extract every background fact unless it is relevant to the argument.
""",
    },
    "analysis": {
        "plural": "analyses",
        "text_field": "analysis",
        "definition": (
            "an inference, interpretation, comparison, explanation, calculation "
            "step, or reasoning step applied to evidence, assumptions, or claims"
        ),
        "rules": """
- Extract reasoning steps explaining why evidence affects the plausibility of a
  claim or hypothesis.
- Extract comparisons of likelihood, expectedness, causal relevance, or
  explanatory power.
- Do not extract raw observations as analysis.
- Do not extract a final hypothesis merely because the author endorses it.
- Make each analysis item atomic: one inference or reasoning step per item.
""",
    },
    "quantitative_result": {
        "plural": "quantitative_results",
        "text_field": "quantitative_result",
        "definition": (
            "a meaningful numerical measurement, estimate, probability, count, "
            "ratio, percentage, Bayes factor, prior, posterior, or calculated result"
        ),
        "rules": """
- Extract numerical values only when they play a meaningful role in the argument.
- Include what the number measures and any relevant conditioning or comparison.
- Preserve units, denominators, ranges, and uncertainty where stated.
- Preserve attribution when different participants provide different values.
- Do not extract section numbers, citation numbers, or incidental dates unless
  they are themselves analytically meaningful.
""",
    },
    "assumption": {
        "plural": "assumptions",
        "text_field": "assumption",
        "definition": (
            "a proposition treated as true, provisionally accepted, or required "
            "for an argument, model, estimate, or calculation"
        ),
        "rules": """
- Extract explicit assumptions and clearly identifiable implicit assumptions.
- An implicit assumption must be necessary for a reasoning step stated in the text.
- Do not invent hidden premises that are merely possible.
- Distinguish assumptions from evidence: assumptions are accepted for analysis
  rather than established by observations in the section.
- State each assumption as a self-contained proposition.
""",
    },
    "limitation": {
        "plural": "limitations",
        "text_field": "limitation",
        "definition": (
            "a caveat, uncertainty, missing-information issue, potential bias, "
            "methodological weakness, restricted scope, or reason for caution"
        ),
        "rules": """
- Extract explicit caveats, uncertainties, missing evidence, source weaknesses,
  possible biases, and restrictions on interpretation.
- Include what result, source, argument, or conclusion the limitation affects.
- Do not convert ordinary disagreement into a limitation.
- Preserve the author's degree of uncertainty.
- Keep separate limitations as separate atomic items.
""",
    },
}


# ---------------------------------------------------------------------------
# Prompt construction (unchanged from epistemic_node_extraction_improved_prompt.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise epistemic-node extraction assistant for an "
    "argument-mining and provenance-indexing pipeline over a source document. "
    "Your only job is to LOCATE and ATTRIBUTE the requested epistemic nodes, "
    "pairing each node with a verbatim supporting quotation copied from the "
    "text. You do not evaluate, endorse, correct, or add information of your "
    "own; you only mine and attribute what the passage itself states."
)


USER_TEMPLATE = """\
Document: "{document_name}" -- {document_description}

Section {number}: {title}

Node type: {node_type_label}

Definition:
A {node_type_label} is {definition}.

Task:
Extract all substantive {plural_label} from the SECTION TEXT below.

For every extracted item, return:

1. "{text_field}": an atomic, self-contained, decontextualized representation
   of the node.
2. "quote": one exact, verbatim, contiguous quotation from the section text
   that licenses the extracted node.

General rules:
- FAITHFUL: extract only {plural_label} actually stated or clearly licensed by \
this section. Each extracted node must be an accurate, non-inflated \
representation of its quote -- no added information, no reversed polarity, and \
no evaluation, endorsement, or correction of your own.
- COMPREHENSIVE: extract {plural_label} from every substantive instance in the \
section, including content inside tables, numbered lists, and bulleted lists -- \
treat each table row or list item as its own extraction opportunity rather than \
only paraphrasing prose that summarizes them. Do not let a named fact, figure, \
or instance embedded inside a longer sentence go unextracted just because it \
isn't that sentence's main assertion.
- DECONTEXTUALIZED: resolve pronouns, bridging references, abbreviations, and \
omitted subjects/times/locations so each node stands alone, naming the specific \
entities involved rather than leaving them implicit. Do this for every node in \
the section, not just the first one that introduces an entity -- if an earlier \
node names an entity in full, later nodes about the same entity must still name \
it in full, not fall back to a bare pronoun or generic noun. Preserve \
attribution (who is asserting, arguing, or estimating) consistently across \
every node that needs it, including the document's own author's or narrator's \
own estimates and conclusions, not only content attributed to named third \
parties.
- ATOMIC: each node must represent only one question, proposition, observation, \
inference, result, assumption, or limitation. When the source text bundles \
multiple independent facts or conditions together -- e.g. a numbered list \
enumerating several conditions, or a sentence joining unrelated propositions \
with "and"/"but" -- split them into separate nodes rather than one bundled \
node. A single conditional statement with one antecedent and one consequent \
may remain one node.
- The "quote" MUST be an exact, verbatim, contiguous substring of the section
  text. Copy it character-for-character.
- Do NOT normalize whitespace, correct spelling, fix punctuation, or combine
  non-contiguous fragments.
- Do NOT return character offsets or line numbers.
- Extract only information actually stated or clearly licensed by this section.
- Preserve negation, uncertainty, attribution, and modality.
- If the section contains no valid {plural_label}, return an empty list.

Node-specific rules:
{node_rules}

Return ONLY JSON, with no prose before or after, in exactly this form:

{{
  "{plural_key}": [
    {{
      "{text_field}": "<decontextualized {node_type_label}>",
      "quote": "<verbatim contiguous substring>"
    }}
  ]
}}

SECTION TEXT:
\"\"\"
{raw}
\"\"\"
"""


def build_user_prompt(
    node_type: str,
    number: str,
    title: str,
    raw: str,
    document_name: str,
    document_description: str,
) -> str:
    config = NODE_CONFIGS[node_type]

    return USER_TEMPLATE.format(
        number=number,
        title=title,
        node_type_label=node_type.replace("_", " "),
        definition=config["definition"],
        plural_label=config["plural"].replace("_", " "),
        plural_key=config["plural"],
        text_field=config["text_field"],
        node_rules=config["rules"].strip(),
        raw=raw,
        document_name=document_name,
        document_description=document_description,
    )


# ---------------------------------------------------------------------------
# Together/OpenAI-compatible client
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    api_key = os.environ.get("TOGETHER_API_KEY")

    if not api_key:
        sys.exit(
            "TOGETHER_API_KEY is missing. Add it to the project-root .env file."
        )

    return OpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=api_key,
        max_retries=3,
        timeout=300,
    )


def preflight(client: OpenAI, model: str) -> None:
    """Fail early if the model or Together configuration is invalid."""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": 'Reply with exactly this JSON: {"ok": true}',
            }
        ],
    )

    message = (response.choices[0].message.content or "").strip()
    usage = response.usage

    print(
        f"  preflight LLM ok -> {message[:60]!r} "
        f"({getattr(usage, 'completion_tokens', '?')} completion tokens)"
    )


def call_llm(
    client: OpenAI,
    model: str,
    node_type: str,
    number: str,
    title: str,
    raw: str,
    document_name: str,
    document_description: str,
    max_tokens: int,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_user_prompt(
                node_type=node_type,
                number=number,
                title=title,
                raw=raw,
                document_name=document_name,
                document_description=document_description,
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=messages,
    )

    usage = response.usage

    return {
        "content": response.choices[0].message.content or "",
        # Together defaults max_tokens to 2048 when omitted, which silently
        # truncates dense sections mid-JSON. Passing it explicitly and recording
        # finish_reason turns that into a loud warning instead (see the caller).
        "finish_reason": response.choices[0].finish_reason,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        },
    }


# ---------------------------------------------------------------------------
# Tolerant JSON parsing (unchanged from epistemic_node_extraction_improved_prompt.py)
# ---------------------------------------------------------------------------

def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _unescape_lenient(s: str) -> str:
    """Undo the handful of JSON escapes the model actually uses, for text
    captured by regex rather than a real JSON parser."""
    s = s.replace("\\\\", "\x00")
    s = s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    return s.replace("\x00", "\\")


def _regex_extract_pairs(text: str, text_field: str) -> list[dict[str, str]]:
    """Fallback for when the model's JSON has an unescaped inner quote inside a
    <text_field>/quote string (breaks json.loads, and the bad character
    survives into every {...}/[...] substring re-search too). Anchors on the
    literal "<text_field>": "...", "quote": "..." } field structure instead of
    requiring valid JSON."""
    pattern = re.compile(
        r'"' + re.escape(text_field) + r'"\s*:\s*"(.*?)"\s*,\s*"quote"\s*:\s*"(.*?)"\s*\}',
        re.DOTALL,
    )
    pairs = []
    for m in pattern.finditer(text):
        node_text = _unescape_lenient(m.group(1).strip())
        quote = _unescape_lenient(m.group(2).strip())
        if node_text and quote:
            pairs.append({"node_text": node_text, "quote": quote})
    return pairs


def parse_nodes(text: str, node_type: str) -> tuple[list[dict[str, str]], int]:
    """Parse node/quote pairs from a possibly imperfect LLM response."""

    config = NODE_CONFIGS[node_type]
    plural_key = config["plural"]
    text_field = config["text_field"]

    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    data = _try_json(cleaned)

    if data is None:
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                data = _try_json(match.group(0))
                if data is not None:
                    break

    if data is None:
        pairs = _regex_extract_pairs(cleaned, text_field)
        return [
            {"node_type": node_type, "node_text": p["node_text"], "quote": p["quote"]}
            for p in pairs
        ], 0

    if isinstance(data, dict):
        if isinstance(data.get(plural_key), list):
            data = data[plural_key]
        else:
            candidate_lists = [
                value for value in data.values() if isinstance(value, list)
            ]
            data = candidate_lists[0] if candidate_lists else [data]

    if not isinstance(data, list):
        return [], 0

    valid: list[dict[str, str]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        node_text = item.get(text_field)
        quote = item.get("quote")

        if (
            isinstance(node_text, str)
            and node_text.strip()
            and isinstance(quote, str)
            and quote.strip()
        ):
            valid.append(
                {
                    "node_type": node_type,
                    "node_text": node_text.strip(),
                    "quote": quote.strip(),
                }
            )

    return valid, len(data) - len(valid)


# ---------------------------------------------------------------------------
# Deterministic provenance resolution
# ---------------------------------------------------------------------------

def resolve_node(
    pair: dict[str, str],
    doc_id: str,
    document_name: str,
    section_number: str,
    section_raw: str,
    section_chunks: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    """Resolve one epistemic node to a section-level and chunk-level span.

    Adds doc_id/document_name to the record and folds doc_id into the node_id
    fingerprint, so identical text/quote coincidences across documents never
    collide on id.
    """

    section_match = locate(pair["quote"], section_raw, threshold)

    if section_match["tier"] == "reject":
        return None

    matched_text = section_raw[section_match["start"]:section_match["end"]]

    best_chunk = None
    best_chunk_match = None
    best_key = None

    for chunk in sorted(
        section_chunks,
        key=lambda item: item["chunk_order_in_section"],
    ):
        chunk_result = locate(matched_text, chunk["text"], threshold)

        if chunk_result["tier"] == "reject":
            continue

        key = (chunk_result["tier"] == "exact", chunk_result["score"])

        if best_key is None or key > best_key:
            best_chunk = chunk
            best_chunk_match = chunk_result
            best_key = key

    if best_chunk is None:
        chunk_id = None
        chunk_match = "split_across_chunks"
        span = None
    else:
        chunk_id = best_chunk["chunk_id"]
        chunk_match = best_chunk_match["tier"]
        span = [best_chunk_match["start"], best_chunk_match["end"]]

    node_type = pair["node_type"]

    fingerprint = (
        f"{doc_id}|"
        f"{node_type}|"
        f"{chunk_id or ''}|"
        f"{pair['node_text']}|"
        f"{pair['quote']}"
    )

    return {
        "node_id": (
            f"{node_type}-"
            + hashlib.md5(fingerprint.encode("utf-8")).hexdigest()
        ),
        "node_type": node_type,
        "doc_id": doc_id,
        "document_name": document_name,
        "section_number": section_number,
        "chunk_id": chunk_id,
        "node_text": pair["node_text"],
        "quote": pair["quote"],
        "span": span,
        "section_match_tier": section_match["tier"],
        "section_match_score": section_match["score"],
        "chunk_match": chunk_match,
        "is_appendix": section_number[:1].isalpha(),
        "is_excluded": section_number.startswith("C"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def slugify(name: str) -> str:
    """Lowercase, non-alnum -> '-', strip repeat/leading/trailing '-'."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return s.strip("-") or "doc"


def make_cache_key(
    doc_id: str,
    doc_fingerprint: str,
    section_number: str,
    node_type: str,
    model: str,
    section_raw: str,
) -> str:
    """Include doc_id + a document fingerprint (hash of name + description) and
    a section-text hash, so switching documents or editing a document's framing
    on a shared warm cache can't silently replay another document's response."""

    raw_hash = hashlib.md5(section_raw.encode("utf-8")).hexdigest()

    return (
        f"{doc_id}|"
        f"{section_number}|"
        f"{node_type}|"
        f"{PROMPT_VERSION}|"
        f"{model}|"
        f"{raw_hash}|"
        f"{doc_fingerprint}"
    )


def _norm_text(s: str) -> str:
    """Lowercase + whitespace-collapse for rapidfuzz text comparison."""
    return re.sub(r"\s+", " ", s.lower()).strip()


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def _numbers_in(s: str) -> set[str]:
    """Extract numeric tokens (counts, percentages, Bayes factors, etc.) for the
    numeric-conflict guard below. Deliberately crude -- a set-membership check,
    not unit-aware comparison."""
    return {tok.replace(",", "") for tok in _NUMBER_RE.findall(s or "")}


def _numeric_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if two candidate duplicates disagree on an embedded number.

    Motivated by a real, previously-documented defect in this repo: two
    different node types independently extracting the same source quote have
    rendered a Bayes-factor cell as 1.39 vs 0.39 (see CLAUDE.md's "Node vs.
    claims pipeline comparison" section). Two node texts can be near-identical
    by rapidfuzz/cosine/NLI-entailment and still disagree on the one number
    that matters -- entailment models are not reliably sensitive to a single
    digit changing. This is a cheap, deliberately blunt veto: if the numbers
    mentioned in the two node_texts (or their quotes) don't match, refuse the
    merge rather than silently picking one of two possibly-wrong values as
    "the" answer. It does not attempt to catch the subtler certainty-reversal
    case (e.g. "no bats were sold" vs "not known if any bats were sold") --
    that failure mode has no reliable cheap regex signature, so it is a known
    residual risk, not something this guard covers.
    """
    a_nums = _numbers_in(a["node_text"]) | _numbers_in(a.get("quote", ""))
    b_nums = _numbers_in(b["node_text"]) | _numbers_in(b.get("quote", ""))
    if not a_nums or not b_nums:
        return False
    return a_nums != b_nums


# ---------------------------------------------------------------------------
# Document config loading
# ---------------------------------------------------------------------------

def load_documents_config(
    documents_path: str | None,
    global_sections: list[str] | None,
) -> list[dict[str, Any]]:
    """Load and normalize the per-document config list.

    When --documents is omitted, synthesize a single entry reproducing the
    improved-prompt defaults (eric_decision.pdf, its description, its curated
    12-section sample), so a zero-flag run is a like-for-like smoke test of the
    old script's default behavior across all node types.
    """
    if documents_path:
        raw = json.loads(Path(documents_path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            sys.exit("--documents config must be a JSON list of document objects.")
        entries = raw
    else:
        # Reproduce the improved-prompt default exactly (eric_decision.pdf, its
        # description, its curated 12-section sample) -- EXCEPT if the user also
        # passed a global --sections, honor that as the restriction instead, so
        # `--sections 7,7.1` with no --documents actually narrows the default
        # document rather than being silently overridden by the baked-in sample.
        entries = [
            {
                "pdf": DEFAULT_DOCUMENT_NAME,
                "name": DEFAULT_DOCUMENT_NAME,
                "description": DEFAULT_DOCUMENT_DESCRIPTION,
                "sections": global_sections if global_sections else DEFAULT_SECTIONS,
            }
        ]

    normalized: list[dict[str, Any]] = []
    used_ids: dict[str, int] = {}

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("pdf"):
            sys.exit(f"--documents entry #{i} is missing the required 'pdf' field.")

        pdf_raw = entry["pdf"]
        pdf_path = Path(pdf_raw)
        if not pdf_path.is_absolute():
            pdf_path = ROOT / pdf_path

        name = entry.get("name") or Path(pdf_raw).name

        description = entry.get("description")
        if not description:
            print(
                f"  WARNING [{name}]: no 'description' given -- using a generic "
                f"placeholder. A vague document description measurably degrades "
                f"extraction quality (see the improved-prompt findings); add a "
                f"specific one to this document's config entry."
            )
            description = PLACEHOLDER_DESCRIPTION

        # doc_id from name, deterministically disambiguated on collision.
        base_id = slugify(name)
        if base_id in used_ids:
            used_ids[base_id] += 1
            doc_id = f"{base_id}-{used_ids[base_id]}"
            print(
                f"  NOTE: doc_id '{base_id}' already used; assigning '{doc_id}' "
                f"to document '{name}' (config position #{i})."
            )
        else:
            used_ids[base_id] = 1
            doc_id = base_id

        sections = entry.get("sections")
        if sections is not None and not isinstance(sections, list):
            sys.exit(f"--documents entry #{i} 'sections' must be a list if given.")

        normalized.append(
            {
                "doc_order": i,
                "doc_id": doc_id,
                "pdf_path": pdf_path,
                "pdf_config": pdf_raw,
                "name": name,
                "description": description,
                "sections": sections,  # may be None -> resolved per document
                "global_sections": global_sections,
                "first_body_page": int(entry.get("first_body_page", DEFAULT_FIRST_BODY_PAGE)),
                "toc_page_start": int(entry.get("toc_page_start", DEFAULT_TOC_PAGE_START)),
                "toc_page_end": int(entry.get("toc_page_end", DEFAULT_TOC_PAGE_END)),
            }
        )

    return normalized


# ---------------------------------------------------------------------------
# Deduplication -- shared graph helpers
# ---------------------------------------------------------------------------

def _connected_components(node_ids: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adj: dict[str, set[str]] = {n: set() for n in node_ids}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for n in node_ids:
        if n in seen:
            continue
        stack = [n]
        comp: list[str] = []
        seen.add(n)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _largest_clique(members: list[str], edges: set[tuple[str, str]]) -> list[str]:
    """Largest fully-mutually-confirmed subset of `members` (>=2). Brute force;
    components are expected to be single digits of members, so this is fine.

    Avoids single-linkage chaining: a connected component of pairwise-confirmed
    edges is only accepted as a group if every pair inside is confirmed;
    otherwise we fall back to its largest clique.
    """
    def connected(a: str, b: str) -> bool:
        return (a, b) in edges or (b, a) in edges

    n = len(members)
    for size in range(n, 1, -1):
        for subset in combinations(members, size):
            if all(connected(a, b) for a, b in combinations(subset, 2)):
                return list(subset)
    return []


def _clique_groups(node_ids: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    """Turn a confirmed-edge graph into clique groups of size >=2."""
    groups: list[list[str]] = []
    for comp in _connected_components(node_ids, edges):
        if len(comp) < 2:
            continue
        clique = _largest_clique(comp, edges)
        if len(clique) >= 2:
            groups.append(clique)
    return groups


def _spans_overlap(a: list[int] | None, b: list[int] | None) -> bool:
    if not a or not b:
        return False
    return a[0] < b[1] and b[0] < a[1]


# ---------------------------------------------------------------------------
# Deduplication -- Tier 1 (same-document span overlap)
# ---------------------------------------------------------------------------

def tier1_span_overlap(
    records: list[dict[str, Any]],
    text_threshold: float,
) -> tuple[list[list[str]], dict[str, dict[str, Any]], int]:
    """Return (groups, by_id, numeric_conflicts_skipped). Each group is a list
    of node_ids that are the same fact extracted twice within one document:
    same doc_id + same chunk_id + overlapping spans AND high node_text
    similarity. The text-similarity guard is what keeps legitimately-distinct
    ATOMIC siblings (which can share/overlap a source quote) from being
    wrongly merged.

    Deliberately spans node_type -- the whole point of this tier is to also
    catch the same fact picked up under two different types (e.g. the same
    number-bearing sentence extracted as both `evidence` and
    `quantitative_result`). The text-similarity gate is what actually guards
    against false merges here, not type-matching, so dropping node_type from
    the bucket key is safe: two genuinely different pieces of content
    (whether same-type ATOMIC siblings or different-type extractions of an
    overlapping span) still won't clear the similarity bar just because their
    spans overlap. The _numeric_conflict check adds a second, narrower guard
    for the one failure mode text-similarity alone can miss (see its
    docstring).
    """
    from rapidfuzz.fuzz import ratio as fuzz_ratio

    by_id = {r["node_id"]: r for r in records}

    # bucket by (doc_id, chunk_id); only within-bucket pairs can be span-
    # overlap candidates. chunk_id null (split_across_chunks) is excluded.
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        if r.get("chunk_id") is None or not r.get("span"):
            continue
        key = (r["doc_id"], r["chunk_id"])
        buckets.setdefault(key, []).append(r)

    all_group_ids: list[str] = []
    groups: list[list[str]] = []
    numeric_conflicts_skipped = 0

    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        ids = [r["node_id"] for r in bucket]
        edges: set[tuple[str, str]] = set()
        for ra, rb in combinations(bucket, 2):
            if ra["node_id"] == rb["node_id"]:
                continue
            if not _spans_overlap(ra["span"], rb["span"]):
                continue
            sim = fuzz_ratio(_norm_text(ra["node_text"]), _norm_text(rb["node_text"])) / 100.0
            if sim < text_threshold:
                continue
            if _numeric_conflict(ra, rb):
                numeric_conflicts_skipped += 1
                continue
            edges.add((ra["node_id"], rb["node_id"]))
        if not edges:
            continue
        for clique in _clique_groups(ids, edges):
            groups.append(clique)
            all_group_ids.extend(clique)

    return groups, by_id, numeric_conflicts_skipped


# ---------------------------------------------------------------------------
# Deduplication -- Tier 2 (cross-document embedding + NLI)
# ---------------------------------------------------------------------------

def embed_texts(
    client: OpenAI,
    model: str,
    items: list[tuple[str, str]],
    cache_path: Path,
    no_cache: bool,
    batch_size: int = 64,
) -> tuple[dict[str, list[float]], int]:
    """Embed (node_id, node_text) pairs via Together, caching on (model,
    node_id). Mirrors embed_nodes.py's incremental disk cache.

    Returns (embeddings_by_node_id, new_tokens) -- new_tokens counts only
    tokens actually billed this call (cache hits are free), same convention
    as the LLM extraction loop's new_tokens/total_tokens split.
    """
    cache: dict[str, list[float]] = {}
    if cache_path.exists() and not no_cache:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    todo = [(nid, txt) for nid, txt in items if f"{model}|{nid}" not in cache]
    print(f"    embeddings: {len(items) - len(todo)} cached, {len(todo)} to embed")

    new_tokens = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        resp = client.embeddings.create(
            model=model,
            input=[DEDUP_INSTRUCT + txt for _, txt in batch],
        )
        usage = getattr(resp, "usage", None)
        new_tokens += getattr(usage, "total_tokens", 0) or 0
        rows = sorted(resp.data, key=lambda d: d.index)
        for (nid, _txt), row in zip(batch, rows):
            cache[f"{model}|{nid}"] = row.embedding
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    return {nid: cache[f"{model}|{nid}"] for nid, _ in items}, new_tokens


def tier2_semantic(
    pool_records: list[dict[str, Any]],
    client: OpenAI,
    embed_model: str,
    embed_cache_path: Path,
    no_cache: bool,
    dedup_threshold: float,
    use_nli: bool,
    require_nli: bool,
    nli_entail_threshold: float,
    nli_contra_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cross-document AND cross-node-type semantic dedup over the post-Tier-1
    pool. Returns (groups, info) where each group is a dict with member pool
    rep_ids, edges, and avg_cosine, and info records what actually ran.

    Deliberately spans node_type -- catches the same fact independently
    extracted under two different types (e.g. a number-bearing sentence
    picked up as both `evidence` and `quantitative_result`), not just the
    same type restated across documents. The embedding-cosine candidate gate
    plus bidirectional NLI confirmation are what guard against false merges
    here; they don't care whether the two texts came from the same
    extraction prompt or two different ones. A numeric-conflict veto (see
    `_numeric_conflict`) additionally refuses to merge two texts that are
    otherwise near-identical but disagree on an embedded number -- the one
    disagreement NLI entailment models aren't reliably sensitive to.

    Raises ImportError for numpy (caller turns that into graceful skip).
    """
    import numpy as np  # required; caller handles ImportError

    info: dict[str, Any] = {
        "ran": True,
        "nli_attempted": use_nli,
        "nli_confirmed": False,
        "nli_skipped_reason": None,
        "dedup_threshold": dedup_threshold,
        "nli_entail_threshold": nli_entail_threshold,
        "nli_contra_threshold": nli_contra_threshold,
        "embed_tokens_new": 0,
        "numeric_conflicts_skipped": 0,
    }

    # Try to load NLI unless disabled.
    ce = None
    label2idx: dict[str, int] = {}
    if use_nli:
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(NLI_MODEL)
            id2label = {int(k): v.lower() for k, v in ce.model.config.id2label.items()}
            label2idx = {v: k for k, v in id2label.items()}
            info["nli_confirmed"] = True
        except ImportError as exc:
            if require_nli:
                sys.exit(
                    "--require-nli was set but sentence_transformers/torch are "
                    "not importable. Install them in the hipporag env:\n"
                    "    pip install sentence-transformers torch\n"
                    f"(import error: {exc})"
                )
            print(
                "    WARNING: sentence_transformers/torch not importable -- "
                "NLI confirmation SKIPPED; degrading to cosine-only clustering "
                "(every cosine candidate treated as confirmed). Install with "
                "`pip install sentence-transformers torch` for the full guard, "
                "or pass --require-nli to make this a hard failure."
            )
            info["nli_skipped_reason"] = f"ImportError: {exc}"
    else:
        info["nli_skipped_reason"] = "--no-nli passed"

    groups: list[dict[str, Any]] = []

    if len(pool_records) >= 2:
        emb_map, embed_new_tokens = embed_texts(
            client,
            embed_model,
            [(r["_rep_id"], r["_rep_text"]) for r in pool_records],
            embed_cache_path,
            no_cache,
        )
        info["embed_tokens_new"] += embed_new_tokens

        ids = [r["_rep_id"] for r in pool_records]
        rec_by_id = {r["_rep_id"]: r for r in pool_records}
        mat = np.array([emb_map[i] for i in ids], dtype=np.float32)
        norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sim = norm @ norm.T

        # cosine candidate pairs
        cand: list[tuple[int, int, float]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                c = float(sim[i, j])
                if c >= dedup_threshold:
                    cand.append((i, j, c))

        # NLI confirmation (bidirectional) unless degraded to cosine-only.
        # The numeric-conflict veto applies regardless of NLI availability --
        # it catches a failure mode (disagreeing on one embedded number)
        # that entailment scoring is not reliably sensitive to.
        confirmed: set[tuple[str, str]] = set()
        cos_of: dict[tuple[str, str], float] = {}
        if cand and ce is not None:
            text_by_id = {r["_rep_id"]: r["_rep_text"] for r in pool_records}
            pairs_text = [(text_by_id[ids[i]], text_by_id[ids[j]]) for i, j, _ in cand]
            ab = ce.predict([[a, b] for a, b in pairs_text], apply_softmax=True, show_progress_bar=False)
            ba = ce.predict([[b, a] for a, b in pairs_text], apply_softmax=True, show_progress_bar=False)
            ent_i = label2idx["entailment"]
            con_i = label2idx["contradiction"]
            for (i, j, c), sab, sba in zip(cand, ab, ba):
                ent = min(float(sab[ent_i]), float(sba[ent_i]))
                con = max(float(sab[con_i]), float(sba[con_i]))
                if ent < nli_entail_threshold or con >= nli_contra_threshold:
                    continue
                a_item = {"node_text": rec_by_id[ids[i]]["_rep_text"], "quote": rec_by_id[ids[i]].get("_rep_quote", "")}
                b_item = {"node_text": rec_by_id[ids[j]]["_rep_text"], "quote": rec_by_id[ids[j]].get("_rep_quote", "")}
                if _numeric_conflict(a_item, b_item):
                    info["numeric_conflicts_skipped"] += 1
                    continue
                confirmed.add((ids[i], ids[j]))
                cos_of[(ids[i], ids[j])] = c
        else:
            for i, j, c in cand:
                a_item = {"node_text": rec_by_id[ids[i]]["_rep_text"], "quote": rec_by_id[ids[i]].get("_rep_quote", "")}
                b_item = {"node_text": rec_by_id[ids[j]]["_rep_text"], "quote": rec_by_id[ids[j]].get("_rep_quote", "")}
                if _numeric_conflict(a_item, b_item):
                    info["numeric_conflicts_skipped"] += 1
                    continue
                confirmed.add((ids[i], ids[j]))
                cos_of[(ids[i], ids[j])] = c

        for clique in _clique_groups(ids, confirmed):
            cosines = [
                cos_of.get((a, b), cos_of.get((b, a)))
                for a, b in combinations(clique, 2)
                if (a, b) in cos_of or (b, a) in cos_of
            ]
            avg_cos = round(sum(cosines) / len(cosines), 4) if cosines else None
            groups.append(
                {
                    "members": clique,  # these are pool rep_ids
                    "avg_cosine": avg_cos,
                }
            )

    return groups, info


# ---------------------------------------------------------------------------
# Deduplication -- orchestration and output assembly
# ---------------------------------------------------------------------------

def run_dedup(
    resolved: list[dict[str, Any]],
    sort_key_of: dict[str, tuple],
    args: argparse.Namespace,
    client: OpenAI,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run both dedup tiers and assemble nodes_combined + dedup_groups.

    Returns (combined_rows, group_records, dedup_stats). Never raises for a
    missing optional dependency -- degrades and records why.
    """
    by_id = {r["node_id"]: r for r in resolved}

    def rep_of(node_ids: list[str]) -> str:
        # longest node_text; tie -> earliest (doc_order, section_order, ...).
        # sort_key_of holds a (int, int, str, int) tuple, so it can't be
        # negated -- sort ascending on (-len, sort_key) and take the first.
        return sorted(
            node_ids,
            key=lambda nid: (-len(by_id[nid]["node_text"]), sort_key_of[nid]),
        )[0]

    # --- Tier 1: span overlap (needs only rapidfuzz) ---
    t1_groups, _, t1_numeric_conflicts = tier1_span_overlap(
        resolved, args.span_overlap_text_threshold
    )
    in_t1: set[str] = set()
    t1_group_of_member: dict[str, list[str]] = {}
    for g in t1_groups:
        for nid in g:
            in_t1.add(nid)
            t1_group_of_member[nid] = g

    # Build the Tier-2 pool: one representative per Tier-1 group + every
    # un-merged node. Each pool item carries its underlying node_id set and
    # the representative's own quote (_rep_quote), needed by Tier 2's
    # numeric-conflict veto.
    pool: list[dict[str, Any]] = []
    seen_t1: set[int] = set()
    for g in t1_groups:
        gid = id(g)
        if gid in seen_t1:
            continue
        seen_t1.add(gid)
        rep = rep_of(g)
        pool.append({"_rep_id": rep, "_rep_text": by_id[rep]["node_text"],
                     "_rep_quote": by_id[rep]["quote"],
                     "node_type": by_id[rep]["node_type"], "_members": list(g),
                     "_tier1": True})
    for r in resolved:
        if r["node_id"] in in_t1:
            continue
        pool.append({"_rep_id": r["node_id"], "_rep_text": r["node_text"],
                     "_rep_quote": r["quote"],
                     "node_type": r["node_type"], "_members": [r["node_id"]],
                     "_tier1": False})

    members_of_pool = {p["_rep_id"]: p["_members"] for p in pool}
    tier1_flag_of_pool = {p["_rep_id"]: p["_tier1"] for p in pool}

    # --- Tier 2: embedding + NLI (needs numpy; optionally sentence_transformers) ---
    numpy_ok = True
    t2_groups: list[dict[str, Any]] = []
    t2_info: dict[str, Any] = {"ran": False, "nli_attempted": False,
                               "nli_confirmed": False,
                               "nli_skipped_reason": "not reached"}
    try:
        import numpy  # noqa: F401
    except ImportError:
        numpy_ok = False

    if not numpy_ok:
        print(
            "\n  WARNING: numpy is not importable -- Tier 2 (cross-document "
            "embedding+NLI dedup) SKIPPED. Tier 1 (span-overlap) results are "
            "still applied. Install numpy (and, for NLI, sentence-transformers "
            "and torch) in the hipporag env to enable Tier 2:\n"
            "    pip install numpy sentence-transformers torch"
        )
        t2_info["nli_skipped_reason"] = "numpy not importable"
    elif len(pool) < 2:
        t2_info["nli_skipped_reason"] = "fewer than 2 pool items"
    else:
        use_nli = not args.no_nli
        t2_groups, t2_info = tier2_semantic(
            pool_records=pool,
            client=client,
            embed_model=args.embed_model,
            embed_cache_path=Path(args.embed_cache),
            no_cache=args.no_cache,
            dedup_threshold=args.dedup_threshold,
            use_nli=use_nli,
            require_nli=args.require_nli,
            nli_entail_threshold=args.nli_entail_threshold,
            nli_contra_threshold=args.nli_contra_threshold,
        )

    # pool reps that got merged by Tier 2
    t2_pool_reps: set[str] = set()
    for g in t2_groups:
        for rep in g["members"]:
            t2_pool_reps.add(rep)

    # --- Assemble final groups (one final group per node, exactly once) ---
    group_records: list[dict[str, Any]] = []
    final_group_of: dict[str, dict[str, Any]] = {}  # node_id -> group_record

    def member_view(node_ids: list[str]) -> list[dict[str, Any]]:
        ordered = sorted(node_ids, key=lambda nid: sort_key_of[nid])
        return [
            {
                "node_id": nid,
                "doc_id": by_id[nid]["doc_id"],
                "document_name": by_id[nid]["document_name"],
                "section_number": by_id[nid]["section_number"],
                "node_type": by_id[nid]["node_type"],
                "quote": by_id[nid]["quote"],
            }
            for nid in ordered
        ]

    def make_group_record(node_ids: list[str], tier: str, avg_cosine, nli_confirmed: bool) -> dict[str, Any]:
        rep = rep_of(node_ids)
        docs = sorted({by_id[nid]["doc_id"] for nid in node_ids})
        types = sorted({by_id[nid]["node_type"] for nid in node_ids})
        gid_prefix = "spanoverlap" if tier == "span_overlap" else "semantic"
        rec = {
            "dedup_group_id": f"{gid_prefix}-{by_id[rep]['node_type']}-{by_id[rep]['node_id']}",
            "dedup_tier": tier,
            # representative's own type, kept for backward-compat display;
            # `node_types` below is the full set spanned by this group.
            "node_type": by_id[rep]["node_type"],
            "node_types": types,
            "cross_type": len(types) > 1,
            "representative_node_id": rep,
            "representative_node_text": by_id[rep]["node_text"],
            "member_count": len(node_ids),
            "documents": docs,
            "cross_document": len(docs) > 1,
            "avg_cosine": avg_cosine,
            "nli_confirmed": nli_confirmed,
            "members": member_view(node_ids),
        }
        return rec

    # Tier-2 groups first (they may absorb Tier-1 group representatives).
    for g in t2_groups:
        underlying: list[str] = []
        for rep in g["members"]:
            underlying.extend(members_of_pool[rep])
        rec = make_group_record(
            underlying,
            tier="semantic",
            avg_cosine=g["avg_cosine"],
            nli_confirmed=t2_info.get("nli_confirmed", False),
        )
        group_records.append(rec)
        for nid in underlying:
            final_group_of[nid] = rec

    # Surviving Tier-1 groups (whose pool rep was NOT absorbed by Tier 2).
    for g in t1_groups:
        rep = rep_of(g)
        if rep in t2_pool_reps:
            continue  # absorbed into a semantic group above
        # (also skip if any member already claimed -- shouldn't happen, but be safe)
        if any(nid in final_group_of for nid in g):
            continue
        rec = make_group_record(
            list(g),
            tier="span_overlap",
            avg_cosine=None,
            nli_confirmed=False,
        )
        group_records.append(rec)
        for nid in g:
            final_group_of[nid] = rec

    # --- Build nodes_combined rows: one row per final group ---
    combined_rows: list[dict[str, Any]] = []
    consumed: set[str] = set()

    # emit merged-group rows keyed by representative, in overall sort order
    emitted_group_reps: set[str] = set()
    for r in sorted(resolved, key=lambda x: sort_key_of[x["node_id"]]):
        nid = r["node_id"]
        if nid in consumed:
            continue
        grp = final_group_of.get(nid)
        if grp is not None:
            rep = grp["representative_node_id"]
            if rep in emitted_group_reps:
                consumed.add(nid)
                continue
            emitted_group_reps.add(rep)
            rep_rec = by_id[rep]
            row = dict(rep_rec)
            row["node_types"] = grp["node_types"]
            row["dedup_group_id"] = grp["dedup_group_id"]
            row["dedup_group_size"] = grp["member_count"]
            row["dedup_tier"] = grp["dedup_tier"]
            row["dedup_cross_document"] = grp["cross_document"]
            row["dedup_cross_type"] = grp["cross_type"]
            row["dedup_members"] = grp["members"]
            combined_rows.append(row)
            for m in grp["members"]:
                consumed.add(m["node_id"])
        else:
            row = dict(r)
            row["node_types"] = [r["node_type"]]
            row["dedup_group_id"] = f"{r['node_type']}-solo-{r['node_id']}"
            row["dedup_group_size"] = 1
            row["dedup_tier"] = None
            row["dedup_cross_document"] = False
            row["dedup_cross_type"] = False
            row["dedup_members"] = None
            combined_rows.append(row)
            consumed.add(nid)

    combined_rows.sort(key=lambda x: sort_key_of[x["node_id"]])

    n_t1_merged = sum(len(g) for g in t1_groups)
    dedup_stats = {
        "dedup_ran": True,
        "input_records": len(resolved),
        "tier1_span_overlap": {
            "groups": len(t1_groups),
            "records_merged": n_t1_merged,
            "text_threshold": args.span_overlap_text_threshold,
            "numeric_conflicts_skipped": t1_numeric_conflicts,
        },
        "tier2_semantic": {
            "ran": t2_info.get("ran", False),
            "groups": len(t2_groups),
            "nli_attempted": t2_info.get("nli_attempted", False),
            "nli_confirmed": t2_info.get("nli_confirmed", False),
            "nli_skipped_reason": t2_info.get("nli_skipped_reason"),
            "dedup_threshold": args.dedup_threshold,
            "nli_entail_threshold": args.nli_entail_threshold,
            "nli_contra_threshold": args.nli_contra_threshold,
            "embed_tokens_new": t2_info.get("embed_tokens_new", 0),
            "numeric_conflicts_skipped": t2_info.get("numeric_conflicts_skipped", 0),
        },
        "final_groups": len(group_records),
        "final_rows": len(combined_rows),
        "rows_removed_by_dedup": len(resolved) - len(combined_rows),
        "cross_document_groups": sum(1 for g in group_records if g["cross_document"]),
        "cross_type_groups": sum(1 for g in group_records if g["cross_type"]),
    }

    return combined_rows, group_records, dedup_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--documents", default=None,
                        help="Path to a JSON list of document config objects. "
                             "Omit for the default single-document smoke test.")
    parser.add_argument("--node-types", default=None,
                        help="Comma-separated node types. Default: all 7, sorted.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Together-hosted chat model identifier.")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Minimum fuzzy quote-match similarity [0, 1].")
    parser.add_argument("--sections", default=None,
                        help="Global fallback section list for documents whose "
                             "config entry omits 'sections'. If neither exists, "
                             "all of a document's located sections are used.")
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Max completion tokens per section.")
    parser.add_argument("--max-tokens-budget", type=int, default=200_000,
                        help="Abort after this many NEW tokens (global across "
                             "the whole multi-doc, multi-type run).")
    parser.add_argument("--save-dir", default=None,
                        help="Output root. Default: artifacts/nodes_multidoc.")
    parser.add_argument("--cache", default=None,
                        help="Shared LLM response cache. Default: "
                             "outputs/nodes_multidoc/llm_cache.json.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore existing cached LLM/embedding responses.")

    # --- dedup flags ---
    parser.add_argument("--dedup-threshold", type=float, default=0.90,
                        help="Tier-2 cosine candidate threshold. UNCALIBRATED "
                             "default for real multi-document data; e5 node-text "
                             "cosine is compressed (synonyms >=0.96 per "
                             "pairing_funnel.py), so treat this as a starting "
                             "point to tune, not a validated value.")
    parser.add_argument("--nli-entail-threshold", type=float, default=0.5,
                        help="Min bidirectional entailment to confirm a merge.")
    parser.add_argument("--nli-contra-threshold", type=float, default=0.5,
                        help="Max contradiction (either direction) to allow a merge.")
    parser.add_argument("--span-overlap-text-threshold", type=float, default=0.90,
                        help="Min node_text rapidfuzz ratio for a Tier-1 "
                             "span-overlap merge (guards against merging "
                             "distinct ATOMIC siblings).")
    parser.add_argument("--require-nli", action="store_true",
                        help="Hard-fail if NLI (sentence_transformers/torch) is "
                             "unavailable, instead of degrading to cosine-only.")
    parser.add_argument("--no-nli", action="store_true",
                        help="Skip NLI entirely (fast, cosine-only Tier 2).")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Together embedding model for Tier-2 dedup.")
    parser.add_argument("--embed-cache", default=None,
                        help="Embedding cache. Default: "
                             "outputs/nodes_multidoc/embed_cache.json.")

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    # node types
    if args.node_types:
        node_types = [t.strip() for t in args.node_types.split(",") if t.strip()]
        unknown = [t for t in node_types if t not in NODE_CONFIGS]
        if unknown:
            parser.error(f"unknown node type(s): {unknown}. "
                         f"Choose from {sorted(NODE_CONFIGS)}.")
    else:
        node_types = sorted(NODE_CONFIGS)

    global_sections = None
    if args.sections:
        global_sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    save_dir = Path(args.save_dir) if args.save_dir else ROOT / "artifacts" / "nodes_multidoc"
    cache_path = Path(args.cache) if args.cache else ROOT / "outputs" / "nodes_multidoc" / "llm_cache.json"
    if args.embed_cache is None:
        args.embed_cache = str(ROOT / "outputs" / "nodes_multidoc" / "embed_cache.json")

    documents = load_documents_config(args.documents, global_sections)

    print(f"\nModel:      {args.model} @ Together")
    print(f"Node types: {', '.join(node_types)}")
    print(f"Documents:  {len(documents)}")
    for d in documents:
        print(f"  - [{d['doc_id']}] {d['name']} <- {d['pdf_config']}")
    print(f"Save dir:   {save_dir}")
    print(f"Cache:      {cache_path}")
    print(f"Budget:     {args.max_tokens_budget:,} new tokens (global)\n")

    client = build_client()
    preflight(client, args.model)

    cache: dict[str, Any] = {}
    if cache_path.exists() and not args.no_cache:
        cache = load_json_object(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # global accumulators
    all_resolved: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    sort_key_of: dict[str, tuple] = {}
    extraction_idx = 0
    new_tokens = 0
    total_tokens = 0
    aborted = False
    truncated: list[str] = []
    skipped_docs: list[dict[str, Any]] = []
    per_doc_type_stats: list[dict[str, Any]] = []
    resolved_doc_configs: list[dict[str, Any]] = []

    for d in documents:
        doc_id = d["doc_id"]
        doc_order = d["doc_order"]
        name = d["name"]
        description = d["description"]

        print(f"\n=== Document [{doc_id}] {name} ===")
        if not d["pdf_path"].exists():
            print(f"  WARNING: PDF not found at {d['pdf_path']} -- skipping document.")
            skipped_docs.append({"doc_id": doc_id, "name": name,
                                 "reason": f"pdf not found: {d['pdf_path']}"})
            continue

        recovered = recover_document(
            d["pdf_path"],
            first_body_page=d["first_body_page"],
            toc_page_start=d["toc_page_start"],
            toc_page_end=d["toc_page_end"],
        )

        if not recovered["toc"] or not recovered["sections"]:
            print(
                f"  WARNING: document '{name}' yielded an EMPTY table of "
                f"contents (parse_toc found {len(recovered['toc'])} entries, "
                f"located {len(recovered['sections'])} sections). This is the "
                f"expected outcome for a document with no machine-extractable "
                f"dotted TOC (e.g. a typical arXiv paper). Skipping this "
                f"document with 0 sections processed -- adjust its "
                f"toc_page_start/toc_page_end/first_body_page in the config if "
                f"it DOES have a TOC elsewhere."
            )
            skipped_docs.append({"doc_id": doc_id, "name": name,
                                 "reason": "empty TOC (no machine-extractable sections)"})
            continue

        # ordered located sections + raw text map
        located = recovered["sections"]
        body = recovered["body"]
        section_order = {s["number"]: i for i, s in enumerate(located)}
        raw_by_section = {
            s["number"]: {"title": s["title"], "raw": body[s["start"]:s["end"]].strip()}
            for s in located
        }

        # chunks for this document (from recover_document, not data/chunks.json)
        chunks_by_section: dict[str, list[dict[str, Any]]] = {}
        for ch in recovered["chunks"]:
            chunks_by_section.setdefault(ch["section_number"], []).append(ch)

        # resolve section scope
        if d["sections"]:
            scope = d["sections"]
        elif global_sections:
            scope = global_sections
        else:
            scope = [s["number"] for s in located]
        # keep only sections actually located, in document order
        scope = [n for n in scope if n in raw_by_section]
        scope.sort(key=lambda n: section_order.get(n, 10**6))

        print(f"  located {len(located)} sections; extracting {len(scope)} in scope")

        doc_fingerprint = hashlib.md5(
            f"{name}|{description}".encode("utf-8")
        ).hexdigest()[:8]

        for node_type in node_types:
            config = NODE_CONFIGS[node_type]
            raw_pairs: list[dict[str, Any]] = []
            invalid_total = 0
            sections_processed: set[str] = set()

            for number in scope:
                sec = raw_by_section.get(number)
                if sec is None or not sec["raw"]:
                    continue

                key = make_cache_key(
                    doc_id=doc_id,
                    doc_fingerprint=doc_fingerprint,
                    section_number=number,
                    node_type=node_type,
                    model=args.model,
                    section_raw=sec["raw"],
                )
                cached = key in cache and not args.no_cache

                if cached:
                    entry = cache[key]
                    total_tokens += entry["usage"]["total_tokens"]
                else:
                    if new_tokens >= args.max_tokens_budget:
                        print(f"  [{doc_id}/{node_type}/{number}] token budget "
                              f"reached ({new_tokens:,}); saving partial results")
                        aborted = True
                        break
                    entry = call_llm(
                        client=client, model=args.model, node_type=node_type,
                        number=number, title=sec["title"], raw=sec["raw"],
                        document_name=name, document_description=description,
                        max_tokens=args.max_tokens,
                    )
                    cache[key] = entry
                    write_json(cache_path, cache)
                    new_tokens += entry["usage"]["total_tokens"]
                    total_tokens += entry["usage"]["total_tokens"]

                if entry.get("finish_reason") == "length":
                    tag = f"{doc_id}/{node_type}/{number}"
                    truncated.append(tag)
                    print(f"  [{tag}] WARNING: response hit max_tokens "
                          f"({args.max_tokens:,}) and was truncated -- nodes "
                          f"likely incomplete or missing")

                pairs, invalid_count = parse_nodes(entry["content"], node_type)
                invalid_total += invalid_count
                sections_processed.add(number)
                for pair in pairs:
                    raw_pairs.append({"section_number": number, **pair})

            if aborted:
                # record what we have for this type before stopping everything
                pass

            # resolve spans for this doc+type
            records: list[dict[str, Any]] = []
            dropped: list[dict[str, Any]] = []
            for pair in raw_pairs:
                number = pair["section_number"]
                rec = resolve_node(
                    pair=pair, doc_id=doc_id, document_name=name,
                    section_number=number, section_raw=raw_by_section[number]["raw"],
                    section_chunks=chunks_by_section.get(number, []),
                    threshold=args.threshold,
                )
                if rec is None:
                    dropped.append(pair)
                    continue
                records.append(rec)
                sort_key_of[rec["node_id"]] = (
                    doc_order, section_order.get(number, 10**6), node_type, extraction_idx,
                )
                extraction_idx += 1
                all_resolved.append(rec)

            for pair in raw_pairs:
                all_raw.append({"doc_id": doc_id, "document_name": name, **pair})

            # per-doc-per-type output files
            plural = config["plural"]
            out_dir = save_dir / "by_doc" / doc_id / node_type
            write_json(out_dir / f"{plural}_raw.json",
                       [{"doc_id": doc_id, "document_name": name, **p} for p in raw_pairs])
            write_json(out_dir / f"{plural}.json", records)
            write_json(out_dir / "dropped_pairs.json", dropped)
            type_stats = {
                "doc_id": doc_id, "node_type": node_type,
                "sections_processed": len(sections_processed),
                "nodes_extracted": len(raw_pairs),
                "invalid_pairs": invalid_total,
                "nodes_resolved": len(records),
                "nodes_dropped": len(dropped),
            }
            write_json(out_dir / f"{plural}_stats.json", type_stats)
            per_doc_type_stats.append(type_stats)

            est = total_tokens / 1_000_000 * RATE_PER_MTOK
            print(f"  [{node_type:20s}] {len(records):>3} resolved "
                  f"/ {len(raw_pairs):>3} extracted, {len(dropped)} dropped "
                  f"(cumulative {total_tokens:,} tok ~= ${est:.4f})")

            if aborted:
                break

        resolved_doc_configs.append({
            "doc_id": doc_id, "doc_order": doc_order, "name": name,
            "description": description, "pdf": str(d["pdf_config"]),
            "sections_scope": scope,
            "first_body_page": d["first_body_page"],
            "toc_page_start": d["toc_page_start"], "toc_page_end": d["toc_page_end"],
        })

        if aborted:
            break

    # ---- combined raw (pre-dedup), sorted ----
    all_resolved.sort(key=lambda r: sort_key_of[r["node_id"]])
    save_dir.mkdir(parents=True, exist_ok=True)
    write_json(save_dir / "nodes_combined_raw.json", all_resolved)

    print(f"\n=== Deduplication ({len(all_resolved)} resolved nodes) ===")

    combined_rows, group_records, dedup_stats = run_dedup(
        all_resolved, sort_key_of, args, client,
    )

    write_json(save_dir / "nodes_combined.json", combined_rows)
    write_json(save_dir / "dedup_groups.json",
               [g for g in group_records if g["member_count"] >= 2])

    # ---- combined stats ----
    by_type_counts: dict[str, int] = {}
    for r in all_resolved:
        by_type_counts[r["node_type"]] = by_type_counts.get(r["node_type"], 0) + 1

    embed_tokens_new = dedup_stats["tier2_semantic"].get("embed_tokens_new", 0)
    llm_cost = total_tokens / 1_000_000 * RATE_PER_MTOK
    embed_cost = embed_tokens_new / 1_000_000 * EMBED_RATE_PER_MTOK

    combined_stats = {
        "documents_configured": len(documents),
        "documents_skipped": skipped_docs,
        "node_types": node_types,
        "nodes_resolved_total": len(all_resolved),
        "nodes_by_type": by_type_counts,
        "per_doc_type": per_doc_type_stats,
        "tokens": {"total": total_tokens, "new_this_run": new_tokens},
        # Two different models billed at two different Together rates (see
        # RATE_PER_MTOK/EMBED_RATE_PER_MTOK), so the total is a sum of two
        # separately-computed costs, not one blended rate over every token.
        "cost_breakdown_usd": {
            "llm": round(llm_cost, 4),
            "embed": round(embed_cost, 6),
            "total": round(llm_cost + embed_cost, 4),
        },
        "estimated_cost_usd": round(llm_cost + embed_cost, 4),
        "aborted_on_budget": aborted,
        "truncated": truncated,
        "dedup": dedup_stats,
    }
    write_json(save_dir / "combined_stats.json", combined_stats)

    run_config = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "embed_model": args.embed_model,
        "node_types": node_types,
        "fuzzy_threshold": args.threshold,
        "global_sections": global_sections,
        "documents": resolved_doc_configs,
        "documents_skipped": skipped_docs,
        "dedup": {
            "span_overlap_text_threshold": args.span_overlap_text_threshold,
            "dedup_threshold": args.dedup_threshold,
            "nli_entail_threshold": args.nli_entail_threshold,
            "nli_contra_threshold": args.nli_contra_threshold,
            "require_nli": args.require_nli,
            "no_nli": args.no_nli,
            "tier2_ran": dedup_stats["tier2_semantic"]["ran"],
            "nli_confirmed": dedup_stats["tier2_semantic"]["nli_confirmed"],
            "nli_skipped_reason": dedup_stats["tier2_semantic"]["nli_skipped_reason"],
        },
        "max_tokens": args.max_tokens,
        "max_tokens_budget": args.max_tokens_budget,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reproduce": (
            "python scripts/extraction-variants/epistemic_node_extraction_multidoc.py"
            + (f" --documents {args.documents}" if args.documents else "")
            + (f" --node-types {args.node_types}" if args.node_types else "")
            + (f" --sections {args.sections}" if args.sections else "")
        ),
        "note": (
            "Quotes are resolved deterministically after LLM extraction; the "
            "model is never asked for offsets. Dedup is within-node_type only "
            "(Tier 1 = same-document span overlap + text near-dup; Tier 2 = "
            "cross-document embedding + bidirectional NLI). Layout defaults "
            "(first_body_page/toc_page_*) are calibrated to eric_decision.pdf "
            "and will likely need per-document overrides for other documents."
        ),
    }
    write_json(save_dir / "run_config.json", run_config)

    print("\n=== Done ===")
    for k, v in combined_stats.items():
        if k in ("per_doc_type", "documents_skipped"):
            continue
        print(f"  {k:>22}: {v}")
    if skipped_docs:
        print(f"  {'documents_skipped':>22}: {[s['doc_id'] for s in skipped_docs]}")
    if aborted:
        print("\n  NOTE: run stopped at the token budget; results are partial.")
    if truncated:
        print(f"\n  WARNING: {len(truncated)} section(s) truncated -> "
              f"{', '.join(truncated)}")

    print(f"\n  combined raw    -> {save_dir / 'nodes_combined_raw.json'}")
    print(f"  combined (dedup)-> {save_dir / 'nodes_combined.json'}")
    print(f"  dedup groups    -> {save_dir / 'dedup_groups.json'}")
    print(f"  combined stats  -> {save_dir / 'combined_stats.json'}")
    print(f"  run config      -> {save_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
