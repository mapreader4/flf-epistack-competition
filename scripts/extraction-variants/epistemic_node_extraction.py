#!/usr/bin/env python3
### written by Arpita Saha - saha119@purdue.edu
"""
Extract epistemic nodes with provenance from curated sections of eric_decision.pdf.

This is a generalized version of the earlier claim-extraction pipeline. It can
extract the following node types independently:

    research_question
    hypothesis
    evidence
    analysis
    quantitative_result
    assumption
    limitation

The pipeline is:

    section text
        -> LLM extracts {node_text, quote}
        -> deterministic section-level quote matching
        -> deterministic chunk attribution
        -> provenance-grounded JSON records

The model is never asked for character offsets. It only returns a
decontextualized node and one verbatim supporting quotation. Exact/fuzzy string
matching resolves provenance afterward.

This script uses Together AI through its OpenAI-compatible API.

Examples
--------
Extract research questions:

    python scripts/extract_epistemic_nodes.py \
        --node-type research_question \
        --sections 7,7.1,7.2

Extract hypotheses:

    python scripts/extract_epistemic_nodes.py \
        --node-type hypothesis \
        --sections 7,7.1,7.2

Extract evidence:

    python scripts/extract_epistemic_nodes.py \
        --node-type evidence \
        --sections 7,7.1,7.2

Dependencies
------------
This script expects the existing project modules:

    scripts/chunk_decision.py
    scripts/span_match.py
    data/chunks.json
    eric_decision.pdf

Environment
-----------
Create a .env file at the project root containing:

    TOGETHER_API_KEY=...
    TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "scripts"))

from chunk_decision import (  # noqa: E402
    build_body,
    extract_pages,
    locate_headings,
    parse_toc,
)
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

PROMPT_VERSION = "epistemic-nodes-v1"

# Used only for a friendly cost estimate. Together's listed rate for
# meta-llama/Llama-3.3-70B-Instruct-Turbo as of 2026-07-19 (together.ai/pricing).
RATE_PER_MTOK = 1.04


# ---------------------------------------------------------------------------
# Default section sample
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


# ---------------------------------------------------------------------------
# Node-type configuration
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
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise epistemic-node extraction assistant for an "
    "argument-mining and provenance-indexing pipeline over a public, "
    "judicial-style decision document. Your only job is to locate and "
    "attribute the requested epistemic nodes, pairing each node with a "
    "verbatim supporting quotation copied from the text. You do not evaluate, "
    "endorse, correct, or add information of your own."
)


USER_TEMPLATE = """\
Document: "eric_decision.pdf" -- a judge's written decision in a structured
public debate about COVID-19 origins, weighing competing hypotheses under
explicit Bayesian argument.

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
- Each extracted node must be ATOMIC: represent only one question,
  proposition, observation, inference, result, assumption, or limitation.
- Each node must be DECONTEXTUALIZED: resolve pronouns, bridging references,
  abbreviations, and omitted subjects when necessary so the node stands alone.
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
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
    )

    usage = response.usage

    return {
        "content": response.choices[0].message.content or "",
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        },
    }


# ---------------------------------------------------------------------------
# Tolerant JSON parsing
# ---------------------------------------------------------------------------

def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _unescape_lenient(s: str) -> str:
    """Undo the handful of JSON escapes the model actually uses, for text
    captured by regex rather than a real JSON parser. Protecting "\\\\" first
    keeps a literal backslash from being mistaken for the start of one of the
    other escapes handled below."""
    s = s.replace("\\\\", "\x00")
    s = s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    return s.replace("\x00", "\\")


def _regex_extract_pairs(text: str, text_field: str) -> list[dict[str, str]]:
    """Fallback for when the model's JSON has an unescaped inner quote inside
    a <text_field>/quote string (breaks json.loads, and the same bad
    character survives into every {...}/[...] substring re-search too).
    Matches on the literal "<text_field>": "...", "quote": "..." } field
    structure instead of requiring the whole response to be valid JSON:
    regex backtracking means a "\"" that isn't immediately followed by the
    next field's exact delimiter doesn't end the match early."""
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


def parse_nodes(
    text: str,
    node_type: str,
) -> tuple[list[dict[str, str]], int]:
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
        # Strict JSON parsing failed everywhere -- fall back to a structural
        # match on the known text_field/quote pattern instead of dropping the
        # whole response's nodes silently.
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
    section_number: str,
    section_raw: str,
    section_chunks: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    """Resolve one epistemic node to a section-level and chunk-level span."""

    section_match = locate(
        pair["quote"],
        section_raw,
        threshold,
    )

    if section_match["tier"] == "reject":
        return None

    matched_text = section_raw[
        section_match["start"]:section_match["end"]
    ]

    best_chunk = None
    best_chunk_match = None
    best_key = None

    for chunk in sorted(
        section_chunks,
        key=lambda item: item["chunk_order_in_section"],
    ):
        chunk_result = locate(
            matched_text,
            chunk["text"],
            threshold,
        )

        if chunk_result["tier"] == "reject":
            continue

        key = (
            chunk_result["tier"] == "exact",
            chunk_result["score"],
        )

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
        span = [
            best_chunk_match["start"],
            best_chunk_match["end"],
        ]

    node_type = pair["node_type"]

    fingerprint = (
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
# Section-text recovery
# ---------------------------------------------------------------------------

def recover_sections() -> dict[str, dict[str, str]]:
    """Recover section title and raw text using the existing parser."""

    pages = extract_pages()
    toc = parse_toc(pages)
    body, _offsets = build_body(pages, first_body_page=4)
    sections = locate_headings(body, toc)

    recovered: dict[str, dict[str, str]] = {}

    for section in sections:
        raw = body[section["start"]:section["end"]].strip()

        recovered[section["number"]] = {
            "title": section["title"],
            "raw": raw,
        }

    return recovered


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


def make_cache_key(
    section_number: str,
    node_type: str,
    model: str,
    section_raw: str,
) -> str:
    """Include a section-text hash so changed text invalidates old cache entries."""

    raw_hash = hashlib.md5(section_raw.encode("utf-8")).hexdigest()

    return (
        f"{section_number}|"
        f"{node_type}|"
        f"{PROMPT_VERSION}|"
        f"{model}|"
        f"{raw_hash}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--node-type",
        required=True,
        choices=sorted(NODE_CONFIGS),
        help="Epistemic node type to extract.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Together-hosted model identifier.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Minimum fuzzy-match similarity in the range [0, 1].",
    )

    parser.add_argument(
        "--sections",
        default=",".join(DEFAULT_SECTIONS),
        help="Comma-separated section numbers.",
    )

    parser.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Output directory. Defaults to artifacts/<node_type> "
            "under the project root."
        ),
    )

    parser.add_argument(
        "--cache",
        default=None,
        help=(
            "LLM response cache path. Defaults to "
            "outputs/<node_type>/llm_cache.json."
        ),
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cached LLM responses.",
    )

    parser.add_argument(
        "--max-tokens-budget",
        type=int,
        default=200_000,
        help="Abort after spending this many new tokens.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    node_type = args.node_type
    config = NODE_CONFIGS[node_type]

    requested_sections = [
        section.strip()
        for section in args.sections.split(",")
        if section.strip()
    ]

    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else ROOT / "artifacts" / node_type
    )

    cache_path = (
        Path(args.cache)
        if args.cache
        else ROOT / "outputs" / node_type / "llm_cache.json"
    )

    print(f"Node type:  {node_type}")
    print(f"Model:      {args.model} @ Together")
    print(f"Threshold:  fuzzy >= {args.threshold}")
    print(
        f"Sections:   {len(requested_sections)} -> "
        f"{', '.join(requested_sections)}"
    )
    print(f"Save dir:   {save_dir}")
    print(f"Cache:      {cache_path}")
    print(f"Budget:     {args.max_tokens_budget:,} new tokens\n")

    client = build_client()
    preflight(client, args.model)

    print("\n=== Recovering per-section raw text ===")
    sections = recover_sections()
    print(f"  recovered {len(sections)} sections from the PDF")

    chunks_path = ROOT / "data" / "chunks.json"

    if not chunks_path.exists():
        sys.exit(f"Missing required file: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    chunks_by_section: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        chunks_by_section.setdefault(
            chunk["section_number"],
            [],
        ).append(chunk)

    cache: dict[str, Any] = {}

    if cache_path.exists() and not args.no_cache:
        cache = load_json_object(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== LLM extraction ===")

    raw_pairs: list[dict[str, Any]] = []
    invalid_total = 0
    new_tokens = 0
    total_tokens = 0
    aborted = False
    sections_processed: set[str] = set()

    for number in requested_sections:
        section = sections.get(number)

        if section is None:
            print(f"  [{number}] not found -- skipping")
            continue

        if not section["raw"]:
            print(f"  [{number}] empty section -- skipping")
            continue

        key = make_cache_key(
            section_number=number,
            node_type=node_type,
            model=args.model,
            section_raw=section["raw"],
        )

        cached = key in cache and not args.no_cache

        if cached:
            entry = cache[key]
            total_tokens += entry["usage"]["total_tokens"]
        else:
            if new_tokens >= args.max_tokens_budget:
                print(
                    f"  [{number}] token budget reached "
                    f"({new_tokens:,} new tokens); saving partial results"
                )
                aborted = True
                break

            entry = call_llm(
                client=client,
                model=args.model,
                node_type=node_type,
                number=number,
                title=section["title"],
                raw=section["raw"],
            )

            cache[key] = entry
            write_json(cache_path, cache)

            new_tokens += entry["usage"]["total_tokens"]
            total_tokens += entry["usage"]["total_tokens"]

        pairs, invalid_count = parse_nodes(
            entry["content"],
            node_type,
        )

        invalid_total += invalid_count
        sections_processed.add(number)

        for pair in pairs:
            raw_pairs.append(
                {
                    "section_number": number,
                    **pair,
                }
            )

        label = config["plural"].replace("_", " ")
        cache_status = (
            "cached"
            if cached
            else f"{entry['usage']['total_tokens']:,} tok"
        )
        estimated_cost = total_tokens / 1_000_000 * RATE_PER_MTOK

        invalid_suffix = (
            f", {invalid_count} invalid"
            if invalid_count
            else ""
        )

        print(
            f"  [{number}] {len(pairs):>3} {label}"
            f"{invalid_suffix} "
            f"({cache_status}; cumulative {total_tokens:,} tok "
            f"~= ${estimated_cost:.4f})"
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    plural_name = config["plural"]

    raw_output_path = save_dir / f"{plural_name}_raw.json"
    resolved_output_path = save_dir / f"{plural_name}.json"
    dropped_output_path = save_dir / "dropped_pairs.json"
    stats_output_path = save_dir / f"{plural_name}_stats.json"
    run_config_output_path = save_dir / "run_config.json"

    write_json(raw_output_path, raw_pairs)

    print("\n=== Deterministic span resolution ===")

    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    section_exact = 0
    section_fuzzy = 0
    chunk_exact = 0
    chunk_fuzzy = 0
    chunk_split = 0

    for pair in raw_pairs:
        number = pair["section_number"]

        record = resolve_node(
            pair=pair,
            section_number=number,
            section_raw=sections[number]["raw"],
            section_chunks=chunks_by_section.get(number, []),
            threshold=args.threshold,
        )

        if record is None:
            dropped.append(pair)
            continue

        records.append(record)

        if record["section_match_tier"] == "exact":
            section_exact += 1
        else:
            section_fuzzy += 1

        if record["chunk_match"] == "exact":
            chunk_exact += 1
        elif record["chunk_match"] == "fuzzy":
            chunk_fuzzy += 1
        else:
            chunk_split += 1

    write_json(resolved_output_path, records)
    write_json(dropped_output_path, dropped)

    extracted_count = len(raw_pairs)
    dropped_count = len(dropped)

    drop_rate = (
        round(dropped_count / extracted_count, 4)
        if extracted_count
        else 0.0
    )

    stats = {
        "node_type": node_type,
        "sections_requested": len(requested_sections),
        "sections_processed": len(sections_processed),
        "nodes_extracted": extracted_count,
        "invalid_pairs": invalid_total,
        "nodes_resolved": len(records),
        "nodes_dropped": dropped_count,
        "drop_rate": drop_rate,
        "section_match": {
            "exact": section_exact,
            "fuzzy": section_fuzzy,
        },
        "chunk_match": {
            "exact": chunk_exact,
            "fuzzy": chunk_fuzzy,
            "split_across_chunks": chunk_split,
        },
        "tokens": {
            "total": total_tokens,
            "new_this_run": new_tokens,
        },
        "estimated_cost_usd": round(
            total_tokens / 1_000_000 * RATE_PER_MTOK,
            4,
        ),
        "aborted_on_budget": aborted,
    }

    write_json(stats_output_path, stats)

    run_config = {
        "node_type": node_type,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "fuzzy_threshold": args.threshold,
        "sections": requested_sections,
        "extraction_unit": "section",
        "max_tokens_budget": args.max_tokens_budget,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reproduce": (
            "python scripts/extract_epistemic_nodes.py "
            f"--node-type {node_type} "
            f"--sections {args.sections} "
            f"--threshold {args.threshold}"
        ),
        "note": (
            "Quotes are resolved deterministically after LLM extraction. "
            "The model is never asked for offsets. Span values are "
            "chunk-relative [start, end)."
        ),
    }

    write_json(run_config_output_path, run_config)

    print("\n=== Done ===")

    for key, value in stats.items():
        print(f"  {key:>20}: {value}")

    if aborted:
        print(
            "\n  NOTE: The run stopped at the token budget. "
            "The output contains partial results."
        )

    print(f"\n  raw output      -> {raw_output_path}")
    print(f"  resolved output -> {resolved_output_path}")
    print(f"  dropped pairs   -> {dropped_output_path}")
    print(f"  statistics      -> {stats_output_path}")
    print(f"  run config      -> {run_config_output_path}")


if __name__ == "__main__":
    main()