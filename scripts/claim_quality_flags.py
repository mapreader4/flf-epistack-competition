"""Cheap, zero-LLM-cost heuristic pre-filter for two of the four qualitative
criteria in claude-docs/claims-*-qualitative-eval.md: DECONTEXTUALIZED and
ATOMIC. (COMPREHENSIVE is handled separately by scripts/coverage_gaps.py;
FAITHFUL isn't checkable from claims.json text alone.)

This is deliberately not the final verdict -- lexical heuristics are a bad
proxy for genuinely semantic criteria (see the flagged/unflagged rationale
below), so this script's only job is to shrink ~600-1,400 claims down to the
subset worth spending judge-model tokens on. Every flag here is documented in
the qualitative evals as a real observed failure pattern, not an invented
proxy:

DECONTEXTUALIZATION flags:
  - bare_pronoun_start: claim opens with an unresolved pronoun/demonstrative
    ("It...", "This...", "They..."). Directly the mechanism the evals
    documented for v1 ("I" -> "the judge" resolves reliably; other referents
    don't always).
  - bare_definite_no_anchor: claim uses a generic definite description ("the
    market", "the model", "the proposal", ...) but contains no proper-noun-
    looking token of its own -- exactly the §1.1.1 "Huanan Seafood Market" ->
    "the market" drift pattern and the §4.5.6 "Rootclaim's model" -> "the
    model" drift pattern, generalized past those two specific nouns.

ATOMICITY flags:
  - conjunction_heavy: 2+ of {" and ", " but ", ";"} -- multi-clause
    compounding, the mechanism behind the §4.5.6 claim-2 bundled-list example
    and the §1.1.1 claim-3 unrelated-"but"-clause example.
  - enumeration_language: a cardinal/quantity word ("two", "three", "several",
    ...) combined with 2+ commas -- the "three unlikely things must happen:
    ..." bundling pattern specifically.

False positives are expected and fine (e.g. "the debate" used as a genuinely
self-contained generic reference, or a legitimately single-proposition claim
that happens to use "and"). The point is recall on the known failure
mechanisms, not precision -- a judge model sorts out true/false positives
downstream at near-zero marginal cost per claim.

Usage:
    python scripts/claim_quality_flags.py
    python scripts/claim_quality_flags.py --claims artifacts/claims_improved_prompt/claims.json --out artifacts/claims_improved_prompt/quality_flags.json
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PRONOUN_START_RE = re.compile(
    r"^(it|this|that|these|those|they|he|she|its|their|his|her)\b", re.IGNORECASE
)
_GENERIC_DEFINITE_RE = re.compile(
    r"\bthe (market|model|proposal|paper|study|calculation|argument|analysis|"
    r"debate|hypothesis|author|judge|estimate|result|results|data|evidence|"
    r"finding|findings|explanation|claim|scenario|approach|method|framework|"
    r"assessment|conclusion|report|document|figure|table|section)\b",
    re.IGNORECASE,
)
_QUANTITY_WORD_RE = re.compile(
    r"\b(two|three|four|five|six|several|multiple|numerous|various)\b", re.IGNORECASE
)
_SENTENCE_INITIAL_STOPWORDS = {
    "the", "this", "that", "these", "those", "they", "he", "she", "it", "its",
    "a", "an", "and", "but", "or", "if", "when", "while", "although",
    "because", "since", "as", "for", "one", "two", "three", "four", "five",
    "several", "multiple", "also", "however", "overall", "in", "on", "at",
    "to", "of", "is", "are", "was", "were", "i", "not", "so", "yet",
}


def has_own_proper_noun(text: str) -> bool:
    """True if `text` contains a capitalized token (not counting the sentence's
    own first word, which is always capitalized regardless of content) that
    isn't a common capitalized function/quantity word -- a rough proxy for
    "this claim names its own entity rather than relying on prior context"."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    for w in words[1:]:
        if w[0].isupper() and w.lower() not in _SENTENCE_INITIAL_STOPWORDS:
            return True
    return False


def decontextualization_flags(claim_text: str) -> list[str]:
    flags = []
    if _PRONOUN_START_RE.match(claim_text.strip()):
        flags.append("bare_pronoun_start")
    if _GENERIC_DEFINITE_RE.search(claim_text) and not has_own_proper_noun(claim_text):
        flags.append("bare_definite_no_anchor")
    return flags


def atomicity_flags(claim_text: str) -> list[str]:
    flags = []
    n_and = len(re.findall(r"\band\b", claim_text, re.IGNORECASE))
    n_but = len(re.findall(r"\bbut\b", claim_text, re.IGNORECASE))
    n_semi = claim_text.count(";")
    if (n_and + n_but + n_semi) >= 2:
        flags.append("conjunction_heavy")
    n_commas = claim_text.count(",")
    if _QUANTITY_WORD_RE.search(claim_text) and n_commas >= 2:
        flags.append("enumeration_language")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default=str(ROOT / "artifacts" / "claims" / "claims.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "claims" / "quality_flags.json"))
    ap.add_argument("--sections", default=None,
                     help="comma-separated section numbers to restrict to (default: all sections in the input file) "
                          "-- for comparing pipelines on a common scope")
    args = ap.parse_args()

    claims = json.loads(Path(args.claims).read_text(encoding="utf-8"))
    if args.sections:
        wanted = {s.strip() for s in args.sections.split(",") if s.strip()}
        claims = [c for c in claims if c["section_number"] in wanted]

    # epistemic_node_extraction*.py's typed artifacts (and
    # scripts/combine_epistemic_nodes.py's combined output) share this exact
    # schema but call the two claims.json-specific fields node_id/node_text
    # instead of claim_id/claim_text -- normalize rather than forking this
    # script per schema, same fix as annotate_sections.py.
    for c in claims:
        if "claim_text" not in c and "node_text" in c:
            c["claim_text"] = c["node_text"]
            c["claim_id"] = c.get("node_id", c.get("claim_id"))

    # Preserve claims.json's own ordering per section (extraction order) so
    # the judge stage can later show "prior claims in this section" as context
    # for decontextualization -- see claude-docs note on why that context
    # matters (a claim can only be judged self-contained relative to what a
    # reader would already have seen).
    by_section: dict[str, list[dict]] = {}
    for c in claims:
        by_section.setdefault(c["section_number"], []).append(c)

    flagged = []
    for section, sec_claims in by_section.items():
        for i, c in enumerate(sec_claims):
            decon = decontextualization_flags(c["claim_text"])
            atom = atomicity_flags(c["claim_text"])
            if not decon and not atom:
                continue
            record = {
                "claim_id": c["claim_id"],
                "section_number": section,
                "position_in_section": i,
                "claim_text": c["claim_text"],
                "decontextualization_flags": decon,
                "atomicity_flags": atom,
            }
            if "node_type" in c:
                record["node_type"] = c["node_type"]
            flagged.append(record)

    out = {
        "claims_path": str(args.claims),
        "n_claims": len(claims),
        "n_flagged": len(flagged),
        "flagged": flagged,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    n_decon = sum(1 for f in flagged if f["decontextualization_flags"])
    n_atom = sum(1 for f in flagged if f["atomicity_flags"])
    print(f"{len(claims)} claims total, {len(flagged)} flagged ({100 * len(flagged) / len(claims):.1f}%)")
    print(f"  decontextualization candidates: {n_decon}")
    print(f"  atomicity candidates:           {n_atom}")

    tag_counts: dict[str, int] = {}
    for f in flagged:
        for tag in f["decontextualization_flags"] + f["atomicity_flags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\nBy flag:")
    for tag, n in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tag:24s} {n:4d}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
