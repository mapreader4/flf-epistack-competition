"""Coverage-gap detector: find stretches of each section's raw text that no
extracted claim's quote touches, without spending any LLM tokens.

This is a cheap, deterministic proxy for the "comprehensiveness" criterion in
claude-docs/claims-*-qualitative-eval.md -- those evals hand-read 4-5 sections
and found comprehensiveness failures clustered on structured content (tables,
numbered lists). This script operationalizes that check over every section a
claims.json run covers, using the same span_match.locate() tiered matcher the
rest of the pipeline uses to find each claim's quote in its section's raw text
(not the chunk-relative `span` field in claims.json -- chunk boundaries are
token-budget cuts that don't line up with a section's real paragraph/list
structure, so reassembling coverage from chunk-relative spans would need to
re-stitch chunks back together for no benefit; recover_sections() already
gives the real section text directly, same as annotate_sections.py uses).

For each section: locate every claim's quote in the raw text, merge the
resulting intervals, and report the complement (uncovered stretches) above
--min-gap-chars as candidate misses, each heuristically tagged:
  - list_item: gap starts at a numbered/bulleted list marker
  - table_row: gap has an unusually high digit density (tabular data)
  - prose: everything else (the more interesting category for follow-up --
    a prose gap wasn't caught by the two known structural failure modes)

Tags are heuristics, not ground truth -- this script is a triage tool to point
at candidate misses cheaply, not a substitute for reading the flagged spans.

Usage:
    python scripts/coverage_gaps.py
    python scripts/coverage_gaps.py --claims artifacts/claims_improved_prompt/claims.json --out artifacts/claims_improved_prompt/coverage_gaps.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "extraction-variants"))

from chunk_decision import split_sentences  # noqa: E402
from extract_claims_original import recover_sections  # noqa: E402
from span_match import locate  # noqa: E402

_LIST_MARKER_RE = re.compile(r"^(\d{1,2}[.)]|[a-zA-Z][.)]|[•‣◦⁃∙*\-])\s+")
_NUMERIC_TOKEN_RE = re.compile(r"\d[\d.,%]*")


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ivs = sorted(intervals)
    merged = [ivs[0]]
    for s, e in ivs[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def complement(merged: list[tuple[int, int]], total_len: int) -> list[tuple[int, int]]:
    gaps = []
    prev = 0
    for s, e in merged:
        if s > prev:
            gaps.append((prev, s))
        prev = max(prev, e)
    if prev < total_len:
        gaps.append((prev, total_len))
    return gaps


def strip_range(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def classify_sentence(sent: str) -> str:
    """Tag one sentence-length piece of a gap. Classifying per-sentence rather
    than over a whole (possibly 1000+ char) gap blob matters: a handful of
    footnote numbers or timestamps scattered across a long argumentative
    paragraph can push a whole-blob digit density over threshold even though
    no individual sentence is remotely table-like -- diluting the signal
    across a short sentence instead avoids that false positive."""
    stripped = sent.strip()
    if _LIST_MARKER_RE.match(stripped):
        return "list_item"
    digits = sum(ch.isdigit() for ch in stripped)
    density = digits / max(1, len(stripped))
    numeric_tokens = len(_NUMERIC_TOKEN_RE.findall(stripped))
    if density > 0.15 or numeric_tokens >= 4:
        return "table_row"
    return "prose"


def analyze_section(number: str, raw: str, claims: list[dict], threshold: float,
                     min_gap_chars: int) -> dict:
    intervals = []
    n_unlocated = 0
    for c in claims:
        m = locate(c["quote"], raw, threshold)
        if m["tier"] == "reject":
            n_unlocated += 1
            continue
        intervals.append((m["start"], m["end"]))

    merged = merge_intervals(intervals)
    covered_chars = sum(e - s for s, e in merged)
    total_chars = len(raw)

    gaps = []
    for s, e in complement(merged, total_chars):
        s, e = strip_range(raw, s, e)
        gap_text = raw[s:e]
        if len(gap_text) < min_gap_chars or not any(ch.isalnum() for ch in gap_text):
            continue

        sentences = split_sentences(gap_text) or [gap_text]
        tags = [classify_sentence(sent) for sent in sentences]
        tag_chars: dict[str, int] = {}
        for sent, tag in zip(sentences, tags):
            tag_chars[tag] = tag_chars.get(tag, 0) + len(sent)
        dominant_tag = max(tag_chars, key=tag_chars.get)

        gaps.append({
            "start": s,
            "end": e,
            "length": e - s,
            "tag": dominant_tag,
            "preview": gap_text[:160].replace("\n", " "),
            "segments": [{"tag": tag, "text": sent[:200]} for sent, tag in zip(sentences, tags)][:8],
        })

    return {
        "section_number": number,
        "total_chars": total_chars,
        "covered_chars": covered_chars,
        "coverage_ratio": round(covered_chars / total_chars, 4) if total_chars else None,
        "n_claims": len(claims),
        "n_unlocated_claims": n_unlocated,
        "gaps": gaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default=str(ROOT / "artifacts" / "claims" / "claims.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "claims" / "coverage_gaps.json"))
    ap.add_argument("--threshold", type=float, default=0.90,
                     help="min similarity (0-1) for locating a quote in the section's raw text")
    ap.add_argument("--min-gap-chars", type=int, default=40,
                     help="ignore uncovered stretches shorter than this (connective words between claims)")
    args = ap.parse_args()

    sections_meta = json.loads((ROOT / "data" / "sections.json").read_text(encoding="utf-8"))
    claims = json.loads(Path(args.claims).read_text(encoding="utf-8"))
    sections = recover_sections()

    by_section: dict[str, list[dict]] = {}
    for c in claims:
        by_section.setdefault(c["section_number"], []).append(c)

    results = []
    for meta in sections_meta:
        num = meta["number"]
        raw = sections.get(num, {}).get("raw", "")
        if not raw.strip():
            continue  # header-only section, no body text to have gaps in
        results.append(analyze_section(num, raw, by_section.get(num, []), args.threshold, args.min_gap_chars))

    results.sort(key=lambda r: (r["coverage_ratio"] if r["coverage_ratio"] is not None else 1.0))

    all_gaps = [(r["section_number"], g) for r in results for g in r["gaps"]]
    tag_counts: dict[str, int] = {}
    tag_chars: dict[str, int] = {}
    for _num, g in all_gaps:
        tag_counts[g["tag"]] = tag_counts.get(g["tag"], 0) + 1
        tag_chars[g["tag"]] = tag_chars.get(g["tag"], 0) + g["length"]

    total_chars = sum(r["total_chars"] for r in results)
    total_covered = sum(r["covered_chars"] for r in results)
    total_unlocated = sum(r["n_unlocated_claims"] for r in results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "claims_path": str(args.claims),
        "threshold": args.threshold,
        "min_gap_chars": args.min_gap_chars,
        "overall_coverage_ratio": round(total_covered / total_chars, 4) if total_chars else None,
        "tag_counts": tag_counts,
        "tag_chars": tag_chars,
        "sections": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Analyzed {len(results)} sections with body text ({len(sections_meta) - len(results)} header-only, skipped)")
    print(f"Overall coverage: {total_covered:,}/{total_chars:,} chars ({100 * total_covered / total_chars:.1f}%)")
    if total_unlocated:
        print(f"WARNING: {total_unlocated} claim quote(s) could not be located in their section's raw text at all "
              f"(threshold={args.threshold}) -- these inflate the apparent gap size; check them before trusting low-coverage sections")
    print(f"\nGaps >= {args.min_gap_chars} chars, by tag:")
    for tag in sorted(tag_counts, key=lambda t: -tag_chars[t]):
        print(f"  {tag:12s} {tag_counts[tag]:4d} gaps, {tag_chars[tag]:7,d} chars")

    print("\nWorst 10 sections by coverage ratio:")
    for r in results[:10]:
        print(f"  {r['section_number']:>6s}  coverage={r['coverage_ratio']:.2f}  "
              f"claims={r['n_claims']:3d}  gaps={len(r['gaps']):3d}  chars={r['total_chars']:5d}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
