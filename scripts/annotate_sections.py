"""Render each section's raw (pre-chunking) text annotated with bracketed
claim markers, for manual comparison across extraction pipelines (v1, v2, ...).

Each claim's `quote` is located inside the section's raw text via
span_match.locate() -- the same tiered exact/whitespace/fuzzy matcher the
rest of the pipeline uses -- rather than trusting claims.json's chunk-relative
`span`, since chunk boundaries don't line up with offsets into the section's
raw text (chunks are token-budget slices of a whitespace-flattened
transform; see chunk_decision.py's pack_chunks()).

Writes one .txt file per section to <out-dir>/<number>.txt: the section text
with "⟦n⟧" inserted immediately before each located claim's quote, followed
by a numbered list of claim_text values in the same reading order. The
distinctive bracket glyph (rather than "[n]") is deliberate -- the source
document is full of its own citation markers like "[12]" or "[26, 39]", and a
plain "[n]" scheme is visually indistinguishable from those. Claims whose
quote can't be located above --threshold are listed separately rather than
silently dropped or mis-positioned. Fuzzy-tier matches (quote not verbatim in
the raw text, usually because PDF extraction spliced a footnote into the
middle of the source sentence) are marked with a trailing "*" and have their
start offset snapped to the nearest word boundary, since the raw fuzzy offset
can otherwise land mid-word (e.g. "ga[26]ps").

Usage:
    python scripts/annotate_sections.py
    python scripts/annotate_sections.py --claims artifacts/claims_v2/claims.json --out-dir artifacts/claims_v2/annotated
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "extraction-variants"))

from extract_claims_original import recover_sections
from span_match import locate


def _snap_to_word_start(text: str, pos: int) -> int:
    """If `pos` falls strictly inside a word, walk it back to that word's start.

    Only the fuzzy tier needs this: exact matches always land on a real
    substring boundary, but a fuzzy alignment's reported offset can fall a
    few characters into a token when the true quote isn't contiguous in the
    raw text (e.g. a footnote spliced mid-sentence by PDF extraction).
    """
    if 0 < pos < len(text) and not text[pos - 1].isspace() and not text[pos].isspace():
        while pos > 0 and not text[pos - 1].isspace():
            pos -= 1
    return pos


def annotate_section(raw: str, claims: list[dict], threshold: float) -> tuple[str, list[tuple[str, dict]], list[dict]]:
    """Return (annotated_text, [(tier, claim), ...]_in_reading_order, unlocated_claims)."""
    located = []
    unlocated = []
    for c in claims:
        m = locate(c["quote"], raw, threshold)
        if m["tier"] == "reject":
            unlocated.append(c)
        else:
            start = m["start"] if m["tier"] == "exact" else _snap_to_word_start(raw, m["start"])
            located.append((start, m["tier"], c))

    located.sort(key=lambda t: t[0])

    annotated = raw
    for i in range(len(located) - 1, -1, -1):
        start, tier, _c = located[i]
        star = "*" if tier == "fuzzy" else ""
        annotated = annotated[:start] + f"⟦{i + 1}{star}⟧ " + annotated[start:]

    return annotated, [(tier, c) for _start, tier, c in located], unlocated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default=str(ROOT / "artifacts" / "claims" / "claims.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "claims" / "annotated"))
    ap.add_argument("--threshold", type=float, default=0.90,
                     help="min similarity (0-1) for locating a quote in the section's raw text")
    args = ap.parse_args()

    sections_meta = json.loads((ROOT / "data" / "sections.json").read_text(encoding="utf-8"))
    claims = json.loads(Path(args.claims).read_text(encoding="utf-8"))
    # epistemic_node_extraction.py's typed artifacts (artifacts/{limitation,assumption,...}/*.json)
    # share this exact schema but call the two claims.json-specific fields
    # node_id/node_text instead of claim_id/claim_text -- normalize rather than
    # forking this script per schema.
    for c in claims:
        if "claim_text" not in c and "node_text" in c:
            c["claim_text"] = f"[{c.get('node_type', '?').upper()}] {c['node_text']}"
            c["claim_id"] = c.get("node_id", c.get("claim_id"))
    sections = recover_sections()

    by_section: dict[str, list[dict]] = {}
    for c in claims:
        by_section.setdefault(c["section_number"], []).append(c)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_unlocated = 0
    for meta in sections_meta:
        num = meta["number"]
        title = meta["title"]
        raw = sections.get(num, {}).get("raw", "")
        sec_claims = by_section.get(num, [])
        out_path = out_dir / f"{num.replace('/', '-')}.txt"

        if not raw.strip():
            out_path.write_text(
                f"{num}  {title}\n\n(no body text -- parent header only)\n",
                encoding="utf-8",
            )
            n_written += 1
            continue

        annotated, located_claims, unlocated_claims = annotate_section(raw, sec_claims, args.threshold)

        lines = [f"{num}  {title}", "", annotated.strip(), ""]
        if not sec_claims:
            lines.append("(no claims extracted for this section)")
        else:
            lines.append("Claims:")
            any_fuzzy = False
            for i, (tier, c) in enumerate(located_claims, start=1):
                star = "*" if tier == "fuzzy" else ""
                lines.append(f"⟦{i}{star}⟧ {c['claim_text']}")
                any_fuzzy = any_fuzzy or tier == "fuzzy"
            if any_fuzzy:
                lines.append("")
                lines.append("* = fuzzy-matched location (quote not verbatim in the raw text above); "
                              "position is approximate, snapped to the nearest word boundary")
            if unlocated_claims:
                lines.append("")
                lines.append("Unlocated (quote not found in section text above):")
                for c in unlocated_claims:
                    lines.append(f"- {c['claim_text']}  (quote: {c['quote']!r})")
                n_unlocated += len(unlocated_claims)

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_written += 1

    print(f"Wrote {n_written} section files to {out_dir}")
    if n_unlocated:
        print(f"{n_unlocated} claim(s) could not be located in their section's raw text -- see per-file 'Unlocated' notes")


if __name__ == "__main__":
    main()
