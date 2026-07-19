"""Combine all seven extracted epistemic node types -- full 64-section corpus,
improved-prompt run (epistemic_node_extraction_improved_prompt.py) -- into one
JSON file, with no claims.json content mixed in.

Each node's `node_text` is already an atomic, decontextualized, self-contained
sentence (that's what the DECONTEXTUALIZED/ATOMIC prompt rules are for), which
makes it a cleaner unit than a whole chunk of raw prose to hand to HippoRAG:
scripts/run_hipporag_index.py currently calls hipporag.index(docs=...) with
each chunk's full (pronoun-laden, multi-sentence) text as one passage. A list
of `node_text` values from this combined file is a drop-in alternative `docs`
list -- one passage per extracted fact/question/hypothesis/etc. instead of one
passage per ~300-token chunk.

Node types are unioned, not deduplicated across type -- the same sentence can
occasionally be extracted (independently, sometimes with slightly different
wording) by two different type-specific prompts (e.g. a number-bearing
sentence extracted as both `quantitative_result` and `evidence`). No attempt
is made to resolve that here; downstream consumers that care should dedupe on
(section_number, quote) or an embedding-similarity pass.

Output goes under artifacts/nodes_combined/, not artifacts/epistemic/ -- the
latter already holds a separate, unrelated structure-layer prototype
(candidate_pairs.json, cards.jsonl, nodes.jsonl, scores.jsonl); this script's
output is not part of that.

--suffix selects which run to combine: "_improved_prompt" (default) combines
epistemic_node_extraction_improved_prompt.py's full-64-section outputs
(artifacts/<type>_improved_prompt/); "" combines Arpita's original
epistemic_node_extraction.py outputs (artifacts/<type>/, curated 12 sections
only -- research_question has no resolved file there and is skipped with a
warning, since that run never finished its span-resolution step).

Usage:
    python scripts/combine_epistemic_nodes.py
    python scripts/combine_epistemic_nodes.py --suffix "" --out artifacts/nodes_combined/nodes_original.json
    python scripts/annotate_sections.py --claims artifacts/nodes_combined/nodes.json --out-dir artifacts/nodes_combined/annotated
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# type -> resolved-output filename (matches NODE_CONFIGS["plural"] in
# epistemic_node_extraction*.py -- not re-imported here to avoid depending on
# either extraction script's module-level side effects at import time).
PLURAL_FILENAMES = {
    "research_question": "research_questions.json",
    "hypothesis": "hypotheses.json",
    "evidence": "evidence_items.json",
    "analysis": "analyses.json",
    "quantitative_result": "quantitative_results.json",
    "assumption": "assumptions.json",
    "limitation": "limitations.json",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="_improved_prompt",
                     help='artifact-dir suffix, e.g. "_improved_prompt" (default) or "" for the original run')
    ap.add_argument("--out", default=None,
                     help="output path (default: artifacts/nodes_combined/nodes.json, "
                          "or nodes_original.json if --suffix is empty)")
    args = ap.parse_args()

    combined: list[dict] = []
    n_types_found = 0
    for node_type, filename in PLURAL_FILENAMES.items():
        path = ROOT / "artifacts" / f"{node_type}{args.suffix}" / filename
        if not path.exists():
            print(f"  skip {node_type}: {path.relative_to(ROOT)} not found")
            continue
        nodes = json.loads(path.read_text(encoding="utf-8"))
        combined.extend(nodes)
        n_types_found += 1
        print(f"  {node_type:20s} {len(nodes):4d} nodes  <- {path.relative_to(ROOT)}")

    # Stable, readable ordering: by section (document order via sections.json),
    # then node type, then original extraction order.
    sections_meta = json.loads((ROOT / "data" / "sections.json").read_text(encoding="utf-8"))
    section_order = {m["number"]: i for i, m in enumerate(sections_meta)}
    combined.sort(key=lambda n: (section_order.get(n["section_number"], 999), n["node_type"]))

    default_name = "nodes.json" if args.suffix == "_improved_prompt" else "nodes_original.json"
    out_path = Path(args.out) if args.out else ROOT / "artifacts" / "nodes_combined" / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    by_type: dict[str, int] = {}
    for n in combined:
        by_type[n["node_type"]] = by_type.get(n["node_type"], 0) + 1

    print(f"\n{len(combined)} nodes combined across {n_types_found} types")
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {t:20s} {n:4d}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
