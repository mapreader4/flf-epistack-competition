"""Convert the multi-document extraction engine's deduped output (Lineage-B rows in
`nodes_combined.json`) into the graph's `epistemic_store.Node` schema (Lineage-A
`nodes.jsonl`) that pairing_funnel / score_dfquad / the query lane consume.

Lineage-B row  (from epistemic_node_extraction_multidoc.py):
  node_id="{node_type}-{md5}", node_type, document_name, section_number, chunk_id,
  node_text, quote, span, is_appendix, node_types[], dedup_group_size, dedup_tier,
  dedup_cross_document, dedup_cross_type, dedup_members[{node_id,doc_id,document_name,
  section_number,node_type,quote}]

Graph Node (epistemic_store):
  node_id="n-%05d", type in NODE_TYPES, canonical_text, fingerprint, provenance[
  {claim_id,chunk_id,section_number,quote,span,tier,document}], payload, tier, meta.

Key behaviours (see plan):
  * TYPE MAP (7 extraction types -> graph NODE_TYPES) below.
  * A dedup group (size>=2) becomes ONE node whose provenance has one entry PER MEMBER,
    each carrying its own `document` -> a cross-document merged fact keeps quotes from
    both pdfs. This is the payoff of cross-document dedup.
  * Every provenance carries `document` (asserted non-null before writing) so the query
    lane's resolve_document() never falls back to the default single document.
  * quote whitespace is collapsed (layout-mode extraction leaves padded spaces).

    python scripts/multidoc_to_graph_nodes.py \
        --in  artifacts/nodes_multidoc_2doc/nodes_combined.json \
        --out artifacts/epistemic_2doc/nodes.jsonl \
        --authors '{"eric_decision.pdf":"Eric Stansifer ...","will_decision.pdf":"Will ..."}'
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from epistemic_store import (  # noqa: E402
    Node, Provenance, node_id, fingerprint, layer_of, write_jsonl, validate_store, NODE_TYPES,
)

# 7 extraction node types -> graph NODE_TYPES (see plan §3 for justification)
TYPE_MAP = {
    "research_question": "subquestion",   # discourse question node (question layer)
    "hypothesis": "hypothesis",           # 1:1
    "evidence": "evidence",               # 1:1 (data)
    "analysis": "claim",                  # inference/interpretation -> argument-layer claim
    "quantitative_result": "estimate",    # a claim whose content is a number (data) + payload.number
    "assumption": "assumption",           # 1:1 (argument)
    "limitation": "rebuttal",             # caveat/weakness acts as an attacker (argument)
}
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")

# --- hypothesis demotion (defect 3a, option i) --------------------------------------
# The extractor over-tags speculative sub-claims / mechanisms / conditionals as
# `hypothesis` (e.g. 260 for eric). A document has only ~2-5 genuinely COMPETING
# top-level hypotheses it is trying to adjudicate between. Rather than a fragile regex,
# ONE gpt-oss call per document reads every hypothesis statement and returns the
# indices of the top-level ones; everything else tagged hypothesis -> claim. The LLM
# understands context ("zoonotic origin vs lab leak" is top-level, "aerosolization could
# have occurred" is a sub-scenario) and generalizes to any topic (eggs, LHC, ...) with
# no pattern changes. Temperature 0 + on-disk cache => reproducible and ~free on re-run.
HYP_SELECT_PROMPT_VERSION = "hyp-select-v1"
_HYP_SYSTEM = (
    "You identify the genuine top-level competing hypotheses in a single document. "
    "Most documents have only 2-5 top-level hypotheses that the document as a whole is "
    "trying to adjudicate between; everything else tagged as a hypothesis is really a "
    "sub-scenario, mechanism, conditional, or supporting sub-claim UNDER one of them."
)
_HYP_USER = """\
Below are {n} statements that an extractor tagged as "hypothesis" in ONE document.
Return ONLY the indices of the GENUINE TOP-LEVEL COMPETING hypotheses -- the small set
(usually 2-5) of rival explanations the document as a whole is deciding between. Do NOT
include sub-scenarios, mechanisms, conditional/"if" claims, requirements, probability
sub-estimates, or supporting sub-arguments under a top-level hypothesis.

Return ONLY JSON of the exact form: {{"top_level": [0, 3, 7]}}

STATEMENTS:
{listing}
"""


def select_top_level_hypotheses(rows, *, model, cache_path, client=None,
                                no_llm=False) -> set[str]:
    """One gpt-oss call PER DOCUMENT -> set of lineage-B node_ids to keep as hypothesis.
    Falls back to keeping ALL hypotheses (no demotion) when --no-llm or on any failure --
    never silently demotes the core hypotheses. Cached by (doc, prompt_version, ids)."""
    from collections import defaultdict
    by_doc = defaultdict(list)
    for r in rows:
        if r.get("node_type") == "hypothesis":
            by_doc[r["document_name"]].append(r)
    if not by_doc:
        return set()
    if no_llm:
        print("  --no-llm: keeping ALL hypothesis nodes (no demotion)")
        return {r["node_id"] for rs in by_doc.values() for r in rs}

    cache = {}
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    if client is None:
        client = _build_client()

    keep: set[str] = set()
    for doc, rs in by_doc.items():
        ids = [r["node_id"] for r in rs]
        key = f"{HYP_SELECT_PROMPT_VERSION}|{model}|{doc}|" + "|".join(sorted(ids))
        if key in cache:
            picked = cache[key]
        else:
            listing = "\n".join(f"[{i}] {r['node_text']}" for i, r in enumerate(rs))
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=4000,
                    messages=[{"role": "system", "content": _HYP_SYSTEM},
                              {"role": "user", "content": _HYP_USER.format(
                                  n=len(rs), listing=listing)}])
                content = resp.choices[0].message.content or ""
                s, e = content.find("{"), content.rfind("}")
                data = json.loads(content[s:e + 1])
                idxs = [int(x) for x in data.get("top_level", []) if isinstance(x, (int, float))]
                picked = [ids[j] for j in idxs if 0 <= j < len(ids)]
            except Exception as ex:
                print(f"  WARNING [{doc}]: hypothesis-selection call failed ({ex}); "
                      f"keeping ALL {len(rs)} as hypothesis for this doc")
                picked = ids
            cache[key] = picked
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))
        keep.update(picked)
        print(f"  [{doc}] top-level hypotheses: kept {len(picked)}/{len(rs)}")
    return keep


def _build_client():
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY missing (needed for hypothesis selection; or pass --no-llm).")
    return OpenAI(base_url="https://api.together.xyz/v1", api_key=key, max_retries=3, timeout=300)


def norm_ws(s: str | None) -> str | None:
    """Collapse runs of whitespace (layout-mode extraction pads with alignment spaces)."""
    if not s:
        return s
    return re.sub(r"\s+", " ", s).strip()


def prov_from_member(m: dict, *, representative_row: dict | None) -> Provenance:
    """Build a Provenance from a dedup member. The representative member also has
    chunk_id + span (from the row); other members carry only quote/section/document."""
    if representative_row is not None:  # the representative -> full grounding
        span = representative_row.get("span")
        return Provenance(
            claim_id=representative_row["node_id"],
            chunk_id=representative_row.get("chunk_id"),
            section_number=representative_row.get("section_number"),
            quote=norm_ws(representative_row.get("quote")),
            span=span,
            tier="T1" if span else "T2",
            document=representative_row["document_name"],
        )
    return Provenance(
        claim_id=m.get("node_id"),
        chunk_id=None,
        section_number=m.get("section_number"),
        quote=norm_ws(m.get("quote")),
        span=None,
        tier="T2",
        document=m["document_name"],
    )


def convert(rows: list[dict], authors: dict[str, str], keep_hyp_ids: set[str]) -> list[Node]:
    nodes: list[Node] = []
    unknown_types: Counter = Counter()
    demoted = 0
    for i, r in enumerate(rows):
        ntype = TYPE_MAP.get(r["node_type"])
        if ntype is None:
            unknown_types[r["node_type"]] += 1
            ntype = "claim"
        text = r["node_text"]

        # defect-3a demotion: hypotheses NOT chosen as top-level by the LLM -> claim
        demoted_hyp = False
        if ntype == "hypothesis" and r["node_id"] not in keep_hyp_ids:
            ntype = "claim"
            demoted_hyp = True
            demoted += 1

        # provenance: representative first, then one entry per OTHER dedup member
        rep_doc = r["document_name"]
        provenance = [prov_from_member(None, representative_row=r)]
        members = r.get("dedup_members") or []
        for m in members:
            if m.get("node_id") == r["node_id"]:
                continue  # representative already added
            if not m.get("document_name"):
                continue  # never emit a null-document provenance
            provenance.append(prov_from_member(m, representative_row=None))

        payload = {"attribution": authors.get(rep_doc, rep_doc)}
        if ntype == "estimate":
            m = _NUMBER_RE.search(text or "")
            if m:
                payload["number"] = m.group(0)

        span = r.get("span")
        meta = {
            "layer": layer_of(ntype),
            "source": rep_doc,
            "type_source": "multidoc-extraction+deterministic-map",
            "extraction_type": r["node_type"],
            "node_types": r.get("node_types"),
            "dedup_tier": r.get("dedup_tier"),
            "dedup_group_size": r.get("dedup_group_size", 1),
            "dedup_cross_document": r.get("dedup_cross_document", False),
            "dedup_cross_type": r.get("dedup_cross_type", False),
            "is_appendix": r.get("is_appendix", False),
            "demoted_from_hypothesis": demoted_hyp,
        }
        nodes.append(Node(
            node_id=node_id(i), type=ntype, canonical_text=text,
            fingerprint=fingerprint(text), provenance=provenance,
            payload=payload, tier="T1" if span else "T2", meta=meta,
        ))
    if unknown_types:
        print(f"  NOTE: {sum(unknown_types.values())} rows had an unmapped node_type "
              f"-> defaulted to 'claim': {dict(unknown_types)}")
    kept_hyp = sum(1 for n in nodes if n.type == "hypothesis")
    print(f"  hypothesis demotion (defect 3a): kept {kept_hyp} as hypothesis, "
          f"demoted {demoted} -> claim")
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="nodes_combined.json from the multidoc engine")
    ap.add_argument("--out", required=True, help="graph nodes.jsonl to write")
    ap.add_argument("--authors", default="{}", help='JSON map document_name -> attribution string')
    ap.add_argument("--model", default="openai/gpt-oss-120b", help="model for top-level hypothesis selection")
    ap.add_argument("--hyp-cache", default=str(ROOT / "outputs" / "epistemic_2doc" / "hyp_select_cache.json"))
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM hypothesis selection (keep all hypotheses)")
    args = ap.parse_args()

    rows = json.loads(Path(args.inp).read_text())
    authors = json.loads(args.authors)

    # defect-3a option (i): one gpt-oss call per document picks the top-level hypotheses
    keep_hyp_ids = select_top_level_hypotheses(
        rows, model=args.model, cache_path=args.hyp_cache, no_llm=args.no_llm)
    nodes = convert(rows, authors, keep_hyp_ids)

    # --- correctness invariant: every provenance carries a document -------------------
    missing = [n.node_id for n in nodes if any(not p.document for p in n.provenance)]
    if missing:
        sys.exit(f"ABORT: {len(missing)} nodes have a provenance with no document "
                 f"(resolve_document would mis-attribute them). e.g. {missing[:5]}")

    errs = validate_store(nodes, [])
    if errs:
        print("  VALIDATION ERRORS:", errs[:10])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_written = write_jsonl(args.out, nodes)

    # --- summary ---------------------------------------------------------------------
    def doc_of(n):  # node-level document = first provenance (mirrors resolve_document)
        return n.provenance[0].document
    per_doc = Counter(doc_of(n) for n in nodes)
    per_type = Counter(n.type for n in nodes)
    cross = sum(1 for n in nodes if len({p.document for p in n.provenance}) > 1)
    merged = sum(1 for n in nodes if n.meta.get("dedup_group_size", 1) >= 2)
    print(f"\nwrote {n_written} nodes -> {args.out}")
    print(f"  per document (node-level): {dict(per_doc)}")
    print(f"  per type: {dict(per_type)}")
    print(f"  merged (dedup group>=2): {merged}   cross-document provenance: {cross}")
    print(f"  nodes with a missing provenance.document: 0 (asserted)")


if __name__ == "__main__":
    main()
