"""Step-7 verification for the two-document graph. Runs every cross-document check in
one shot and prints a report. Read-only; safe to re-run.

    python scripts/verify_2doc.py --data-dir artifacts/epistemic_2doc
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DOC = "eric_decision.pdf"  # single-doc fallback, mirrors the query lane


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def resolve_document(node) -> str:
    for pr in node.get("provenance") or []:
        if pr.get("document"):
            return pr["document"]
    for pr in node.get("provenance") or []:
        if pr.get("chunk_id"):
            return DEFAULT_DOC
    return (node.get("meta") or {}).get("source") or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="artifacts/epistemic_2doc")
    args = ap.parse_args()
    d = Path(args.data_dir)

    nodes = load_jsonl(d / "nodes.jsonl")
    cards = load_jsonl(d / "cards.jsonl")
    scores = {s["node_id"]: s for s in load_jsonl(d / "scores.jsonl")}
    nbi = {n["node_id"]: n for n in nodes}
    cbi = {c["card_id"]: c for c in cards}
    doc = {n["node_id"]: resolve_document(n) for n in nodes}
    docs = sorted(set(doc.values()))

    print("=" * 66)
    print("STEP 7 — TWO-DOCUMENT GRAPH VERIFICATION")
    print("=" * 66)

    # (a) node count per document
    per_doc = Counter(doc.values())
    print("\n(a) nodes per document:")
    for k, v in per_doc.items():
        print(f"    {k}: {v}")

    # (b) zero nodes with missing/unknown document
    unknown = [nid for nid, dd in doc.items() if dd == "unknown"]
    print(f"\n(b) nodes with unknown/missing document: {len(unknown)}"
          f"{'  ' + str(unknown[:5]) if unknown else '  (OK: 0)'}")

    # cards + cross-document cards
    def card_doc_prem(c):
        return {doc.get(p) for p in c["premises"] if p in doc}
    cross_cards = [c for c in cards if c["premises"][0] in doc
                   and doc.get(c["target"]) and doc[c["premises"][0]] != doc.get(c["target"])]
    print(f"\n(cards) total {len(cards)}  |  by kind {dict(Counter(c['kind'] for c in cards))}"
          f"  |  by channel {dict(Counter(c['provenance'].get('channel') for c in cards))}")
    print(f"        CROSS-DOCUMENT cards (premise doc != target doc): {len(cross_cards)}")
    print("        directions:", dict(Counter(
        f"{doc[c['premises'][0]].split('_')[0]}->{doc[c['target']].split('_')[0]}"
        for c in cross_cards)))

    # (c) evidence-for on each top-level hypothesis -> does the support chain span both docs?
    def walk_support(nid, seen, depth=0, maxd=6):
        prem = []
        for cid in scores.get(nid, {}).get("in_supporters", []):
            for p in cbi[cid]["premises"]:
                prem.append(p)
                if depth < maxd and p not in seen:
                    seen.add(p); prem += walk_support(p, seen, depth + 1, maxd)
        return prem
    hyps = [n["node_id"] for n in nodes if n["type"] == "hypothesis"]
    print("\n(c) evidence-for on top-level hypotheses (support chain document span):")
    for h in hyps:
        reached = walk_support(h, {h})
        spanned = Counter(doc[p] for p in reached)
        mode = "grouped_by_document" if len(spanned) > 1 else "chain"
        star = "  <== SPANS BOTH DOCS" if len(spanned) > 1 else ""
        print(f"    {h} [{doc[h].split('_')[0]}] str={scores.get(h,{}).get('strength',0):.3f} "
              f"supporters={len(reached)} docs={dict(spanned)} mode={mode}{star}")
        print(f"        \"{nbi[h]['canonical_text'][:80]}\"")

    # (d) contested nodes spanning both documents
    contested = []
    for nid, s in scores.items():
        sup, att = s.get("in_supporters", []), s.get("in_attackers", [])
        if not (sup and att):
            continue
        sdocs = {doc[p] for c in sup for p in cbi[c]["premises"] if p in doc}
        adocs = {doc[p] for c in att for p in cbi[c]["premises"] if p in doc}
        alldocs = sdocs | adocs
        contested.append((nid, len(sup), len(att), s["vs"], s["va"], alldocs))
    contested.sort(key=lambda r: -(r[1] + r[2]))
    cross_contested = [c for c in contested if len(c[5]) > 1]
    print(f"\n(d) contested nodes: {len(contested)} total, "
          f"{len(cross_contested)} span BOTH documents")
    for nid, ns, na, vs, va, dd in cross_contested[:8]:
        w = "support" if vs > va else ("attack" if va > vs else "tie")
        print(f"    {nid} [{doc[nid].split('_')[0]}] sup={ns} att={na} winner={w} docs={sorted(x.split('_')[0] for x in dd)}")
        print(f"        \"{nbi[nid]['canonical_text'][:78]}\"")

    # (e) weakest-link (ablation) on the first top-level hypothesis
    if hyps:
        print(f"\n(e) weakest-link (top ablation deltas) for {hyps[0]}:")
        base = scores.get(hyps[0], {}).get("strength", 0)
        sup = scores.get(hyps[0], {}).get("in_supporters", []) + scores.get(hyps[0], {}).get("in_attackers", [])
        # report the direct cards + their premise text (full ablation is in query_epistemic)
        for cid in sup[:5]:
            c = cbi[cid]
            p = c["premises"][0]
            print(f"    {cid} {c['provenance'].get('relation_label')} <- {p} [{doc.get(p,'?').split('_')[0]}] "
                  f"{nbi[p]['canonical_text'][:60]}")

    print("\n" + "=" * 66)
    print(f"SUMMARY: {len(nodes)} nodes ({dict(per_doc)}), {len(cards)} cards, "
          f"{len(cross_cards)} cross-doc cards, {len(cross_contested)} cross-doc contested, "
          f"{len(hyps)} top-level hypotheses.")
    print("=" * 66)


if __name__ == "__main__":
    main()
