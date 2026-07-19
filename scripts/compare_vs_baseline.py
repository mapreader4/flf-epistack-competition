"""Head-to-head: graph-guided answer vs a naive direct-LLM call on the same file.

Tests design.md's core claim (lines 245-277): the graph tells the LLM what to focus
on and where the conflicts are, vs a naive call that gets the whole document and
guesses what matters. Both use the SAME model (gpt-oss-120b) and question; the only
difference is the context.

  BASELINE : gpt-oss( full document text + question )              — no structure
  GRAPH    : gpt-oss( question + our ranked evidence chain
                      + the ablation-identified weakest link )     — graph-guided

Writes artifacts/epistemic/query_examples/comparison.md.

    python scripts/compare_vs_baseline.py
    python scripts/compare_vs_baseline.py --target n-00502 --no-llm   # show inputs only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import query_epistemic as q  # reuse the query lane's loaders + traversals

ART = ROOT / "artifacts" / "epistemic"
DOC = ROOT / "data" / "eric_decision_raw.txt"
MODEL = "openai/gpt-oss-120b"

QUESTION = ("What is the strongest evidence that the COVID-19 outbreak originated at "
            "the Huanan Seafood Market, and what is the single weakest link in that "
            "argument?")


def build_graph_context(target: str) -> dict:
    nodes = q.load_jsonl(ART / "nodes.jsonl")
    cards = q.load_jsonl(ART / "cards.jsonl")
    scores = q.load_jsonl(ART / "scores.jsonl")
    nbi = {n["node_id"]: n for n in nodes}
    cbi = {c["card_id"]: c for c in cards}
    sbi = {s["node_id"]: s for s in scores}

    ev = q.evidence_for(target, nbi, sbi, cbi)
    wl = q.weakest_link(target, nodes, cards, nbi, cbi, top_n=3)

    def flat(entry, depth=0, out=None):
        out = out if out is not None else []
        out.append({"node_id": entry["node_id"], "strength": round(entry.get("strength") or 0, 3),
                    "section": (entry.get("provenance") or [{}])[0].get("section_number"),
                    "quote": (entry.get("provenance") or [{}])[0].get("quote"),
                    "text": entry["text"]})
        for c in entry.get("children", []):   # no cap: evidence-for queries want the full connected chain
            flat(c, depth + 1, out)
        return out

    evidence = []
    for child in ev.get("children", []):       # all direct supporters, not just the top 5
        evidence.extend(flat(child))
    # dedup by node_id (a node reachable via multiple parents appears once), keep first/highest-ranked occurrence
    seen, deduped = set(), []
    for e in evidence:
        if e["node_id"] in seen:
            continue
        seen.add(e["node_id"])
        deduped.append(e)
    return {
        "hypothesis": {"node_id": target, "text": nbi[target]["canonical_text"],
                       "strength": round(sbi[target]["strength"], 3)},
        "ranked_evidence": deduped,            # uncapped: all reachable evidence reaches the prose prompt
        "weakest_link_by_ablation": [
            {"card_id": w["card_id"], "delta": round(w["delta"], 4), "kind": w["kind"],
             "premise": w["premises"][0]["text"] if w["premises"] else None} for w in wl],
    }


def ask(client, model, system, user, max_tokens):
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=None,
                    help="entry node; if omitted, resolved from the question (Step-9 mapping)")
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-llm", action="store_true", help="show the two contexts, skip API calls")
    ap.add_argument("--out", default=str(ART / "query_examples" / "comparison.md"))
    args = ap.parse_args()

    if not args.target:
        import resolve_question
        hits = resolve_question.resolve(args.question, top_n=1)
        args.target = hits[0][0]
        print(f"resolved question -> {args.target} (cos {hits[0][1]:.3f}): {hits[0][2][:70]}")
    ctx = build_graph_context(args.target)
    doc = DOC.read_text()

    graph_user = (f"Question: {args.question}\n\n"
                  "Structured argument-graph result (evidence already ranked by DF-QuAD "
                  "strength; the weakest link was found by removing each card and "
                  "re-scoring — largest delta = most load-bearing):\n"
                  f"{json.dumps(ctx, indent=2)}")
    graph_sys = ("Answer ONLY from the structured result. Cite section numbers and quote "
                 "verbatim text where given. State the weakest link explicitly. Do not add "
                 "anything not present in the structured result.")
    base_user = f"DOCUMENT:\n{doc}\n\nQuestion: {args.question}"
    base_sys = ("You are given a document. Answer the question using it. Cite section "
                "numbers where possible.")

    if args.no_llm:
        print("=== GRAPH context (tokens ~%d) ===" % (len(graph_user) // 4))
        print(graph_user[:2000])
        print("\n=== BASELINE gets the full document (tokens ~%d) ===" % (len(base_user) // 4))
        return

    client = q.build_client()
    print(f"Model: {args.model}  |  target: {args.target}\n")
    print("… baseline (full document, no structure)")
    base_ans = ask(client, args.model, base_sys, base_user, max_tokens=1200)
    print("… graph-guided (ranked evidence + ablation)")
    graph_ans = ask(client, args.model, graph_sys, graph_user, max_tokens=1200)

    md = (f"# Comparison — graph-guided vs naive direct LLM\n\n"
          f"*Model: {args.model} (both). Question and file identical; only the context differs.*\n\n"
          f"**Question:** {args.question}\n\n"
          f"**Target hypothesis:** `{args.target}` — \"{ctx['hypothesis']['text']}\" "
          f"(graph strength {ctx['hypothesis']['strength']})\n\n"
          f"## A. Naive baseline — gpt-oss with the full document ({len(doc)//4}k tokens), no structure\n\n"
          f"{base_ans}\n\n"
          f"## B. Graph-guided — gpt-oss with ranked evidence + ablation weakest-link\n\n"
          f"{graph_ans}\n\n"
          f"## The graph-supplied structure (what B had that A didn't)\n\n"
          f"```json\n{json.dumps(ctx, indent=2)}\n```\n")
    Path(args.out).write_text(md)
    print(f"\nwrote -> {args.out}")
    print("\n" + "=" * 70 + "\nBASELINE:\n" + base_ans[:1500])
    print("\n" + "=" * 70 + "\nGRAPH-GUIDED:\n" + graph_ans[:1500])


if __name__ == "__main__":
    main()
