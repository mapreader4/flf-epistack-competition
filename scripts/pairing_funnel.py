"""Step C — the pairing funnel (design.md step 6), redesigned for candidate QUALITY.

The bottleneck was never budget size — it was candidate *selection*. A pure bottom-up
"pair everything, filter, send survivors" funnel wastes most of the LLM budget on
topically-similar-but-argumentatively-unrelated pairs (old hit rate ~29.5%). This
version splits the budget so most pairs are ones that actually produce cards:

  50%  TOP-DOWN   for each question/hypothesis node, its k nearest data/evidence
                  nodes by embedding — pairing evidence with the question it is ABOUT.
                  Highest hit rate: you are pairing evidence with its hypothesis.
  30%  data → argument      (bottom-up, cosine-ranked)
  15%  argument → question  (bottom-up, cosine-ranked)
   5%  CONTRADICTIONS        data↔data / argument↔argument, but ONLY when the DeBERTa
                  NLI contradiction score is high (cosine cannot tell "agree" from
                  "contradict"). Skipped unless NLI is enabled (needs a GPU) — its
                  budget is redistributed to top-down otherwise.

Blocking rules: question→question is never paired; data→data / argument→argument only
enter through the NLI-gated contradiction channel. No leakage guard here anymore — the
conclusory/evidentiary holdout is retired; every node (including the judge's own
conclusions) is in the pool and gets attributed to its speaker instead (attribute_sources.py).

This is the WHOLE middle of the pipeline in one command (embed -> pairing_funnel -> score):
  A. RETYPE SWEEP  — retype interpretive conclusions data->argument so the bottom-up
                     channel can attach raw counts beneath them (idempotent).
  B. CANDIDATES    — the four channels above, deduped.
  C. CAP + RANK    — sort by cosine/NLI confidence, take top MAX_LLM_PAIRS.
  D. LABEL         — send to gpt-oss (label_pairs.label_pairs), cached.
  E. MERGE         — append the new cards to cards.jsonl, renumbered + deduped.

Output: candidate_pairs.json {meta, pairs:[...]} and cards.jsonl. --no-label stops after C.

    python scripts/pairing_funnel.py --max-llm-pairs 3000      # full build (retype+pair+label+merge)
    python scripts/pairing_funnel.py --no-label                # candidates only, no LLM
    python scripts/pairing_funnel.py --new-nodes ids.txt       # incremental: pair only new nodes, append
    python scripts/pairing_funnel.py --nli --device cuda       # + contradiction channel (GPU)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import (  # noqa: E402
    read_nodes, layer_of, write_jsonl, read_cards, write_cards, validate_store, card_id,
)
from label_pairs import label_pairs  # noqa: E402

# budget fractions of MAX_LLM_PAIRS
BUDGET = {"top_down": 0.50, "data->argument": 0.30, "argument->question": 0.15,
          "contradiction": 0.05}
TOPK_PER_HYP = 50           # nearest data nodes retrieved per question node (top-down)
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# --- FIX 1 retype sweep -------------------------------------------------------------
# Data-layer nodes whose text carries inference markers are interpretive conclusions,
# not raw data; they belong in the argument layer so the bottom-up data->argument
# channel can attach the raw counts beneath them (see design.md "THE TWO DIRECTIONS").
_JUDGMENT = [re.compile(p, re.I) for p in [
    r"\bpoints?\s+to\b", r"\bsuggests?\b", r"\bfavou?rs?\b",
    r"\bleans?\s+(towards?|toward)\b", r"\bindicat(e|es|ed|ing)\b",
    r"supports?\s+(the\s+)?(hypothesis|zoonotic|lab.?leak|claim|conclusion|idea|view|notion)",
    r"\bconsistent\s+with\b",
    r"the\s+author\s+(assess|assesses|believes|concludes|estimates|judges|argues|thinks|considers)",
    r"\bimplicat(e|es|ed|ing)\b", r"\bimplies\b", r"\b(more|less)\s+likely\b",
    r"\bis\s+evidence\s+(for|that|against|of)\b",
    r"bayes\s+factor\s+(of\s+|in\s+favou?r|is\s+|for\s+the|favou?rs)",
    r"\bweighs?\s+(in|against|towards?|for)\b",
    r"\b(strengthens?|weakens?|undermines?|bolsters?)\b",
    r"\bprobability\s+of\b.*\bbeing\b",
]]
_EXCLUDE = re.compile(
    r"is\s+called|is\s+defined|refers?\s+to|we\s+define|let\s+\w+\s+be|"
    r"shading\s+(on\s+the\s+map\s+)?indicates|a\s+bayes\s+factor\s+is\s+how|"
    r"is\s+the\s+(ratio|event|probability|virus|disease|number|name)|"
    r"in\s+probability\s+theory|\bcoin\b|\bheads\b|can\s+help\s+us|"
    r"help\s+us\s+(gain|understand)|we\s+(could|can)\s+say|"
    r"using\s+hypothesis\s+testing|amenable\s+to",
    re.I,
)


def retype_bridges(nodes) -> int:
    """Retype interpretive-conclusion nodes from data -> argument (in place). Idempotent:
    already-argument nodes are skipped. Returns the count moved. Original type kept in
    meta.retyped_from."""
    moved = 0
    for n in nodes:
        if layer_of(n.type) != "data":
            continue
        t = n.canonical_text
        if _EXCLUDE.search(t) or not any(p.search(t) for p in _JUDGMENT):
            continue
        n.meta["retyped_from"] = n.type
        n.meta["retype_reason"] = "interpretive-conclusion (pairing_funnel retype sweep)"
        n.type = "claim"
        n.meta["layer"] = "argument"
        moved += 1
    return moved


def write_nodes(path, nodes) -> int:
    return write_jsonl(path, nodes)


def _load_new_ids(spec: str) -> set[str]:
    p = Path(spec)
    if p.exists():
        txt = p.read_text()
        return {x.strip() for x in re.split(r"[\s,]+", txt) if x.strip().startswith("n-")}
    return {x.strip() for x in spec.split(",") if x.strip()}


def make_nli(model_name, device=None):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, device=device)
    id2label = {int(k): v.lower() for k, v in ce.model.config.id2label.items()}
    return ce, {v: k for k, v in id2label.items()}


def nli_scores(ce, label2idx, pairs_text, batch_size=64, single_ordering=False):
    ab = np.asarray(ce.predict([[a, b] for a, b in pairs_text], batch_size=batch_size,
                               apply_softmax=True, show_progress_bar=False))
    if single_ordering:
        return (ab[:, label2idx["entailment"]], ab[:, label2idx["contradiction"]],
                ab[:, label2idx["neutral"]])
    ba = np.asarray(ce.predict([[b, a] for a, b in pairs_text], batch_size=batch_size,
                               apply_softmax=True, show_progress_bar=False))
    ent = np.maximum(ab[:, label2idx["entailment"]], ba[:, label2idx["entailment"]])
    con = np.maximum(ab[:, label2idx["contradiction"]], ba[:, label2idx["contradiction"]])
    neu = np.minimum(ab[:, label2idx["neutral"]], ba[:, label2idx["neutral"]])
    return ent, con, neu


def _pair(src, dst, channel, cos, ent=None, con=None):
    rel = float(cos) if ent is None else float(max(ent, con))
    return dict(src=src, dst=dst, channel=channel, cosine=float(cos),
                nli_entail=None if ent is None else float(ent),
                nli_contra=None if con is None else float(con),
                relation_signal=rel, confidence=rel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default=str(ROOT / "artifacts" / "epistemic" / "nodes.jsonl"))
    ap.add_argument("--emb", default=str(ROOT / "artifacts" / "epistemic" / "embeddings.npy"))
    ap.add_argument("--emb-index", default=str(ROOT / "artifacts" / "epistemic" / "embeddings_index.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "epistemic" / "candidate_pairs.json"))
    ap.add_argument("--cards", default=str(ROOT / "artifacts" / "epistemic" / "cards.jsonl"),
                    help="cards output (labeling + merge run inline unless --no-label)")
    ap.add_argument("--max-llm-pairs", type=int, default=1000)
    ap.add_argument("--topk-per-hyp", type=int, default=TOPK_PER_HYP)
    ap.add_argument("--new-nodes", default=None,
                    help="incremental ingest: comma-list or a file of node_ids; pair only "
                         "these against the existing graph and APPEND (dedup) to --out/--cards")
    ap.add_argument("--no-retype", action="store_true", help="skip the FIX-1 retype sweep")
    ap.add_argument("--no-label", action="store_true", help="stop after writing candidate_pairs (no LLM)")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--nli", action="store_true", help="enable NLI rerank + contradiction channel (needs GPU to be fast)")
    ap.add_argument("--nli-model", default=NLI_MODEL)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default auto)")
    ap.add_argument("--nli-cap", type=int, default=4000, help="max pairs sent to NLI (tractability)")
    args = ap.parse_args()

    nodes = read_nodes(args.nodes)

    # --- STEP A: retype sweep (FIX 1) ----------------------------------------------
    # Interpretive conclusions mis-typed as data can't receive support (data->data is
    # blocked), so the support chain stays one hop deep. Retype them to the argument
    # layer before pairing. Idempotent: retyped nodes leave the data layer.
    if not args.no_retype:
        n_ret = retype_bridges(nodes)
        if n_ret:
            write_nodes(args.nodes, nodes)
            print(f"retype sweep: moved {n_ret} interpretive-conclusion nodes data->argument")
        else:
            print("retype sweep: nothing to retype (already applied)")

    text = {n.node_id: n.canonical_text for n in nodes}
    mat = np.load(args.emb)
    idx = json.loads(Path(args.emb_index).read_text())
    row = {nid: i for i, nid in enumerate(idx["node_ids"])}
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    # pool = ALL nodes (leakage guard retired). group by 3-layer.
    by_layer = {"data": [], "argument": [], "question": []}
    for n in nodes:
        if n.node_id in row:
            by_layer[layer_of(n.type)].append(n.node_id)
    data_ids, arg_ids, q_ids = by_layer["data"], by_layer["argument"], by_layer["question"]
    print(f"Pool: {len(nodes)} nodes (no holdout)  data={len(data_ids)} "
          f"arg={len(arg_ids)} q={len(q_ids)}")

    # incremental ingest: restrict every channel so one side is a new node
    new_ids = _load_new_ids(args.new_nodes) if args.new_nodes else None
    if new_ids:
        new_ids &= set(data_ids) | set(arg_ids) | set(q_ids)
        print(f"incremental: {len(new_ids)} new nodes -> pairing them against the existing graph")

    def emb(ids):
        return norm[[row[i] for i in ids]]

    def rank_pairs(src_ids, dst_ids, channel, budget, directed=True):
        """All src×dst (or unordered) pairs, ranked by cosine, top `budget`."""
        if not src_ids or not dst_ids or budget <= 0:
            return []
        if directed:
            sim = emb(src_ids) @ emb(dst_ids).T                 # [S x D]
            flat = [(src_ids[i], dst_ids[j], sim[i, j])
                    for i in range(len(src_ids)) for j in range(len(dst_ids))]
        else:  # unordered within one set
            m = emb(src_ids); sim = m @ m.T
            flat = [(src_ids[i], src_ids[j], sim[i, j])
                    for i, j in combinations(range(len(src_ids)), 2)]
        flat.sort(key=lambda t: -t[2])
        return [_pair(s, d, channel, c) for s, d, c in flat[:budget]]

    def top_down(q_pool, d_pool, budget):
        """For each question node in q_pool, its k nearest data nodes → (data → question)."""
        if not q_pool or not d_pool or budget <= 0:
            return []
        sim = emb(q_pool) @ emb(d_pool).T                        # [Q x D]
        k = min(args.topk_per_hyp, len(d_pool))
        cand = []
        for qi in range(len(q_pool)):
            top = np.argpartition(-sim[qi], k - 1)[:k]
            for j in top:
                cand.append((d_pool[j], q_pool[qi], sim[qi, j]))  # evidence → hypothesis
        cand.sort(key=lambda t: -t[2])
        return [_pair(s, d, "top_down", c) for s, d, c in cand[:budget]]

    M = args.max_llm_pairs
    b_td = int(round(BUDGET["top_down"] * M))
    b_da = int(round(BUDGET["data->argument"] * M))
    b_aq = int(round(BUDGET["argument->question"] * M))
    b_ct = int(round(BUDGET["contradiction"] * M))
    if not args.nli:
        b_td += b_ct                # no NLI → no contradiction channel; give it to top-down
        b_ct = 0

    pairs = []
    if new_ids:
        # same channels, but every pair must touch a new node (new×existing ∪ new×new)
        nd = [d for d in data_ids if d in new_ids]
        na = [a for a in arg_ids if a in new_ids]
        nq = [q for q in q_ids if q in new_ids]
        pairs += top_down(nq, data_ids, b_td)        # new hypotheses ← all data
        pairs += top_down(q_ids, nd, b_td)           # all hypotheses ← new data
        pairs += rank_pairs(nd, arg_ids, "data->argument", b_da)
        pairs += rank_pairs(data_ids, na, "data->argument", b_da)
        pairs += rank_pairs(na, q_ids, "argument->question", b_aq)
        pairs += rank_pairs(arg_ids, nq, "argument->question", b_aq)
    else:
        pairs += top_down(q_ids, data_ids, b_td)
        pairs += rank_pairs(data_ids, arg_ids, "data->argument", b_da)
        pairs += rank_pairs(arg_ids, q_ids, "argument->question", b_aq)

    # contradiction channel: cosine-nearest data↔data / arg↔arg, kept only if NLI says
    # contradiction. Needs NLI; skipped otherwise.
    ce = None
    if args.nli and b_ct > 0:
        print(f"Loading NLI {args.nli_model} on {args.device or 'auto'}…")
        ce, l2i = make_nli(args.nli_model, device=args.device)
        contra_cand = (rank_pairs(data_ids, data_ids, "contradiction", args.nli_cap // 2, directed=False)
                       + rank_pairs(arg_ids, arg_ids, "contradiction", args.nli_cap // 2, directed=False))
        ent, con, neu = nli_scores(ce, l2i, [(text[p["src"]], text[p["dst"]]) for p in contra_cand])
        for p, e, c in zip(contra_cand, ent, con):
            p.update(nli_entail=float(e), nli_contra=float(c),
                     relation_signal=float(c), confidence=float(c))
        contra_cand.sort(key=lambda p: -p["nli_contra"])
        pairs += contra_cand[:b_ct]

    # optional: NLI-rerank the non-contradiction channels too (better precision on GPU)
    if args.nli and ce is not None:
        main_pairs = [p for p in pairs if p["channel"] != "contradiction"]
        cap = min(args.nli_cap, len(main_pairs))
        sub = sorted(main_pairs, key=lambda p: -p["cosine"])[:cap]
        ent, con, neu = nli_scores(ce, l2i, [(text[p["src"]], text[p["dst"]]) for p in sub])
        for p, e, c in zip(sub, ent, con):
            p.update(nli_entail=float(e), nli_contra=float(c),
                     relation_signal=float(max(e, c)),
                     confidence=0.7 * float(max(e, c)) + 0.3 * np.clip((p["cosine"] - .70) / .30, 0, 1))

    # incremental: every pair must touch a new node (safety net for the contradiction channel)
    if new_ids:
        pairs = [p for p in pairs if p["src"] in new_ids or p["dst"] in new_ids]

    # dedup against pairs that already exist (incremental appends; full build starts clean)
    prior_pairs = []
    prior_keys = set()
    if new_ids and Path(args.out).exists():
        prior_pairs = json.loads(Path(args.out).read_text()).get("pairs", [])
        prior_keys = {(p["src"], p["dst"]) for p in prior_pairs}

    # dedup (safety) + global rank + cap
    seen, uniq = set(), []
    for p in sorted(pairs, key=lambda p: -p["confidence"]):
        key = (p["src"], p["dst"])
        if key in seen or key in prior_keys:
            continue
        seen.add(key)
        uniq.append(p)
    uniq = uniq[:M]

    all_pairs = prior_pairs + uniq
    for r, p in enumerate(all_pairs):
        p["rank"] = r

    meta = {"max_llm_pairs": M, "n_pairs": len(all_pairs), "new_this_run": len(uniq),
            "nli_model": args.nli_model if args.nli else None,
            "topk_per_hyp": args.topk_per_hyp, "budget": BUDGET,
            "pool_nodes": len(nodes), "leakage_guard": "retired",
            "by_channel": dict(Counter(p["channel"] for p in all_pairs))}
    Path(args.out).write_text(json.dumps({"meta": meta, "pairs": all_pairs}, indent=2))
    print(f"\nwrote {len(all_pairs)} candidate pairs ({len(uniq)} new) -> {args.out}")
    print("  by channel:", meta["by_channel"])

    if args.no_label:
        print("--no-label: stopping before the LLM labeling step")
        return

    # --- STEP D + E: label the new pairs and merge into cards.jsonl -----------------
    base_cards = read_cards(args.cards) if (new_ids and Path(args.cards).exists()) else []
    new_cards, _events, stats = label_pairs(uniq, text, model=args.model,
                                            start_index=len(base_cards))
    have = {(frozenset(c.premises), c.target, c.kind) for c in base_cards}
    merged = list(base_cards)
    for c in new_cards:
        k = (frozenset(c.premises), c.target, c.kind)
        if k in have:
            continue
        have.add(k)
        c.card_id = card_id(len(merged))
        merged.append(c)
    errs = validate_store(nodes, [], merged)
    if errs:
        print("  VALIDATION ERRORS:", errs[:10])
    n_written = write_cards(args.cards, merged)
    print(f"\nlabeled {stats['pairs_in']} pairs -> {stats['cards_out']} cards; "
          f"merged store: {n_written} cards -> {args.cards}"
          f"  (${stats['tokens']['total'] / 1e6 * 0.30:.3f})")


if __name__ == "__main__":
    main()
