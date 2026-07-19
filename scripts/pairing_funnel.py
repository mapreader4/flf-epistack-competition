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

Output: artifacts/epistemic/candidate_pairs.json = {meta, pairs:[{src,dst,channel,
cosine,nli_entail,nli_contra,relation_signal,confidence,rank}]}.

    python scripts/pairing_funnel.py                       # cosine-only (local)
    python scripts/pairing_funnel.py --nli --device cuda   # + contradiction channel (GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epistemic_store import read_nodes, layer_of  # noqa: E402

# budget fractions of MAX_LLM_PAIRS
BUDGET = {"top_down": 0.50, "data->argument": 0.30, "argument->question": 0.15,
          "contradiction": 0.05}
TOPK_PER_HYP = 50           # nearest data nodes retrieved per question node (top-down)
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


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
    ap.add_argument("--max-llm-pairs", type=int, default=1000)
    ap.add_argument("--topk-per-hyp", type=int, default=TOPK_PER_HYP)
    ap.add_argument("--nli", action="store_true", help="enable NLI rerank + contradiction channel (needs GPU to be fast)")
    ap.add_argument("--nli-model", default=NLI_MODEL)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default auto)")
    ap.add_argument("--nli-cap", type=int, default=4000, help="max pairs sent to NLI (tractability)")
    args = ap.parse_args()

    nodes = read_nodes(args.nodes)
    text = {n.node_id: n.canonical_text for n in nodes}
    mat = np.load(args.emb)
    idx = json.loads(Path(args.emb_index).read_text())
    row = {nid: i for i, nid in enumerate(idx["node_ids"])}
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    # pool = ALL nodes (leakage guard retired). group by 3-layer.
    by_layer = {"data": [], "argument": [], "question": []}
    for n in nodes:
        by_layer[layer_of(n.type)].append(n.node_id)
    data_ids, arg_ids, q_ids = by_layer["data"], by_layer["argument"], by_layer["question"]
    print(f"Pool: {len(nodes)} nodes (no holdout)  data={len(data_ids)} "
          f"arg={len(arg_ids)} q={len(q_ids)}")

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

    def top_down(budget):
        """For each question node, its k nearest data nodes → (data → question)."""
        if not q_ids or not data_ids or budget <= 0:
            return []
        sim = emb(q_ids) @ emb(data_ids).T                       # [Q x D]
        k = min(args.topk_per_hyp, len(data_ids))
        cand = []
        for qi in range(len(q_ids)):
            top = np.argpartition(-sim[qi], k - 1)[:k]
            for j in top:
                cand.append((data_ids[j], q_ids[qi], sim[qi, j]))  # evidence → hypothesis
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
    pairs += top_down(b_td)
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

    # dedup (safety) + global rank + cap
    seen, uniq = set(), []
    for p in sorted(pairs, key=lambda p: -p["confidence"]):
        key = (p["src"], p["dst"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    uniq = uniq[:M]
    for r, p in enumerate(uniq):
        p["rank"] = r

    meta = {"max_llm_pairs": M, "n_pairs": len(uniq),
            "nli_model": args.nli_model if args.nli else None,
            "topk_per_hyp": args.topk_per_hyp, "budget": BUDGET,
            "pool_nodes": len(nodes), "leakage_guard": "retired",
            "by_channel": dict(Counter(p["channel"] for p in uniq))}
    Path(args.out).write_text(json.dumps({"meta": meta, "pairs": uniq}, indent=2))
    print(f"\nwrote {len(uniq)} candidate pairs -> {args.out}")
    print("  by channel:", meta["by_channel"])


if __name__ == "__main__":
    main()
