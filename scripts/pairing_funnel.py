"""Step C — the pairing funnel (design.md steps 6, lines 93-124).

Picks, from the ~160k type-legal node pairs, the best <=MAX_LLM_PAIRS to send to the
labeler. Three stages:

  Stage 1 — type-layer blocking. Data mostly targets arguments; arguments mostly
    target questions. Only these channels survive (budget = share of MAX_LLM_PAIRS):
        data -> argument      directed   60%   (evidence bearing on a claim)
        argument -> question  directed   25%   (a claim bearing on a hypothesis)
        data -> data          symmetric  10%   (contradictions between findings)
        argument -> argument  symmetric   5%   (rebuttals between claims)
    Skipped: data->question, question->question (structurally unlikely relations).

  Stage 2 — cheap local filter. A DeBERTa NLI cross-encoder scores whether two nodes
    are even talking to each other (entailment or contradiction, vs neutral). Local,
    free, CPU. NLI is directional, so we score BOTH orderings and keep the max.
    To keep the CPU pass tractable we first pre-rank each channel by embedding cosine
    and only NLI the top `--nli-cap-per-channel` (the most-related pairs; recall-safe
    because a real relation implies topical proximity).

  Stage 3 — rank & cap. Within each channel, rank by a confidence blend of the NLI
    relation signal and cosine, take the channel's budget share. Union <= MAX_LLM_PAIRS.

Leakage guard: `conclusory` nodes (the judge's answer key) are dropped from the pool
up front, so no card can ever get a conclusory premise/target.

Output: `artifacts/epistemic/candidate_pairs.json` = {meta, pairs:[{src,dst,channel,
cosine,nli_entail,nli_contra,nli_neutral,relation_signal,confidence,rank}]}.

Usage:
    python scripts/pairing_funnel.py                    # full funnel
    python scripts/pairing_funnel.py --no-nli           # cosine-only (fast smoke)
    python scripts/pairing_funnel.py --max-llm-pairs 200
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

# (name, src_layer, dst_layer, directed, budget_fraction)
CHANNELS = [
    ("data->argument",     "data",     "argument", True,  0.60),
    ("argument->question", "argument", "question", True,  0.25),
    ("data->data",         "data",     "data",     False, 0.10),
    ("argument->argument", "argument", "argument", False, 0.05),
]
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def load_roles(nodes, tagged_path):
    role_by_claim = {}
    with open(tagged_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                role_by_claim[r["claim_id"]] = r.get("role")
    out = {}
    for n in nodes:
        cid = n.provenance[0].claim_id if n.provenance else None
        out[n.node_id] = role_by_claim.get(cid)
    return out


def channel_pairs(ids_by_layer, src_layer, dst_layer, directed):
    if directed:
        return list(product(ids_by_layer[src_layer], ids_by_layer[dst_layer]))
    return list(combinations(ids_by_layer[src_layer], 2))


def make_nli(model_name, device=None):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, device=device)      # device=None → auto (cuda if present)
    id2label = {int(k): v.lower() for k, v in ce.model.config.id2label.items()}
    label2idx = {v: k for k, v in id2label.items()}
    return ce, label2idx


def nli_scores(ce, label2idx, pairs_text, batch_size=64, single_ordering=False):
    """Return (entail, contra, neutral) probs per pair. NLI is directional, so by
    default we score BOTH orderings and take the max; `single_ordering` halves the
    compute (A→B only) when speed matters more than catching one-way entailment."""
    ab = np.asarray(ce.predict([[a, b] for a, b in pairs_text], batch_size=batch_size,
                               apply_softmax=True, show_progress_bar=False))
    if single_ordering:
        ent = ab[:, label2idx["entailment"]]
        con = ab[:, label2idx["contradiction"]]
        neu = ab[:, label2idx["neutral"]]
        return ent, con, neu
    ba = np.asarray(ce.predict([[b, a] for a, b in pairs_text], batch_size=batch_size,
                               apply_softmax=True, show_progress_bar=False))
    ent = np.maximum(ab[:, label2idx["entailment"]], ba[:, label2idx["entailment"]])
    con = np.maximum(ab[:, label2idx["contradiction"]], ba[:, label2idx["contradiction"]])
    neu = np.minimum(ab[:, label2idx["neutral"]], ba[:, label2idx["neutral"]])
    return ent, con, neu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default=str(ROOT / "artifacts" / "epistemic" / "nodes.jsonl"))
    ap.add_argument("--tagged", default=str(ROOT / "artifacts" / "epistemic" / "claims_tagged.jsonl"))
    ap.add_argument("--emb", default=str(ROOT / "artifacts" / "epistemic" / "embeddings.npy"))
    ap.add_argument("--emb-index", default=str(ROOT / "artifacts" / "epistemic" / "embeddings_index.json"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "epistemic" / "candidate_pairs.json"))
    ap.add_argument("--max-llm-pairs", type=int, default=1000)
    ap.add_argument("--nli-cap-per-channel", type=int, default=3000,
                    help="only NLI the top-N-by-cosine pairs per channel (tractability)")
    ap.add_argument("--no-nli", action="store_true", help="skip DeBERTa; rank by cosine only")
    ap.add_argument("--nli-model", default=NLI_MODEL,
                    help="e.g. cross-encoder/nli-deberta-v3-xsmall for a faster CPU run")
    ap.add_argument("--device", default=None,
                    help="torch device for NLI: cuda | mps | cpu (default auto). "
                         "On a GPU box the default already picks cuda.")
    ap.add_argument("--single-ordering", action="store_true",
                    help="score NLI A->B only (halves compute; skips one-way entailment)")
    ap.add_argument("--w-nli", type=float, default=0.7, help="confidence weight on NLI vs cosine")
    args = ap.parse_args()

    nodes = read_nodes(args.nodes)
    roles = load_roles(nodes, args.tagged)

    # --- leakage guard: drop conclusory nodes from the pool -----------------------
    kept = [n for n in nodes if roles.get(n.node_id) != "conclusory"]
    dropped = len(nodes) - len(kept)
    text = {n.node_id: n.canonical_text for n in kept}

    # embeddings, normalized, indexed by node_id
    mat = np.load(args.emb)
    idx = json.loads(Path(args.emb_index).read_text())
    row = {nid: i for i, nid in enumerate(idx["node_ids"])}
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    ids_by_layer = {"data": [], "argument": [], "question": []}
    for n in kept:
        ids_by_layer[layer_of(n.type)].append(n.node_id)

    print(f"Pool: {len(kept)} nodes ({dropped} conclusory dropped)  "
          f"data={len(ids_by_layer['data'])} arg={len(ids_by_layer['argument'])} "
          f"q={len(ids_by_layer['question'])}")

    ce = label2idx = None
    if not args.no_nli:
        print(f"Loading NLI model {args.nli_model} on {args.device or 'auto'} "
              f"(first run downloads weights)…")
        ce, label2idx = make_nli(args.nli_model, device=args.device)
        print("  labels:", label2idx)

    all_pairs = []
    for name, sl, dl, directed, frac in CHANNELS:
        raw = channel_pairs(ids_by_layer, sl, dl, directed)
        if not raw:
            continue
        # cosine for every pair in the channel (cheap)
        a = np.array([row[s] for s, _ in raw])
        b = np.array([row[d] for _, d in raw])
        cos = np.sum(norm[a] * norm[b], axis=1)
        order = np.argsort(-cos)                      # best cosine first

        budget = int(round(frac * args.max_llm_pairs))
        # NLI only the top-N by cosine (recall-safe pre-trim), else cosine-only
        if args.no_nli:
            picks = order[:budget]
            for j in picks:
                s, d = raw[j]
                all_pairs.append(dict(src=s, dst=d, channel=name, cosine=float(cos[j]),
                                      nli_entail=None, nli_contra=None, nli_neutral=None,
                                      relation_signal=float(cos[j]),
                                      confidence=float(cos[j])))
            print(f"  [{name:20}] {len(raw):>7} pairs -> top {len(picks)} by cosine")
            continue

        cap = min(args.nli_cap_per_channel, len(order))
        cand = order[:cap]
        texts = [(text[raw[j][0]], text[raw[j][1]]) for j in cand]
        ent, con, neu = nli_scores(ce, label2idx, texts, single_ordering=args.single_ordering)
        rel = np.maximum(ent, con)                    # "there is a relation" signal
        cos_c = cos[cand]
        cos_norm = np.clip((cos_c - 0.70) / 0.30, 0, 1)   # stretch compressed range
        conf = args.w_nli * rel + (1 - args.w_nli) * cos_norm
        take = np.argsort(-conf)[:budget]
        for k in take:
            j = cand[k]
            s, d = raw[j]
            all_pairs.append(dict(src=s, dst=d, channel=name, cosine=float(cos_c[k]),
                                  nli_entail=float(ent[k]), nli_contra=float(con[k]),
                                  nli_neutral=float(neu[k]),
                                  relation_signal=float(rel[k]), confidence=float(conf[k])))
        print(f"  [{name:20}] {len(raw):>7} pairs -> NLI top {cap} -> keep {len(take)} "
              f"(conf {conf[take].min():.2f}–{conf[take].max():.2f})")

    all_pairs.sort(key=lambda p: -p["confidence"])
    all_pairs = all_pairs[:args.max_llm_pairs]
    for rank, p in enumerate(all_pairs):
        p["rank"] = rank

    meta = {
        "max_llm_pairs": args.max_llm_pairs, "n_pairs": len(all_pairs),
        "nli_model": None if args.no_nli else args.nli_model,
        "nli_cap_per_channel": args.nli_cap_per_channel, "w_nli": args.w_nli,
        "pool_nodes": len(kept), "conclusory_dropped": dropped,
        "by_channel": dict(Counter(p["channel"] for p in all_pairs)),
    }
    Path(args.out).write_text(json.dumps({"meta": meta, "pairs": all_pairs}, indent=2))
    print(f"\nwrote {len(all_pairs)} candidate pairs -> {args.out}")
    print("  by channel:", meta["by_channel"])


if __name__ == "__main__":
    main()
