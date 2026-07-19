# Epistemic argument-graph pipeline — what we built & how it works

*As of 2026-07-18. Covers design.md steps 5-8 (built) and how step 9 (query) plugs
in. For the vision read `design.md`; for the frozen data shapes read
`docs/contracts-steps-5-9.md`; for why-decisions read `CONTEXT.md`.*

## What this is

We turn the judge's COVID-origins ruling into a **typed argument graph**: atomic
claims become **nodes**, support/attack relations become **cards** (reified, so joint
premises are first-class), and every node gets a **DF-QuAD strength** in [0,1]. Unlike
HippoRAG (which extracts topic triples and ranks chunks), this graph answers *what
supports what, where sources disagree, and which link is load-bearing.*

The graph is built and running on the real document: **590 nodes → 338 cards → scored,
36 contested nodes.** Cost of the one LLM step (labeling): **$0.07**.

## The workflow

```
claims.json ──type_claims.py──▶ nodes.jsonl (590 typed nodes)         [pre-existing]
                                     │
                    map_layers.py    │  stamp meta.layer ∈ {data,argument,question}
                                     ▼
                                nodes.jsonl (+layer)
                                     │
                   embed_nodes.py    │  e5-large-instruct, 1024-dim
                                     ▼
                        embeddings.npy [590×1024] + embeddings_index.json
                                     │
                 pairing_funnel.py   │  Stage1 type-block → Stage2 NLI(optional) → Stage3 rank/cap
                                     ▼
                        candidate_pairs.json  (≤1000 best pairs)
                                     │
                   label_pairs.py    │  gpt-oss: support / attack / none
                                     ▼
                        cards.jsonl (338 cards)  + card_added/llm_call events
                                     │
                  score_dfquad.py    │  DF-QuAD arithmetic, no LLM
                                     ▼
                        scores.jsonl (strength per node)  ──▶  QUERY LAYER (step 9)
```

Two label systems ride on the same claims and are joined by
`node.provenance[0].claim_id`:
- **type** (in `nodes.jsonl`): the epistemic role — see "How nodes are categorized".
- **role** (in `claims_tagged.jsonl`): the *leakage guard* — evidentiary /
  **conclusory** / procedural. Conclusory = the judge's own verdict/priors = the
  answer key; it is dropped from the graph and reserved for eval.

## The scripts (each mirrors `type_claims.py`: Together client, temp 0, cached, budgeted)

| Script | Does | LLM? | Output |
|---|---|---|---|
| `map_layers.py` | stamp the 3-layer group onto each node | no | `nodes.jsonl` (+`meta.layer`) |
| `embed_nodes.py` | e5-large-instruct over `canonical_text` | embed API | `embeddings.npy` + index |
| `pairing_funnel.py` | pick ≤1000 pairs worth labeling | no (local NLI opt.) | `candidate_pairs.json` |
| `label_pairs.py` | gpt-oss labels each pair support/attack/none | yes (gpt-oss) | `cards.jsonl` + events |
| `score_dfquad.py` | DF-QuAD strengths + `ablate()` | no | `scores.jsonl` |

Plus the substrate: `epistemic_store.py` (Node/Edge/**Card** schema, `read/write_cards`,
`validate_store`, `layer_of`) and `event_store.py` (append-only log; `card_added` /
`score_computed` fold into the snapshot).

### The pairing funnel in detail
Stage 1 **type-layer blocking** keeps only structurally-sensible channels, each given a
share of the 1000-pair budget:

| channel | direction | budget |
|---|---|---|
| data → argument | evidence bearing on a claim | 60% |
| argument → question | claim bearing on a hypothesis | 25% |
| data → data | contradictions between findings | 10% |
| argument → argument | rebuttals between claims | 5% |

(`data→question` and `question→question` are skipped.) Stage 2 is a DeBERTa-NLI cheap
filter; Stage 3 ranks by a cosine+NLI confidence blend and caps per channel.

## How to run it (end to end)

```bash
# from repo root; python is NOT on PATH — use the venv
.venv/bin/python scripts/map_layers.py
.venv/bin/python scripts/embed_nodes.py
.venv/bin/python scripts/pairing_funnel.py --no-nli          # see "speed" below
.venv/bin/python scripts/label_pairs.py --emit-events
.venv/bin/python scripts/score_dfquad.py
```

### Speed note (the DeBERTa NLI stage)
NLI (Stage 2) is a *precision* optimization: it filters topically-close-but-unrelated
pairs before the LLM. At N=590 you don't need it — cosine ranking + the cheap gpt-oss
labeler ($0.07) already works, so we run `--no-nli`. When you *do* want NLI (larger
corpora, tighter cards), it's slow on **this** machine because it's an Intel Mac with
no usable GPU (~5 pairs/sec). Faster options, in order:
- **Run the funnel on a collaborator's CUDA machine** — sentence-transformers
  auto-detects the GPU; the whole NLI pass is seconds. Nothing to change, or force it:
  `--device cuda`.
- **Use the smaller model locally:** `--nli-model cross-encoder/nli-deberta-v3-xsmall`
  (much faster, slightly less accurate).
- **Halve the compute:** `--single-ordering` (scores A→B only).
- **Trim the workload:** `--nli-cap-per-channel 800`.
- There is **no serverless NLI API on Together**; the "API" path is simply that the
  gpt-oss labeler is cheap enough to be the filter, which is what `--no-nli` relies on.

## Worked examples (real output on the current graph)

**Query 1 — "What's the evidence for the lab-leak hypothesis?"** Walk inward over
support (RA) cards, rank premises by strength:
```
n-00323 [hypothesis] "The lab leak theory suggests lineage A is ancestral…"  str 0.650
  ← n-00362 [claim] "The Wuhan Institute of Virology was to be involved with…"  str 0.973
  ← n-00575 [claim] "The DEFUSE research proposal involves modifying viruses…"  str 0.957
  ← n-00576 [claim] "The WIV is involved in the DEFUSE research…"               str 0.799
```
The graph surfaces exactly the right evidence (WIV + DEFUSE), ranked — not a bag of
keyword-matched chunks.

**Query 2 — "Where do sources disagree?"** Nodes with both support and attack cards
(36 of them). Most contested:
```
n-00405 [claim] "Rootclaim argues the observational evidence is equally consistent…"
        strength 0.430   support vs=0.84   attack va=0.98   → attack currently winning
```

**Query 3 — "What's the weakest link?"** `ablate(target)` removes each card, re-scores,
ranks by |Δ|. The largest-Δ card is load-bearing. (See fixture regression:
`score_dfquad.py --selftest`.)

## Using this from the query layer (replacing the HippoRAG path)

`scripts/query_graph.py` today is the **baseline**: HippoRAG `rag_qa` — it retrieves
chunks by topic similarity and hands chunk *text* to one LLM call. The graph never
reaches the answering model. Our query layer inverts that: **the graph does the
structural reasoning, then the top nodes' source quotes go to the LLM guided by the
structure.**

A parallel `scripts/query_epistemic.py` (colleague's lane — see
`docs/handoff-query-eval-lane.md`) reads our artifacts instead of a HippoRAG save-dir.
The swap, concretely:

```python
# --- HippoRAG baseline (query_graph.py) ---------------------------------------
from run_hipporag_index import build_hipporag
hipporag, _ = build_hipporag(save_dir, llm, embed, syn_threshold)
solutions, *_ = hipporag.rag_qa(queries=queries)          # graph only ranks chunks

# --- our graph-guided path (query_epistemic.py) -------------------------------
import sys; sys.path.insert(0, "scripts")
from epistemic_store import read_nodes, read_cards
from score_dfquad import score_graph, ablate

nodes  = read_nodes("artifacts/epistemic/nodes.jsonl")
cards  = read_cards("artifacts/epistemic/cards.jsonl")
scores = score_graph(nodes, cards)                        # {node_id: ScoreRecord}

def evidence_for(target):                                 # Query 1
    prem = [(p, scores[p].strength)
            for c in cards if c.kind == "RA" and c.target == target
            for p in c.premises]
    return sorted(prem, key=lambda x: -x[1])

def disagreements():                                      # Query 2
    return [nid for nid, s in scores.items() if s.in_supporters and s.in_attackers]

def weakest_link(target):                                 # Query 3
    return ablate(nodes, cards, target)                   # {card_id: |Δ|}, sorted

# then: build the LLM prompt from the structural result + the T1 quotes of the top
# nodes (node.provenance[0].quote) and call gpt-oss to write the prose answer —
# reuse type_claims.build_client(), temperature 0, max_tokens=4000.
```

The entry-point problem "map a natural-language question → a target node" is the one
genuinely new piece: embed the question with `embed_nodes`'s e5 template and pick the
nearest hypothesis/claim node, then run the traversals above. Keep every cited node
traceable to its T1 span so answers are auditable — the whole point vs HippoRAG.

## TODOs (and where they go)

- **Enable NLI Stage 2 for real** — run `pairing_funnel.py` with NLI on a GPU box (or
  `--nli-model …xsmall` locally), regenerate `candidate_pairs.json`, re-label.
  *Where:* `pairing_funnel.py` (already flagged), then rerun `label_pairs.py`.
- **Card-quality pass** — the `data→data` "support" cards (similar findings
  reinforcing each other) are the loosest; tighten the labeler prompt or down-weight
  that channel. *Where:* `label_pairs.py` `USER_TEMPLATE` / channel weights in
  `pairing_funnel.py` `CHANNELS`.
- **Joint cards & PA** — MVP builds single-premise cards. Add a pass that clusters
  co-targeting CA cards into one joint attack (design's C2+C3) and adds PA/outweighs
  between competing cards. *Where:* new `consolidate_cards.py` after `label_pairs.py`;
  `score_dfquad.py` already handles multi-premise `min()` and treats PA as attack.
- **Per-node base scores** — calibrate `v0` instead of uniform 0.5 (design.md's
  narrative digits need this). *Where:* `score_dfquad.py` already reads
  `payload["base"]`; feed it from tier/`type_confidence` or a calibration class.
- **Query layer + eval** — the 3 queries + `answer_key.json` scoring. *Where:* new
  `scripts/query_epistemic.py` + eval script (colleague's lane;
  `docs/handoff-query-eval-lane.md`).
- **HippoRAG comparison column** — run the same golden queries through
  `query_graph.py` (baseline) and `query_epistemic.py` (ours), compare. *Where:* an
  eval harness reading both outputs.
- **Attribution / dedup / fast-slow split** — deferred (see `docs/status-and-plan.md`
  TODOs 1-3); needed before a second document.
