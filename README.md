# flf-epistack-competition

A **typed epistemic graph** over a document's argument — not "chunks that mention X," but
hypotheses, evidence, assumptions, and the explicit **support / attack** relations between
them, each grounded in a byte-exact source span, with a strength score for every claim.
Built for the FLF Epistemic Case Study Competition and judged on *generality*, so nothing
in the graph or query path is hard-coded to any one document.

**Case studies:** `eric_decision.pdf` and `will_decision.pdf` — two judges' Bayesian
rulings on the COVID-19 origins debate (zoonosis vs. lab leak). We build a graph per
document and a combined **two-document** graph that reasons *across* the two decisions.

## Quickstart — no API key needed

```sh
git clone <repo> && cd flf-epistack-competition
bash demo.sh
```

`demo.sh` builds a tiny virtualenv and runs two queries on the **pre-built two-document
graph** committed in this repo: the evidence for a hypothesis aggregated across *both*
judges, and the nodes where the two judges disagree. No API key, no GPU, no rebuild — it
reads the committed artifacts and prints structured, provenance-carrying results.

## What the system does

Three layers:

- **Ingestion** — PDF → section-aware chunks → atomic, provenance-anchored claims.
- **Structure** — claims become typed nodes (10 types grouped into 3 layers:
  *data / argument / question*); pairs are labeled into reified **support (RA)** and
  **attack (CA)** cards; a discontinuity-free QuAD (DF-QuAD) semantics scores every node
  in [0,1].
- **Query** — traverse the scored graph: *evidence-for*, *contested* (where sources
  disagree), *weakest-link* (ablation). Answers are ranked evidence chains with quotes and
  section numbers, not chunks.

HippoRAG 2 is the **baseline** (a strong multi-hop passage retriever), not a component: its
graph encodes topical association, not argument, so it cannot express support/attack or
score competing hypotheses.

## Results (built, under `artifacts/`)

| graph | nodes | cards | highlights |
|---|---|---|---|
| single-doc (`artifacts/epistemic/`) | 1,370 | 1,253 (RA 907 / CA 346) | 93 contested nodes; depth-2 evidence chains |
| two-doc (`artifacts/epistemic_2doc/`) | 2,232 (eric 2,077 + will 155) | 1,678 (RA 1,382 / CA 296) | **230 cross-document cards**; 31 cross-doc contested; 4 auto-selected top-level hypotheses |

Every top-level hypothesis is supported by evidence from **both** judges; agreement and
disagreement between them are explicit graph structure rather than something a reader must
infer. `python scripts/verify_2doc.py --data-dir artifacts/epistemic_2doc` prints the full
cross-document breakdown.

## Reproduce the pipeline (needs a Together API key)

```sh
pip install -r requirements.txt
cp .env.example .env          # add TOGETHER_API_KEY   (LLM + embeddings via Together's OpenAI-compatible API)
```

Two-document build — extraction → convert → embed → pair → label → score → verify:

```sh
# 1. extract typed nodes from both PDFs (7 types) + two-tier dedup
python scripts/extraction-variants/epistemic_node_extraction_multidoc.py \
    --documents config_2doc.json --model openai/gpt-oss-120b --require-nli
# 2. convert to the graph schema (+ autonomous top-level-hypothesis selection)
python scripts/multidoc_to_graph_nodes.py \
    --in artifacts/nodes_multidoc_2doc/nodes_combined_raw.json \
    --out artifacts/epistemic_2doc/nodes.jsonl --authors '{"eric_decision.pdf":"Eric ...","will_decision.pdf":"Will ..."}'
# 3. embed
python scripts/embed_nodes.py --nodes artifacts/epistemic_2doc/nodes.jsonl \
    --out artifacts/epistemic_2doc/embeddings.npy --index artifacts/epistemic_2doc/embeddings_index.json
# 4. candidate pairs — DeBERTa/NLI reranking (GPU)
python scripts/pairing_funnel.py --nli --no-label --nodes artifacts/epistemic_2doc/nodes.jsonl \
    --emb artifacts/epistemic_2doc/embeddings.npy --emb-index artifacts/epistemic_2doc/embeddings_index.json \
    --out artifacts/epistemic_2doc/candidate_pairs.json --max-llm-pairs 3000
# 5. label -> cards  (SEPARATE cache: 2-doc node ids overlap the single-doc space)
python scripts/label_pairs.py --pairs artifacts/epistemic_2doc/candidate_pairs.json \
    --nodes artifacts/epistemic_2doc/nodes.jsonl --out artifacts/epistemic_2doc/cards.jsonl \
    --cache outputs/epistemic_2doc/label_cache.json
# 6. score + 7. verify
python scripts/score_dfquad.py --nodes artifacts/epistemic_2doc/nodes.jsonl \
    --cards artifacts/epistemic_2doc/cards.jsonl --out artifacts/epistemic_2doc/scores.jsonl
python scripts/verify_2doc.py --data-dir artifacts/epistemic_2doc
```

For a single document, `pairing_funnel.py` runs the whole middle in one command
(retype → 4-channel candidates → label → merge). The HippoRAG baseline is not vendored:
`git clone https://github.com/OSU-NLP-Group/HippoRAG hippo/HippoRAG`, then
`python scripts/run_hipporag_index.py`.

## More

- `artifacts/README.md` — guide to reading the data.
- `CLAUDE.md` / `CONTEXT.md` / `TASKS.md` / `WORKLOG.md` — how the code runs and the
  decision log.
- `report/` — the write-up.
