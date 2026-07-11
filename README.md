# flf-epistack-competition

Building a three-layer epistemic investigation system, using
[HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) as the baseline for the
lower layers.

**Case study:** `eric_decision.pdf` — Eric Stansifer's judgment in the Rootclaim
covid-origins debate (2023-12-18, 83pp). It was chosen because it is a
*reasoning* artifact rather than a fact corpus: competing hypotheses (zoonosis vs
lab leak), explicit priors, Bayes factors, rebuttals, and a final verdict.

## What's here

The document is parsed, and a HippoRAG 2 knowledge graph is built and published.

| | |
|---|---|
| `data/` | the parse — 195 section-aware chunks + the 64-section tree |
| `artifacts/hipporag2/` | the knowledge graph — 1,673 nodes, 2,169 triples |
| `scripts/` | the pipeline, plus `load_graph.py` to read the results |
| `TASKS.md` | task tracker and running notes |

**→ [`artifacts/README.md`](artifacts/README.md) is the guide to reading and
working with the data.** Start there.

```sh
pip install networkx
python scripts/load_graph.py     # no API key, no HippoRAG install needed
```

## Reproducing the pipeline

The HippoRAG clone is deliberately not committed. Clone it yourself:

```sh
git clone https://github.com/OSU-NLP-Group/HippoRAG hippo/HippoRAG
cp .env.example .env             # add a TOGETHER_API_KEY
python scripts/chunk_decision.py       # PDF    -> data/chunks.json
python scripts/run_hipporag_index.py   # chunks -> OpenIE -> knowledge graph
```

Upstream HippoRAG is left unmodified; the compatibility fixes live as
monkeypatches in `scripts/run_hipporag_index.py`. See `artifacts/README.md` for
what they are and, importantly, why the synonymy threshold is 0.95 rather than
HippoRAG's default 0.8 — that number silently destroys the graph if the
embedding model changes.

## Status

Layers 1 and 2 (parse → HippoRAG 2 KG) are done. Layer 3 — our own hierarchical,
*epistemic* graph — is next. The motivation for it is visible in the current
graph's output: a representative triple is `['HSM', 'related to', 'outbreak']`.
It is topically right and epistemically empty. Nothing in the graph can say
"this evidence supports that hypothesis with Bayes factor X", which for this
document is the entire substance.
