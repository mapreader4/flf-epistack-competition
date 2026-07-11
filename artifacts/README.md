# Artifacts: parsed decision + HippoRAG 2 knowledge graph

Everything here is derived from `eric_decision.pdf` — Eric Stansifer's judgment
in the Rootclaim covid-origins debate (2023-12-18, 83pp), the case study for our
three-layer epistemic system.

**You do not need HippoRAG, an API key, or a GPU to use any of this.** The files
are plain JSON. `scripts/load_graph.py` reads them with only `networkx`.

```sh
pip install networkx
python scripts/load_graph.py
```

---

## Layout

```
data/
  chunks.json                 195 section-aware chunks  (the parse)
  sections.json               64-section tree from the PDF's table of contents
artifacts/hipporag2/
  openie_results.json         per-chunk entities + triples   (HippoRAG phase 1)
  kg_nodes.json               1,673 nodes                    (HippoRAG phase 2)
  kg_edges.json               8,950 directed edge entries
  kg_facts.json               2,169 triples
  kg.graphml                  same graph, for Gephi / networkx / igraph
  kg_stats.json               summary counts
  run_config.json             provenance: models, thresholds, how to reproduce
```

Not published: embedding vectors (31 MB, regenerable) and the LLM response cache.
Rerun `scripts/run_hipporag_index.py` to regenerate both.

---

## The one thing to understand: chunk ids are the join key

Every chunk id is `chunk-<md5(chunk_text)>`. That is **HippoRAG's own hashing
scheme**, not ours. Consequence: `data/chunks.json` joins 1:1 onto the passage
nodes in `kg_nodes.json` — verified, 195/195, zero unmatched on either side.

So the document hierarchy (which section, which page, which parent section) is
recoverable for any node in the graph, even though HippoRAG itself has no concept
of a section. This is what makes a hierarchical layer possible on top.

```python
chunks = {c["chunk_id"]: c for c in json.load(open("data/chunks.json"))}
nodes  = json.load(open("artifacts/hipporag2/kg_nodes.json"))

for n in nodes:
    if n["type"] == "passage":
        sec = chunks[n["id"]]["section_number"]   # e.g. "4.5.3"
```

---

## File formats

### `data/chunks.json`
```json
{
  "chunk_id": "chunk-7ccd048f...",   // = HippoRAG passage node id
  "text": "When the \"novel coronavirus\" started dominating...",
  "section_number": "4.5.3",          // "A.1" etc. for appendices
  "section_title": "Empirical evidence of market outbreaks",
  "section_depth": 3,                 // 1 = top-level section
  "parent_section": "4.5",            // null at top level
  "is_appendix": false,
  "chunk_order_in_section": 0,
  "page": 39,
  "n_tokens": 241
}
```
Chunks never cross a section boundary. Mean 241 tokens, max 331.

### `artifacts/hipporag2/kg_nodes.json`
```json
{ "id": "chunk-7ccd048f...", "type": "passage", "content": "<full chunk text>" }
{ "id": "entity-a1b2c3...",  "type": "entity",  "content": "huanan seafood market" }
```

### `artifacts/hipporag2/kg_edges.json`
```json
{ "source": "entity-a1b2...", "target": "entity-d4e5...", "weight": 1.0 }
```
Three kinds of edge are mixed together here, distinguishable by endpoint type:

| endpoints | meaning |
|---|---|
| entity–entity | a **fact edge** (an OpenIE triple) *or* a **synonymy edge** |
| entity–passage | the entity was extracted from that passage |

⚠️ **The graph is undirected and each edge is stored once per direction.** So
8,950 entries in the file collapse to **5,678 unique undirected edges** in
networkx. Both numbers are correct; don't be alarmed when they disagree.

Fact and synonymy edges are not separately labelled by HippoRAG. To tell them
apart, check whether the pair appears in `kg_facts.json`.

### `artifacts/hipporag2/kg_facts.json`
```json
{ "id": "fact-...", "triple": "['sars-cov-2', 'causes', 'covid']" }
```
Note `triple` is a **string** holding a Python-literal list, not a JSON array —
that is how HippoRAG stores it. Parse with `ast.literal_eval`.

### `artifacts/hipporag2/openie_results.json`
Phase-1 output, before any graph was built. Useful if you want to build a
*different* graph from the same extraction without paying for re-extraction.
```json
{ "docs": [ { "idx": 0, "passage": "...",
              "extracted_entities": [...], "extracted_triples": [[s, p, o], ...] } ] }
```

---

## How it was built

| | |
|---|---|
| OpenIE LLM | `meta-llama/Llama-3.3-70B-Instruct-Turbo` (Together), temp 0, seed 0 |
| Embeddings | `intfloat/multilingual-e5-large-instruct` (Together), 1024-dim |
| Synonymy threshold | **0.95** |
| Pipeline | HippoRAG 2, upstream `main`, **unmodified** |

Reproduce:
```sh
python scripts/chunk_decision.py        # PDF  -> data/chunks.json
python scripts/run_hipporag_index.py    # chunks -> OpenIE -> KG
```
The HippoRAG clone lives at `hippo/` and is deliberately **not** committed —
clone it yourself from OSU-NLP-Group/HippoRAG. Our two compatibility fixes are
monkeypatches in `scripts/run_hipporag_index.py`, so upstream stays pullable.

### Why the synonymy threshold is 0.95 and not HippoRAG's default 0.8
This is the one number that will silently ruin the graph if you change the
embedding model without re-checking it.

HippoRAG's 0.8 default is calibrated for GritLM / OpenAI embeddings. With e5, the
cosine similarity *floor* between completely unrelated entities is ~0.77 and the
median is ~0.84 — so a 0.8 cutoff marks **93.7% of all entity pairs** as
synonyms, turning the graph into a near-clique. (A 6-chunk smoke test produced
1,337 synonymy edges over just 40 entities.)

Genuine synonyms in this corpus sit at ≥0.96:

| pair | cosine |
|---|---|
| `huanan seafood market` ↔ `huanan seafood wholesale market` | 0.983 |
| `covid` ↔ `covid 19` | 0.981 |
| `lab leak` ↔ `lab leak hypothesis` | 0.980 |

At 0.95 the entity-graph density is 0.003, which is sane. **The right threshold
is a property of the embedding model, not of the document** — re-measure it if
you swap models.

---

## Known limitations

1. **The graph is topically right and epistemically empty.** A representative
   triple is `['HSM', 'related to', 'outbreak']`. There is no way to express
   *"this evidence supports that hypothesis with Bayes factor X"* — no claims, no
   evidence roles, no support/rebut edges, no priors. For a document that is
   literally a Bayesian adjudication between two hypotheses, the substance is
   exactly what OpenIE drops. This is the motivation for the layer we are
   building on top.
2. **Footnote splicing.** PDF extraction interleaves footnote text mid-sentence
   in a handful of chunks, so a few have a spliced sentence.
3. **Entity aliasing is partial.** `hsm` (degree 250) and `huanan seafood market`
   remain separate nodes — the abbreviation is not embedding-similar to the
   expansion. Anything doing entity-level aggregation should alias these first.
4. **22 chunks yielded no NER entities** (triples were still extracted). Mostly
   formula-heavy and reference-list chunks.
