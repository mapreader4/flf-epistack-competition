# Three-layer epistemic investigation — task tracker

Case study: `eric_decision.pdf` — Eric Stansifer's judgment in the Rootclaim
covid-origins debate (2023-12-18, 83pp). A long-form reasoned decision that
weighs competing hypotheses (zoonosis vs lab leak) under explicit Bayesian
argument. Chosen because it is a *reasoning* artifact, not just a fact corpus:
it has claims, evidence, Bayes factors, rebuttals, and a final verdict.

Goal: build our own three-layer epistemic system, using HippoRAG 2 as the
baseline/reference implementation for the lower layers.

---

## Status

| # | Task | State |
|---|------|-------|
| 1 | Environment + deps (HippoRAG, minus vllm) | ✅ done |
| 2 | Parse PDF → section-aware chunks | ✅ done |
| 3 | HippoRAG phase 1: OpenIE (NER + triples) | ✅ done |
| 4 | HippoRAG phase 2: knowledge-graph construction | ✅ done |
| 5 | Our own hierarchical graph (layer 3) | ⏳ awaiting spec from Amir |

Everything runs on the **Together key alone** (`.env`). No OpenAI key needed.

---

## What exists now

```
data/chunks.json      195 chunks, 47k tokens, section-aware  (task 2 output)
data/sections.json    64-section tree parsed from the TOC
scripts/chunk_decision.py       PDF → chunks
scripts/run_hipporag_index.py   chunks → OpenIE → HippoRAG KG  (tasks 3+4)
outputs/hipporag/
  openie_results_ner_*.json     per-chunk NER + triples        (task 3 output)
  graph/kg_nodes.json           1,673 nodes                    (task 4 output)
  graph/kg_edges.json           8,950 edges
  graph/kg_facts.json           2,169 triples
  graph/kg.graphml              same graph, for Gephi/networkx
  graph/kg_stats.json
  llm_cache/*.sqlite            cached LLM calls — re-runs are free
.venv/                Python 3.11, HippoRAG deps installed
hippo/HippoRAG/       upstream clone, UNMODIFIED (all fixes are monkeypatches)
```

### Task 1 — environment ✅
- Python 3.11 venv via `uv`. HippoRAG imports clean.
- **vllm excluded** — it has no usable Mac build and we don't need it (we call
  Together's hosted API instead of self-hosting).
- `numpy` pinned `<2`: torch 2.5.1 is built against numpy 1.x and throws
  `Failed to initialize NumPy: _ARRAY_API not found` otherwise.

### Task 2 — chunking ✅
`python scripts/chunk_decision.py` → 195 chunks / 47,120 tokens, mean 241 tok.

The document has a real table of contents, so we parse **that** as the
authoritative section tree (64 sections, all 64 located in the body) rather than
regex-guessing headings from the body. Chunks never cross a section boundary,
and each carries `section_number`, `section_title`, `section_depth`,
`parent_section`, `is_appendix`, and `page`.

Chunk ids are `chunk-<md5(text)>` — **the same scheme HippoRAG's EmbeddingStore
uses**, so our section metadata joins onto HippoRAG's graph nodes for free. This
is what makes layer 3 possible without forking the library.

*Known limitation:* pdf extraction interleaves footnote text mid-sentence in
places, so a few chunks have a spliced sentence. Doesn't break OpenIE, but worth
revisiting if triple quality suffers.

### Tasks 3+4 — HippoRAG OpenIE + KG ✅
```
python scripts/run_hipporag_index.py --limit 6   # smoke test
python scripts/run_hipporag_index.py             # full run (done)
```
Both phases run in one `HippoRAG.index()` call: OpenIE per chunk → embed
passages/entities/facts → fact edges → passage edges → synonymy edges → igraph.

**Config that actually shipped** (differs from what we first picked — see below):
- OpenIE LLM: `meta-llama/Llama-3.3-70B-Instruct-Turbo` @ Together
- Embeddings: `intfloat/multilingual-e5-large-instruct` @ Together (1024-dim)
- Synonymy threshold: **0.95** (HippoRAG's default is 0.8)

**Results.** Phase 1: 195/195 chunks extracted, mean 11.5 triples/chunk, 0
malformed triples, 1 zero-triple chunk. Phase 2: 1,673 nodes (1,478 entities +
195 passages), 8,950 edges (6,588 entity–entity, 2,362 entity–passage), 2,169
facts, entity-graph density 0.003.

**Verified:** all 195 of our chunk ids join 1:1 onto HippoRAG's passage nodes.
Zero unmatched on either side. Layer 3 can hang off this directly.

#### Three upstream problems we had to work around
Upstream is left **unmodified**; every fix is a monkeypatch in the runner.

1. **Two providers, one env var.** HippoRAG builds *both* its LLM client and its
   embedding client as `OpenAI(...)` with no explicit `api_key`, so both read
   `OPENAI_API_KEY`. Two providers can't share it. The runner gives the env var
   to the embedding client and injects Together's key into the LLM client after
   construction. `OpenIE` holds a reference to the same `llm_model`, so one swap
   covers both. (Moot now that everything is on Together, but it keeps the
   OpenAI-embeddings path open.)
2. **Embedding model allowlist.** `_get_embedding_model_class` asserts on any
   name it doesn't recognize, which rejects Together's models. They speak the
   OpenAI embeddings protocol, so we patch the lookup to route `BAAI/`,
   `intfloat/`, `togethercomputer/` to `OpenAIEmbeddingModel`. Note the patch
   must import the class as `from src.hipporag.HippoRAG import HippoRAG` — the
   submodule import shadows the package's lazy `__getattr__` class export.
3. **The synonymy threshold is embedding-specific and 0.8 is wrong for e5.**
   This one silently destroys the graph. e5's cosine *floor* across unrelated
   entities is ~0.77 and its median ~0.84, so HippoRAG's default 0.8 marks
   **93.7% of all entity pairs** as synonyms — a near-complete graph (the 6-chunk
   smoke test produced 1,337 synonymy edges over 40 entities). Genuine synonyms
   here sit at ≥0.96 (`huanan seafood market` ↔ `huanan seafood wholesale
   market` = 0.983). We set 0.95. **If you swap the embedding model, re-measure
   this** — the right value is a property of the embedding, not of the document.

*Together model availability:* only `intfloat/multilingual-e5-large-instruct` is
serverless on this account. Both `BAAI/bge-*` and `togethercomputer/m2-bert-*`
return `model_not_available` and would need a dedicated endpoint.

### Task 5 — our hierarchical graph ⏳
Awaiting Amir's spec. What we have in hand to build on:

- HippoRAG's KG is **flat**: entity nodes + passage nodes, one synonymy-linked
  layer. No notion of claim, evidence, or argumentative role — `['HSM',
  'related to', 'outbreak']` is a typical triple, which is topically right and
  epistemically empty. It cannot represent "this evidence *supports* that
  hypothesis with Bayes factor X".
- We retain the **document hierarchy** (64-section tree) and it joins onto
  HippoRAG's nodes via `chunk-<md5>` (verified 195/195).
- The decision's own structure suggests the epistemic layer: hypotheses
  (zoonosis / lab leak), priors, evidence items, Bayes factors, rebuttals,
  verdict — §7 (Analysis) is where the judge aggregates everything, and §C is
  explicitly *excluded* information.

---

## Skills / knowledge needed

- **HippoRAG 2 internals** — `HippoRAG.index()`, `information_extraction/openie_openai.py`,
  `embedding_store.py` (md5 ids), `HippoRAG.py::add_fact_edges/add_synonymy_edges`.
- **OpenIE prompt engineering** — HippoRAG's NER/triple prompts are GPT-tuned;
  on Llama they may need adjusting. Prompts live in `src/hipporag/prompts/`.
- **Graph libs** — igraph (HippoRAG's) and networkx (likely ours).
- **Argument mining / epistemics** — for layer 3: claim–evidence–warrant
  structure, Bayesian argumentation, defeasible reasoning.

## Cost notes
The full run was ~390 Together calls (195 chunks × NER + triples) plus
embeddings — cents, not dollars. HippoRAG caches every LLM response in
`outputs/hipporag/llm_cache/*.sqlite` keyed on (messages, model, seed,
temperature), so **re-running the index is free and instant** unless the chunks
or the model change. Deleting that sqlite is what forces a real re-extraction.
