# HippoRAG 2 — the exact algorithm

A precise, source-grounded account of how HippoRAG 2 **ingests documents, builds
its graph, and searches it.** Every step cites the function in
`hippo/HippoRAG/src/hipporag/HippoRAG.py` (or the named module). Written so we can
decide, concretely, what to keep vs. replace for the Structure layer.

> One-line summary: HippoRAG 2 is a **retrieval engine**. Indexing turns a corpus
> into a *flat, weighted, undirected* graph of **entity** and **passage** nodes
> plus three vector stores; a query spreads relevance from query-matched *triples*
> through that graph with **Personalized PageRank** and returns the **passages**
> that accumulate the most mass. The unit it returns is always a chunk; an LLM
> then reads those chunks to answer.

---

## 0. The data structures it builds

| Store | Contents | Id scheme |
|---|---|---|
| `chunk_embedding_store` | each chunk's text + embedding | `chunk-<md5(text)>` |
| `entity_embedding_store` | each unique entity phrase + embedding | `entity-<md5(phrase)>` |
| `fact_embedding_store` | each unique triple, embedded **as its stringified form** `"['s', 'p', 'o']"` | `entity-…`-namespaced md5 of the string |
| `graph` (igraph) | the one flat graph: entity + passage **nodes**, weighted undirected **edges** | node `name` = the store hash id |

Ids are `prefix + md5(content).hexdigest()` — `compute_mdhash_id`,
[embedding_store.py:11](../hippo/HippoRAG/src/hipporag/embedding_store.py#L11). This
is why our `chunk_decision.py` reuses the same scheme: our chunks join 1:1 onto
HippoRAG's passage nodes for free.

---

## 1. Ingestion / indexing — `index(docs)` ([HippoRAG.py:233](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L233))

Input: a list of passage strings (our 195 chunks).

### Step 1 — Store & embed the chunks
`chunk_embedding_store.insert_strings(docs)` — hash, embed, and store every chunk.

### Step 2 — OpenIE per chunk (two LLM calls each) — `openie.batch_openie`
For each chunk, **two** sequential LLM calls:

1. **NER** ([openie_openai.py:45](../hippo/HippoRAG/src/hipporag/information_extraction/openie_openai.py#L45)).
   Prompt = *"extract named entities … Respond with a JSON list"* + one fixed
   one-shot example (the "Radio City" paragraph) + the passage
   ([templates/ner.py](../hippo/HippoRAG/src/hipporag/prompts/templates/ner.py)).
   → a list of entity strings.
2. **Triple extraction (NER-conditioned RE)** ([openie_openai.py:81](../hippo/HippoRAG/src/hipporag/information_extraction/openie_openai.py#L81)).
   Prompt = *"construct an RDF graph from the passage and the named-entity list.
   Each triple should contain at least one (preferably two) of the named
   entities. Resolve pronouns to specific names."* + the same one-shot example +
   the passage + the NER JSON
   ([templates/triple_extraction.py](../hippo/HippoRAG/src/hipporag/prompts/templates/triple_extraction.py)).
   → a list of `[subject, predicate, object]` triples.

Results are cached to `openie_results.json` (re-runs are free unless chunks/model
change).

### Step 3 — Normalize and collect
- `text_processing` lowercases/normalizes every triple element — this is why the
  graph's entities are lowercase (`huanan seafood market`, not `HSM`).
- `extract_entity_nodes` → the set of unique entity strings (all subjects +
  objects) and, per chunk, the list of its entities.
- `flatten_facts` → the list of unique triples.

### Step 4 — Embed entities and facts
- `entity_embedding_store.insert_strings(entity_nodes)`.
- `fact_embedding_store.insert_strings([str(fact) …])` — **a fact is embedded as
  the string of the whole triple**, e.g. `"['sars-cov-2', 'causes', 'covid']"`.
  This is the object semantic search runs against at query time.

### Step 5 — Build the edges (into `node_to_node_stats`, then materialize)
Three edge-builders populate a `{(node_a, node_b): weight}` dict; then
`augment_graph` turns it into the igraph.

1. **Fact edges** — `add_fact_edges` ([:744](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L744)).
   For each triple, connect **subject-entity ↔ object-entity**, incrementing the
   edge weight by `1.0` each time that pair co-occurs (both directions).
   **⚠️ The predicate is dropped here.** `triple[1]` is never used to build the
   graph — only `triple[0]` and `triple[2]`. So `[sars-cov-2, causes, covid]` and
   `[sars-cov-2, related to, covid]` collapse to the *same* unlabeled entity–entity
   edge with weight 2. Relation meaning survives **only** in the fact text/embedding.
   Also records `ent_node_to_chunk_ids[entity] ∪= {chunk}` (used for IDF
   down-weighting at query time).
2. **Passage edges** — `add_passage_edges` ([:792](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L792)).
   Connect **passage ↔ each entity extracted from it**, weight `1.0`.
3. **Synonymy edges** — `add_synonymy_edges` ([:836](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L836)).
   KNN over entity embeddings; for each entity (with >2 alphanumeric chars),
   connect it to neighbors with cosine ≥ `synonymy_edge_sim_threshold` (**our
   0.95**; ≤100 neighbors), edge weight = the cosine score.

Then `augment_graph` → `add_new_nodes` (adds every entity + passage row as a
vertex) → `add_new_edges` (materializes `node_to_node_stats` as weighted igraph
edges, skipping self-loops and dangling endpoints) → `save_igraph` (pickle).

### The graph you end up with
- **Node types (only two):** `entity` (name `entity-<md5>`, content = lowercased
  phrase) and `passage` (name `chunk-<md5>`, content = chunk text).
- **Edge types (one flat, undirected, weighted graph):**

  | endpoints | meaning | weight |
  |---|---|---|
  | entity–entity | **fact edge** (an OpenIE triple) | # of triples co-mentioning the pair |
  | entity–entity | **synonymy edge** | cosine similarity (≥ threshold) |
  | passage–entity | entity was extracted from that chunk | 1.0 |

  Fact and synonymy edges carry **no distinguishing label** — you separate them
  only by checking membership in `kg_facts.json`.

### What ingestion does *not* build
- **No relation-typed edges.** Predicates are not nodes and not edge labels.
- **No claim / evidence / hypothesis typing.** Everything is an "entity."
- **No hierarchy, no summaries, no communities.** (That's GraphRAG/RAPTOR, not
  HippoRAG.) The graph is one flat layer.
- **No "importance" computed at build time.** There is no centrality pass, no
  node ranking during indexing. (See §3.)

---

## 2. Retrieval — `retrieve(query)` ([:378](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L378))

1. **Embed the query twice** (`get_query_embeddings`) with two different
   instruction prefixes: `query_to_fact` and `query_to_passage` (the e5 model is
   instruction-tuned, so the task instruction matters).
2. **Fact retrieval** — `get_fact_scores` ([:1305](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L1305)):
   dot the *query_to_fact* embedding against **all triple embeddings** →
   min-max-normalized similarity per triple. (Semantic search over triples, not
   chunks — the v2 change vs. v1's entity linking.)
3. **Recognition memory (LLM fact filter)** — `rerank_facts`
   ([:1537](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L1537)): take the top
   `linking_top_k` candidate triples, then a DSPy LLM filter
   ([filter_default_prompt.py](../hippo/HippoRAG/src/hipporag/prompts/filter_default_prompt.py))
   keeps **up to 4** that are genuinely relevant ("only use facts from the
   candidate list; do not invent"). This is the step the paper calls *recognition
   memory*.
4. **Fallback:** if zero facts survive, return pure `dense_passage_retrieval`
   ([:1345](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L1345)) — plain chunk vector
   search. (So on a query with no good triple match, HippoRAG 2 ≈ ordinary RAG.)
5. **Graph search** — `graph_search_with_fact_entities`
   ([:1422](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L1422)):
   - Each surviving fact's **subject & object entities** become PPR seeds, weighted
     by fact score and **divided by how many chunks the entity appears in**
     (IDF-style down-weighting of promiscuous hubs like `hsm`). Keep the top
     `link_top_k` entities.
   - Run dense passage retrieval and give each passage node a *small* seed weight
     = normalized dense score × `passage_node_weight` (**0.05**).
   - `node_weights = entity_weights + passage_weights` becomes the PPR **reset
     (teleport) vector**.
   - `run_ppr` ([:1587](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L1587)):
     igraph `personalized_pagerank(directed=False, weights='weight',
     reset=node_weights, damping=0.5)`. Mass spreads from the seeds across the
     weighted graph (multi-hop), then we read the scores **at the passage nodes**.
6. **Rank passages** by PPR score → top-k chunks → `qa` → an LLM reads those chunks
   and writes the answer.

**Mental model:** the query lights up a few relevant triples; their entities are
"cues"; PPR does associative pattern-completion over the graph (the paper's
hippocampus analogy); passages sitting in the activated neighborhood win. It is a
smart *retriever/re-ranker*, not an answer structure.

---

## 3. "Important nodes" — how importance actually arises

The user asked how HippoRAG builds important nodes. It **doesn't, at build time.**
Importance is **emergent and query-dependent**:
- **Degree** — hub entities (e.g. `hsm`, degree ~250) are structurally central
  because many triples touch them. That's a side effect, not a computed ranking.
- **Query-time PPR** — the only "importance" that drives results is the per-query
  PageRank stationary distribution seeded by the matched facts.
- **IDF-style down-weighting of the PPR seeds.** *IDF* = "inverse document
  frequency," a classic retrieval idea: a term that appears in many documents is
  less informative (think "the"), so you weight it inversely to how many documents
  contain it; rare, specific terms are more discriminating and score higher.
  **PageRank itself has nothing to do with IDF** — what's IDF-like here is *how the
  Personalized PageRank is seeded*: when a matched fact's entity becomes a
  teleport/seed node, its seed weight is **divided by the number of chunks that
  entity appears in** (its "document frequency") —
  [:1478](../hippo/HippoRAG/src/hipporag/HippoRAG.py#L1478). So a hub like `hsm` (in
  dozens of chunks) contributes far less seed mass than a rare, specific entity, and
  generic hubs can't flood the walk. Two caveats: it's IDF-*style* — a linear `1/df`
  division, not the textbook `log(N/df)` — and it's applied to the PPR **reset
  vector**, not to edge weights or the PageRank algorithm itself.

There is no summarization, no community detection, no salience/centrality index
persisted with the graph.

---

## 4. Why this is insufficient for the competition (the gap we target)

For a document whose substance is *"evidence E shifts hypothesis H by Bayes factor
B,"* HippoRAG's ingestion throws the substance away:

1. **Predicates are erased from the graph.** `supports`, `rebuts`, `is evidence
   for` all become the same unlabeled entity–entity edge. The graph literally
   cannot store a relation type.
2. **No epistemic node types.** No hypothesis / claim / evidence / assumption /
   estimate — only "entities."
3. **No probability or aggregation.** No place for a Bayes factor, a prior, or
   log-odds updating; nothing to recompute or do sensitivity analysis on.
4. **No hierarchy or discourse.** No sub-question tree, no argument tree; one flat
   layer.
5. **The output is a chunk, not a claim.** Even a perfect retrieval hands you a
   passage to read, not an assessable node in an argument.

None of these are bugs — HippoRAG is doing its job (retrieval) well. They're just
orthogonal to what FLF's Structure and Assessment layers ask for, which is exactly
where a typed, hierarchical epistemic graph adds value **on top of** HippoRAG's
retrieval. See [`../CONTEXT.md`](../CONTEXT.md) §5.
