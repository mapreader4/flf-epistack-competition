# Epistemic Graph — System Design & Context (report reference)

A single, current, ground-truthed reference for writing the report. Numbers verified
against the artifacts on disk as of this writing.

---

## 1. Problem & framing

**Competition.** FLF "Epistemic Case Study Competition." The working fixture is
`eric_decision.pdf` — a judge's (Eric Stansifer's) 83-page Bayesian adjudication of the
COVID-19 origins debate (zoonosis vs. lab leak). The system is judged on **generality**:
it must work on other cases (LHC/black holes, eggs, …), so nothing may be hard-coded to
this document.

**What we build.** A **typed epistemic graph** over a document's claims: not "chunks that
mention X," but a structured argument — hypotheses, evidence, assumptions, and the
**support/attack relations** between them, each grounded in a byte-exact source span.

**Baseline (not a component).** HippoRAG 2 is the comparison. It builds an open-domain
knowledge graph of entities/relations and does retrieve-then-read: it ranks chunks and
feeds chunk text to one LLM call — the graph never reaches the answering model, and it
holds no notion of "supports / attacks / weighs against." Our graph does.

**Why a graph beats retrieve-then-read for epistemic questions.** Questions like "what's
the evidence for X," "where do the sources disagree," and "what's the weakest link" are
about *argument structure*, not topical similarity. A vector/keyword retriever returns
paragraphs that mention the topic; it cannot tell a supporting datum from a rebuttal, nor
trace a conclusion back through its premises. The typed graph makes that structure a
first-class, queryable object.

---

## 2. Three layers

```
INGESTION   PDF -> section-aware chunks -> atomic claims (byte-exact provenance)
STRUCTURE   claims -> TYPED nodes -> pairwise support/attack CARDS -> DF-QuAD scores
QUERY       traverse the scored graph; hand the LLM a ranked evidence chain, not chunks
```

The novelty is the **Structure layer**: a reified argument graph with a principled
scoring semantics, sitting between raw extraction and the answer.

---

## 3. Node schema (the typed store)

Container: `scripts/epistemic_store.py` (dataclasses + JSONL I/O + validator; no LLM).

A **Node**: `{node_id ("n-00000"), type, canonical_text, fingerprint ("fp-…", a dedup
hash, separate from the id), provenance[], payload, tier, meta}`.

A **Provenance** grounds a node in a source: `{claim_id, chunk_id, section_number, quote,
span [byte offsets], tier, document}`. `document` is the multi-document hook (null on the
single-doc corpus; set explicitly per source in the 2-doc build).

**10 node types**, grouped into a coarser **3-layer** vocabulary the pairing funnel blocks
on (`LAYER_OF`):

| layer | node types | role |
|---|---|---|
| **data** | evidence, estimate, background, correlation_group | observations / numbers / context |
| **argument** | claim, rebuttal, assumption | interpretive / inferential propositions |
| **question** | hypothesis, verdict, subquestion | what the debate is deciding |

**Tiers** mark trust: **T1** grounded on a byte-exact span, **T2** parsed from a chunk,
**T3** model-inferred.

---

## 4. Card schema & DF-QuAD scoring

A **Card** is a reified relation: `{card_id, kind, weight, premises[node_ids], target,
active, provenance{relation_label, subtype, channel, labeler_model}, tier}`.
- **kind**: **RA** = support, **CA** = attack (PA/outweighs reserved). Cards are n-ary —
  a joint relation needs *all* premises.

**DF-QuAD** (Discontinuity-Free QuAD), `scripts/score_dfquad.py`, pure arithmetic, no LLM:
```
card_strength(c)   = weight(c) * min(strength[p] for p in premises)      # joint = weakest link
aggregate(S)       = 1 - Π(1 - s)                                        # independent cards combine
vs(node)           = aggregate(strengths of active RA cards targeting it)
va(node)           = aggregate(strengths of active CA cards targeting it)
strength(node)     = base + (1-base)(vs-va)   if vs >= va               # base = 0.5
                     base - base(va-vs)        otherwise
```
Nodes are processed in dependency order via Tarjan SCCs (cycles iterated to a fixed
point). Output `scores.jsonl`: per node `{strength, base, vs, va, in_supporters,
in_attackers, scc_id, iterations}`.

---

## 5. The pipeline — one command per stage

```
embed_nodes.py  ->  pairing_funnel.py  ->  score_dfquad.py  ->  query_epistemic*.py
   (embeddings)     (retype + candidates    (DF-QuAD           (evidence-for / contested
                     + label + merge cards)   strengths)         / weakest-link + prose)
```
`pairing_funnel.py` is the whole middle in one command (retype sweep → candidate
generation → cap → **label** via `label_pairs.label_pairs()` → **merge** cards). It was
consolidated from five hand-sequenced scripts so a teammate runs one thing, not five.
`--no-label` stops after candidates (for the GPU/DeBERTa split); `--new-nodes` does
incremental ingest of one document against an existing graph.

---

## 6. The pairing funnel — top-down / bottom-up (the depth story)

The expensive step is labeling (one LLM call per pair), so the funnel picks the best
candidate pairs under a fixed budget `MAX_LLM_PAIRS`. Type-based blocking is **not
symmetric** — the budget splits across four **channels**, each a direction through the
layers:

| channel | budget | direction |
|---|---|---|
| **top-down** | 50% | hypothesis ← nearest evidence (evidence → question) |
| **bottom-up** | 30% + 15% | data → argument, then argument → question |
| **contradiction** | 5% | within-layer, NLI-gated (DeBERTa; cosine can't tell agree from contradict) |

Rules: question↔question never paired; data↔data / arg↔arg only via the NLI contradiction
channel. Candidates ranked by cosine (+ optional DeBERTa NLI rerank on GPU), capped, then
labeled.

**The depth lesson (FIX 1 + FIX 2).** A query for "evidence for hypothesis H" follows
support cards inward from H. Two failures made the chain one hop deep:
1. **Interpretive conclusions were mis-typed as `evidence`/data** ("the sample
   distribution points to shop 6:29"). Data↔data pairing is blocked, so raw counts could
   never attach beneath them — the conclusion sat as a childless leaf. **FIX 1 = a retype
   sweep** moving interpretive conclusions data → argument, which unblocks the bottom-up
   `data→argument` channel. (Now built into `pairing_funnel.retype_bridges`.)
2. **Bridges that didn't connect upward "floated."** **FIX 2** pairs the argument bridges
   to their hypotheses (`argument→question`), closing the chain.

On the single-doc fixture, the flagship hypothesis went from **7 summary leaves (0 raw
data, depth 1)** to **11 supporters over 22 raw-data nodes (depth 2)**, and became
correctly **contested** (support *and* attack) rather than one-sidedly high. The limit:
you cannot force a relation the labeler won't affirm — some raw counts have no argument
they dialectically support, which is an *extraction* gap (missing interpretive bridge
nodes), not a funnel gap.

---

## 7. The query lane

`scripts/query_epistemic.py` (single-doc) / `query_epistemic_single_multidoc.py`
(document-aware). Three query types, all traversing the scored graph then handing the LLM
a **ranked evidence chain with provenance** (not chunks):
- **evidence-for X** — follow `in_supporters` inward recursively; returns the support
  chain with strengths and quotes. Switches to `grouped_by_document` mode when the chain
  spans more than one source.
- **contested** — nodes with both supporters and attackers; reports which side wins and
  which documents each side draws from.
- **weakest-link X** — ablation: remove each card, report how much X's score moves.

`resolve_document(node)` reads `provenance[].document` first (falls back to the single
default doc via chunk_id) — this is what makes cross-document answers possible.

---

## 8. Two-document extension (eric + will)

Goal: one graph spanning **two independent judges' decisions** on the same debate
(`eric_decision.pdf` + `will_decision.pdf`), to answer where the judges **agree/disagree**.

**8.1 Multi-document extraction** — `scripts/extraction-variants/
epistemic_node_extraction_multidoc.py` (teammate mapreader4). Extracts all 7 node types
from multiple PDFs (config lists them), then a **two-tier dedup**: Tier 1 (span overlap +
rapidfuzz text-sim), Tier 2 (embedding cosine + **bidirectional NLI entailment**), both
vetoed by a numeric-conflict guard (won't merge "30 killed" vs "47 killed"). Dedup spans
document *and* type.

**8.2 will ingestion** — the engine ingests PDFs via `chunk_decision.recover_document`,
which needed a dotted TOC and space-preserving text. `will_decision.pdf` has neither
(no TOC, pypdf drops spaces). Additive edits to `chunk_decision.py`: an `extraction_mode`
("layout" recovers spaces) and a `flat_sections()` splitter driven by a `heading_regex`
("^Section N"). eric's path is byte-identical.

**8.3 Converter** — `scripts/multidoc_to_graph_nodes.py` maps the engine's Lineage-B
output (`node_type`/`node_text`/hash-id) to the graph Node schema:
- **Type map**: research_question→subquestion, hypothesis→hypothesis, evidence→evidence,
  analysis→claim, quantitative_result→estimate (+`payload.number`), assumption→assumption,
  limitation→rebuttal.
- **Document stamping**: every provenance gets its source `document` (asserted non-null,
  or the query lane would mis-attribute).
- **Dedup groups** become one node with **one provenance entry per member**, so a merged
  cross-document fact carries quotes from *both* pdfs.
- **Autonomous top-level-hypothesis selection (defect 3a, option i)**: the extractor
  over-tags speculative sub-claims as hypotheses (260 for eric). Instead of a fragile
  regex, **one gpt-oss call per document** reads every hypothesis statement and returns
  the genuine top-level competing ones; the rest are demoted to `claim`. Temperature 0 +
  on-disk cache → reproducible; no patterns → generalizes to any topic (eggs, LHC). On
  this corpus it kept **4** (eric: lab-leak, animal-host-at-HSM; will: zoonotic-more-
  likely, index-case-lab-worker) and demoted 295.

**8.4 Cross-document queries** — because every node is document-attributed, `evidence-for`
on a shared hypothesis returns `grouped_by_document` with premises from both docs; and
`contested` surfaces nodes where Eric and Will (or the sides they quote) disagree. We
deliberately built on the **pre-dedup** nodes so both judges' versions of a shared fact
stay separate and attributed — agreement then appears as explicit **cross-document support
cards** rather than a collapsed node. `verify_2doc.py` runs all these checks in one shot.

---

## 9. Key design decisions & rationale

- **Typed reified cards, not edges.** A joint attack (premises C2∧C3 defeat C4) needs an
  n-ary relation with its own weight; two edges can't express it. Cards also carry the
  DF-QuAD weight and a soft-delete flag.
- **3-layer blocking.** Cuts the O(N²) pair space to the structurally-sensible directions
  and gives the funnel its channels; the depth fixes live here.
- **Fixed LLM budget.** Labeling cost is constant regardless of corpus size (590 or 100k
  nodes) — only the funnel's filtering aggressiveness changes.
- **Retype over re-extract.** FIX 1 corrects mis-typing at the graph layer (idempotent
  sweep) rather than re-running extraction.
- **LLM for judgment, code for mechanics.** Hypothesis selection is a contextual judgment
  → one LLM call; scoring/traversal/dedup-keys are deterministic code.
- **Generality first.** No document-specific patterns in the graph/query path;
  per-document knobs (TOC, headings, description) live in config.

---

## 10. Current results (verified on disk)

**Single-document graph** (`artifacts/epistemic/`):
- **1370 nodes** — evidence 380, claim 397, background 311, estimate 83, assumption 75,
  hypothesis 53, rebuttal 51, verdict 20.
- **1253 cards** — RA 907 / CA 346; channels: top_down 621, data→argument 419,
  argument→question 213. 93 contested nodes.
- Flagship hypothesis (HSM zoonosis) depth-2, contested, evidence chain reaches the raw
  §4.6.1 environmental-sampling data (post FIX 1/2).

**Two-document graph** (`artifacts/epistemic_2doc/`) — **complete**:
- **2232 nodes** — eric 2077 + will 155. Types: claim 567, assumption 426, evidence 396,
  estimate 380, rebuttal 310, subquestion 149, hypothesis 4 (LLM-selected top-level).
  0 nodes with a missing document.
- **1678 cards** (RA 1382 / CA 296; channels top_down 689, data→argument 633,
  argument→question 300, contradiction 56; 56% hit rate). **230 cross-document cards**
  (will→eric 109, eric→will 121).
- **All 4 top-level hypotheses are supported across BOTH documents** (`grouped_by_document`):
  e.g. eric's "animal host at HSM" has 47 supporters (45 eric + 2 will); will's "zoonotic
  more likely" has 22 (16 eric + 6 will).
- **82 contested nodes, 31 spanning both documents** — the zoonosis hypotheses net-supported,
  the lab-leak-index-case claim net-attacked (both judges land on zoonosis, shown as
  structure). DeBERTa pairing was run on GPU by teammate Arpita; labeling/scoring/verify
  ours.

---

## 11. Known limitations

- **Extraction-node gap (defect 3a family).** Extraction yields raw data + final
  conclusions but few intermediate interpretive bridge nodes, so some raw counts have no
  argument to attach to. Fix is in the extraction prompt, not the funnel.
- **gpt-oss truncation.** The 2-doc extraction used `--max-tokens 4000`; long sections hit
  the cap and lost tail nodes (~20 truncation warnings, reported for transparency).
- **Cross-document dedup pending.** Built on pre-dedup nodes; Tier-2 NLI dedup is slow on
  CPU and optional. Folding it in is a 2-minute re-convert.
- **Quote noise (will).** Layout-mode extraction leaves padded spaces / occasional glued
  tokens; canonical_text is clean, provenance quotes are whitespace-normalized.

---

## 12. Script reference (what runs what)

| script | stage |
|---|---|
| `chunk_decision.py` | PDF → section-aware chunks (now: layout-mode + flat-heading support) |
| `extraction-variants/epistemic_node_extraction_multidoc.py` | multi-doc typed extraction + two-tier dedup |
| `multidoc_to_graph_nodes.py` | Lineage-B → graph Node schema; doc-stamp; autonomous hypothesis selection |
| `type_claims.py` | single-doc: claims → typed nodes |
| `embed_nodes.py` | nodes → embeddings (multilingual-e5-large-instruct, 1024-d) |
| `pairing_funnel.py` | retype + 4-channel candidates + label + merge → candidate_pairs + cards |
| `label_pairs.py` | (library + CLI) label candidate pairs → cards via gpt-oss |
| `score_dfquad.py` | cards → DF-QuAD strengths |
| `query_epistemic.py` / `query_epistemic_single_multidoc.py` | evidence-for / contested / weakest-link (+ prose) |
| `compare_vs_baseline.py` | graph-guided answer vs full-document LLM baseline |
| `verify_2doc.py` | all Step-7 cross-document checks in one shot |
| `epistemic_store.py` | Node/Card/Provenance schema, layers, DF-QuAD-side helpers, validator |

**Models.** Default LLM `openai/gpt-oss-120b` (Together); embeddings
`intfloat/multilingual-e5-large-instruct`; NLI `cross-encoder/nli-deberta-v3-base`.
