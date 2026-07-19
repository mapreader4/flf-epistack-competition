# Pipeline analysis — every step at its own scope (2026-07-16)

Grounded on the first real Structure-layer output: `artifacts/epistemic/nodes.jsonl`
(590 typed nodes from `scripts/type_claims.py`, GPT-OSS-120b) plus the earlier
HippoRAG audit (WORKLOG finding -d). For each step: **state**, **quality** (with the
actual numbers where we have them), **causes** of the errors, and **options** to fix.
A whole-pipeline pseudocode is at the bottom.

Legend: ✅ built & measured · 🟡 built, defects found · ⬜ not built yet (planned).

---

## Step 0 — Chunking ✅
- **Scope:** PDF → section-aware chunks.
- **State/quality:** 195 chunks, 64-section TOC tree, ids = `chunk-<md5>`; 195/195
  join onto the KG. Known defect: footnote text spliced mid-sentence in a few chunks
  (`TASKS.md` task 2). Low-impact for now.
- **Options if it bites:** footnote-aware PDF parse (strip/relocate footnote spans
  before packing). Deferred — no evidence it's hurting downstream yet.

---

## Step 1 — Entity detection & canonicalization 🟡 (HippoRAG scope; NOT our store)
**Q1: "are the entities detected properly?"** — Partly. Detection is fine; **resolution
is broken**, and entities don't yet exist in *our* typed store at all.

- **State:** entities live only in the HippoRAG KG (1,478 entity nodes). Our
  `nodes.jsonl` is keyed on *claims*, not entities — so for the Structure layer we
  currently have **no entity layer**.
- **Quality (from the audit, WORKLOG -d):**
  - Detection recall is OK; **canonicalization is the failure**. The protagonist is
    shattered into **11 un-merged HSM nodes** (`hsm` deg 206 vs `huanan seafood
    market`, `wuhan seafood market`, raw URLs); "Bayes" → 4 nodes.
  - Synonymy at cos ≥ 0.95 errs **both** ways: false merges (`cdc`↔`wuhan cdc`) and
    misses (`hsm`↔`huanan seafood market`).
  - **Junk entities:** raw URLs, math fragments (`p o1 a` from `P(O1|A)`) admitted.
- **Causes:** (a) embedding-KNN can't equate an acronym with its expansion (no
  lexical overlap, moderate cosine); (b) a single global threshold can't fit both
  precision and recall; (c) NER has no type filter, so URLs/math tokens pass.
- **Options (cheapest → richest):**
  1. **LLM alias-clustering** instead of embedding-KNN: block candidates by shared
     token / high cosine, then ask the LLM "same real-world entity? y/n" pairwise —
     merges acronym↔expansion, rejects `cdc`↔`wuhan cdc`. (Strong prototype: count
     nodes before/after.)
  2. **Canonical-entity dictionary** built once (LLM proposes canonical names +
     aliases), then string-map every mention. Reusable across the 3 case studies.
  3. **Entity typing / junk filter** — drop URL/number-only tokens at NER.
  4. **Do we even need an entity layer in the store?** For engine B (linking) we need
     candidate retrieval, which entities *help* but embeddings over claim text can
     also provide. Decide by whether shared-entity blocking beats pure claim-embedding
     kNN for surfacing related claims. (Open.)

---

## Step 2 — Claim extraction + provenance ✅
- **Scope:** section text → atomic, decontextualized `{claim, quote, span}`.
- **Quality:** 590 claims, **1.8% drop**, provenance resolved deterministically
  (exact-then-fuzzy, never model offsets). This is the **T1 trust floor** everything
  else gates on. Solid.
- **Latent signal it already carries (confirmed in step 3):** attributions
  (`"Rootclaim argues…"`) and numbers (`"…as 1/50"`) survive into the claim text,
  which is what makes cheap typing possible.
- **Options:** add **evidence attribution** as a first-class extraction field
  (meeting-note item 2) rather than re-deriving it in typing. Minor.

---

## Step 3 — Node typing (engine A) 🟡 — **just built, defects found**
**Q2: "are the types found properly? which mislabeled, causes, fixes?"**

- **State:** `scripts/type_claims.py` → all 590 typed in one pass, $0.03, 0 validation
  errors. Distribution:

  | type | n | | type | n |
  |---|---|---|---|---|
  | background | 154 | | assumption | 31 |
  | evidence | 147 | | rebuttal | 27 |
  | claim (fallback) | 132 | | hypothesis | 23 |
  | estimate | 65 | | verdict | 11 |

- **Defect 3a — `hypothesis` is over-labeled (23, should be ~2–5).** The two real
  top-level rivals are zoonosis vs lab-leak. But the model tagged mechanistic /
  predictive sub-claims as hypotheses too: *"HSM gives uniquely good conditions for
  spread"*, *"The location of early clusters will be driven by happenstance"*. These
  are **evidence or sub-claims**, not mutually-exclusive top-level explanations.
  - *Cause:* the prompt defines `hypothesis` intensively ("a competing explanation")
    but any speculative "X is/will be the case" sentence matches that description
    locally, with no global view of what the *few* rival hypotheses are.
  - *Options:* **(i)** two-stage typing — first a single call identifies the 2–5
    top-level hypotheses for the whole document, then per-claim typing chooses
    *relative to that fixed set* ("is this one of these, evidence-for-one, or
    neither?"). **(ii)** rename the loose sense to `sub_hypothesis`/`prediction` so
    `hypothesis` stays reserved for the mutually-exclusive top set. **(iii)** demand
    a `mutually_exclusive_with` target before allowing the `hypothesis` label.

- **Defect 3b — the `number` field over-captures by ~2×.** 151 nodes (25.6%) carry a
  number, but a crude split says **only ~77 (13%) are epistemic** (probability / ratio
  / odds / Bayes factor) and **~74 are incidental** (dates `2020`, `Jan 23`; counts
  `1200 shops`, `30000 nucleotides`, `64 codons`; population `12 million`). The true
  Bayesian ledger (decision -e's "~8%") is a *subset* of the epistemic 13% — roughly
  the 65 `estimate` nodes plus evidence carrying an explicit likelihood ratio.
  - *This corrects an earlier worry:* "background with a number" is mostly **correct**
    (*"Wuhan has 12 million people"* is genuinely background that happens to cite a
    number) — the bug isn't the type, it's that `number != epistemic weight`.
  - *Options:* **(i)** a second, narrow field `epistemic_number` gated on "is this a
    probability, prior, likelihood ratio, or Bayes factor bearing on a hypothesis?"
    — separate from any incidental figure. **(ii)** a tiny regex/units pre-filter
    (`/`, `%`, `odds`, `x`, `fold`, `factor`) to route only plausible ledger numbers
    to a focused quantitative-extraction pass. This is what selects the dense core
    for the aggregation engine (step 6).

- **Defect 3c — `claim` fallback is 22% (132).** Inspection shows two clusters the
  taxonomy has no home for: **authorial/meta/procedural** statements (*"The author was
  not adequately prepared…"*, *"…performed under acute time pressure due to the debate
  rules"*) and **narrative facts** (*"Dr Zhang Jixian is recognized as the discoverer
  of covid-19"*). They're not wrong, just unresolved.
  - *Options:* add `meta`/`procedural` and treat plain narrative as `background`; or
    accept `claim` as a legitimate generic and let step 4 (relations) decide which
    ones actually carry argumentative load. Recommend the latter first (don't grow the
    taxonomy before we know a type earns an edge).

- **Defect 3d — attribution needs canonicalization.** `the judge` (40) + `The judge`
  (21) are the **same** attributor split by case; likewise noisy singletons. This is
  the *same* problem as step-1 entity resolution, one scope up. Cheap fix: normalize
  case + an alias map. Signal is otherwise good (Rootclaim 39, Saar 8, named
  researchers).

- **Confidence triage:** 509 high / 50 med / **31 low** — the low set is genuinely
  hard (meta-statements about the argument's own structure, e.g. *"Half the positive
  log-odds weight comes from three pieces of evidence"*). Good candidates for a
  human-in-the-loop pass or a second-opinion model.

- **Cross-cutting fix worth doing once:** **second-opinion / self-consistency** — type
  each claim with two prompts (or two models) and flag disagreements for review.
  Cheap given $0.03/full pass; directly raises trust for the judged artifact.

---

## Step 4 — Relation extraction (engine B) ⬜ — **not built**
**Q3: "how are relations extracted? are they correct?"** — They aren't yet. Honest
state: the store has **0 edges**. This is the highest-risk, highest-value step (the
literature puts relation *direction* at F≈0.45 — the weakest link).

- **Plan (meeting-note item 22 — keep every call small):** for each claim, retrieve a
  few candidate neighbours (shared canonical entity from step 1, and/or claim-embedding
  kNN), then ask the LLM **pairwise** on just those pairs: *same claim? / supports? /
  attacks{undermine|undercut|rebut}? / none?* Write typed, directed edges; reify to a
  node only when the edge carries a payload (a Bayes factor).
- **Anticipated failure modes to design against (from the lit map):**
  - direction inversion (A supports B logged as B supports A) → per-edge **negate/invert
    adversarial check**;
  - two opposing-stance claims look most similar under embeddings → never infer
    *relation* from similarity, only *candidacy*;
  - hallucinated links between unrelated candidates → require the edge to cite the
    span/chunk that states it, else tier T3 and low trust.
- **Eval when built:** hand-score precision on a sample; poison-test (flip a key
  claim's sign, duplicate a correlated item) and watch the graph react.

---

## Step 5 — Sub-question / discourse grouping (engine C) ⬜ — not built
- **Scope:** group claims by the sub-question they answer (*"did the outbreak start at
  the market?"*, *"was the furin site engineered?"*) → `subquestion` nodes + `answers`
  edges (meeting-note item 24).
- **Why it matters for generality:** eggs and black-holes have **no TOC** to lean on,
  so this tier must be *built from the claims themselves*, not from `sections.json`.
- **Options:** LLM proposes a small sub-question set from a sample of claims, then
  assigns each claim. **Avoid** embedding/Leiden communities — they file a claim and
  its rebuttal together (lit map warning). Validate: are the questions the ones a human
  would ask? Does every high-load claim map to one?

---

## Step 6 — Quantitative ledger + believability ⬜ — not built
- **Scope:** on the dense core only (the epistemic-number subset from 3b, ~8–13%),
  attach signed `weighs_on` edges (evidence → hypothesis, carrying log-BF), decompose
  priors into `decomposes_into` product chains, and run a **deterministic** additive
  log-odds roll-up (I.J. Good weight-of-evidence) — reproducing the judge's posterior
  as a *labeled consistency check*, not as proof of extraction fidelity.
- **Believability for the qualitative 92%:** no numbers — just "adding a rebuttal
  lowers its target" propagation (meeting-note item 8). Keep the two regimes separate.
- **Depends on:** steps 3b (which numbers), 4 (which edges), and a **T2 parse pass**
  for the ~5 aggregation numbers that never atomized into claims (the §7.6 log-odds
  columns).

---

## Step 7 — Assessment hooks ⬜ — not built
- **Scope:** once structure exists, surface what a human cares about (meeting-note
  item 9): **cruxes** (flip-one-claim sensitivity over the ledger),
  **correlated-evidence** flagged as independent (a `correlation_group` bundling
  shared-source items before summation), **rhetorical-vs-evidential** tags, and
  **missing-evidence** (sub-questions with no supporting node; unanswered Walton
  critical-questions stored as live gaps).
- **Depends on:** steps 4–6.

---

## Whole-pipeline pseudocode

```text
# ============ INGESTION (steps 0–2) ============
chunks     = chunk_pdf(doc)                       # [DONE] 195 chunks, md5 ids, section tree
entities   = openie_ner(chunks)                   # [DONE-ish] HippoRAG NER — detection ok
claims     = extract_claims(chunks)               # [DONE] 590 {claim, quote, span}, T1 provenance

# ============ ENTITY CANONICALIZATION (step 1)  [TODO] ============
canon_ents = llm_alias_cluster(entities)          # merge hsm↔huanan…, split cdc↔wuhan cdc
                                                  #   block by token/cosine, decide by LLM pairwise
claim.entities = link_to_canonical(claims, canon_ents)   # gives step-4 a blocking key

# ============ NODE TYPING (step 3)  [BUILT — engine A] ============
for batch in batches(claims, 20):                 # small calls, cached
    labels = llm_type(batch)                       # role ∈ NODE_TYPES + number + attribution + conf
nodes = [ Node(id, type, text, fingerprint(text), provenance=[T1], payload{number,attr}, meta{type_tier:T3}) ]
# open fixes: 3a two-stage hypothesis set · 3b epistemic_number gate · 3d canonicalize attribution
# quality:    second-opinion typing → flag disagreements

top_hyps   = llm_find_top_hypotheses(nodes)       # [TODO 3a] the 2–5 mutually-exclusive rivals
epi_nums    = filter(nodes, is_epistemic_number)  # [TODO 3b] ledger core ⊂ number-carrying nodes

# ============ RELATION EXTRACTION (step 4)  [TODO — engine B] ============
edges = []
for c in nodes:
    cands = retrieve_candidates(c, k=6)           # shared canon entity ∪ claim-embedding kNN
    for d in cands:
        rel = llm_pairwise(c, d)                   # same | supports | attacks{…} | none  (+cite span)
        if rel != none:
            e = Edge(type=rel, src=c, dst=d, tier= T1 if cited else T3)
            if llm_adversarial_invert(e) survives: # direction sanity: try to refute the edge
                edges.append(e)
dedupe_same_claim(nodes, edges)                   # 'same' → variant_of, keep both provenance, never merge id

# ============ DISCOURSE TIER (step 5)  [TODO — engine C] ============
subqs = llm_propose_subquestions(sample(nodes))   # NOT embedding communities
for c in nodes: edges += answers_edge(c, assign_subq(c, subqs))

# ============ QUANT LEDGER + BELIEVABILITY (step 6)  [TODO] ============
t2_nums = parse_aggregation_numbers(chunks)       # §7.6 log-odds cols that never atomized
for e in weighs_on_edges(epi_nums ∪ t2_nums, top_hyps):
    e.log_bf = extract_bayes_factor(e)
posterior = prior_logodds(top_hyps) + Σ log_bf(weighs_on)   # deterministic; = consistency check
propagate_defeat(edges)                           # qualitative 92%: rebuttal lowers its target

# ============ ASSESSMENT (step 7)  [TODO] ============
cruxes        = sensitivity(posterior, per-edge flip)
corr_flags    = detect_correlation_groups(edges)  # shared-source → discount before Σ
gaps          = subqs_with_no_support(edges) ∪ unanswered_critical_questions(nodes)

# ============ STORE (container)  [BUILT] ============
write_jsonl(nodes, edges)                         # typed, directed, tiered, provenance-anchored
validate_store(nodes, edges)                      # dangling edges / dup ids / bad types
```

---

## Priority read (what to fix first, by signal-per-effort)
1. **3b epistemic-number gate** + **3d attribution canonicalization** — tiny, and they
   sharpen the core selection and a duplication bug immediately.
2. **3a two-stage hypothesis set** — fixes the most visible typing error and is a
   prerequisite for step 4/6 (edges need a fixed hypothesis set to point at).
3. **Step 4 engine B on a small batch** — this is where we *learn* how much relational
   structure survives; do it on the §7 core first, hand-score, then decide breadth.
4. **Step 1 alias-clustering** — needed as a blocking key for step 4 at scale; can lag
   the §7 pilot.
```
