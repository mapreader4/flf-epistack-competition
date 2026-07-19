# Status and plan


## Where we are

```
extract → type → role → embed → pair → filter → label → cards → score → query
  ✅       ✅     ✅      ❌      ❌      ❌       ❌      ❌      ✅      ❌
```

Done:
  - 590 nodes extracted (artifacts/claims/claims.json → artifacts/epistemic/nodes.jsonl)
  - Each node typed into one of 8 categories by gpt-oss (scripts/type_claims.py)
  - Each node tagged with a role — evidentiary/conclusory/procedural — to prevent
    the judge's own conclusions from leaking into scoring (scripts/tag_roles.py)
  - 101 disagreements between two models adjudicated (artifacts/epistemic/role_adjudication.json)
  - Card data structure added to the store (scripts/epistemic_store.py → Card class)
  - Scoring formula implemented and tested (scripts/score_dfquad.py → score_graph, ablate)
  - Test data for the drug-trial example (artifacts/epistemic/fixtures/)
  - Query layer task written up and handed off (docs/colleague-task.md)

Not done:
  - No relationships between nodes exist yet (0 cards)
  - No real scores on the 590 nodes (scorer exists but has nothing to score)
  - No query layer built yet (handed off to colleague)


## What to build next

### A. Type-layer mapping

  The 8 node categories need to be grouped into 3 layers,
  because the pairing step (C below) uses layers to decide
  which pairs are worth checking.

  Mapping:
    data layer:      evidence, estimate, background
    argument layer:  claim, rebuttal, assumption
    question layer:  hypothesis, verdict

  What to do: add a "layer" field to each record in
  artifacts/epistemic/nodes.jsonl, derived from the existing
  "type" field. One script, no LLM call, runs in seconds.

  The 8 types stay — the layer is an extra field on top.


### B. Embed nodes

  Compute a numeric fingerprint (embedding) for each node's text
  so we can later measure how similar two nodes are.

  What to do: run the embedding model (intfloat/multilingual-e5-large-instruct,
  already verified in RECON.md §5) over the "canonical_text" field
  of all 590 nodes.

  Output:
    artifacts/epistemic/embeddings.npy      — a 590 × 1024 number array
    artifacts/epistemic/embeddings_index.json — maps row number to node_id

  No LLM call. Uses the Together embedding API. Cheap and fast.


### C. Pairing funnel

  This is the main engineering step. We need to decide which
  pairs of nodes to send to the LLM for relationship labeling.
  590 nodes = 174,000 possible pairs. We can't label all of them.

  The funnel has three stages:

  Stage 1 — Layer-based blocking (from step A above):
    Only check pairs that make structural sense.
    data → argument pairs get ~60% of the budget.
    argument → question pairs get ~25%.
    data → data (contradicting measurements) gets ~10%.
    argument → argument (rebuttals) gets ~5%.
    Skip data→question and question→question entirely.

  Stage 2 — DeBERTa filter (free, runs locally on CPU):
    For each surviving pair, a small model
    (cross-encoder/nli-deberta-v3-base, already verified in RECON.md §5)
    scores whether the two nodes are even about the same topic.
    Most pairs aren't. This kills ~99% of what's left.

  Stage 3 — Rank and cap:
    Sort survivors by the DeBERTa confidence score.
    Take the top MAX_LLM_PAIRS = 1000 (configurable constant).
    These are the only pairs the LLM will see.

  Output: artifacts/epistemic/candidate_pairs.json
    A list of at most 1000 (node_id_A, node_id_B, deberta_score) triples.


### D. Label pairs → create cards

  For each candidate pair, ask gpt-oss:
    "Does node A support, attack, or have no relation to node B?"

  Model: openai/gpt-oss-120b via Together API.
  Settings: max_tokens=4000, temperature=0.

  Each pair where the LLM finds a relationship becomes a Card
  (stored using epistemic_store.Card, written with write_cards).
  Each LLM call gets logged as an event in artifacts/epistemic/events.jsonl.

  Output: artifacts/epistemic/cards.jsonl
  Cost: at most 1000 calls × ~$0.003 each ≈ $3 total.

  Before writing all cards: manually inspect a sample of 20-30
  labels to check quality. If the LLM is mislabeling support
  as attack (or vice versa), fix the prompt before burning the
  full budget.


### E. Score the real graph

  Run the already-built scorer (scripts/score_dfquad.py) on
  the real nodes + cards from step D.

  Output: artifacts/epistemic/scores.jsonl
  One record per node with its computed strength.

  This takes seconds and uses no LLM — it's pure arithmetic.

  At this point: tell colleague to repoint her paths from
  artifacts/epistemic/fixtures/ to artifacts/epistemic/.
  Her code works on real data with no changes.


## DF-QuAD scoring note

The design doc (design.md lines 227-237) uses example numbers
(C5 = 0.30, removing Card 4 gives 0.45) that are not possible
under the actual formula with uniform base scores.

The formula says: a node with only supporters can never score
below its base (default 0.5). But the design doc shows a
support-only C5 at 0.45, which contradicts the formula.

The real outputs from the scorer (artifacts/epistemic/fixtures/sample_scores.jsonl):
  C5 = 0.675 (not 0.30)
  Remove funding attack (Card 2): score moves by 0.125 (not 0.35)
  Remove direct attack (Card 4): score moves by 0.075 (not 0.15)

The ordering is preserved: the funding attack still matters more
than the direct attack. The insight is correct. The digits aren't.

The scorer is correct. The design doc's numbers were illustrative,
not traced from the formula. See artifacts/epistemic/fixtures/README.md
for the full explanation.


## TODOs — things we skipped, in priority order


### TODO 1: Source attribution
  What it is:
    For each of the 590 nodes, record WHO said it.
    Is it something a debater argued? Something a cited study found?
    Something the judge concluded? A witness statement?

  Where it would go:
    A new field "source_attribution" on each record in
    artifacts/epistemic/nodes.jsonl (or claims_tagged.jsonl),
    with values like {speaker: "Peter Miller", role: "debater"}
    or {speaker: "Andersen et al. 2020", role: "cited_study"}.

  Why it matters:
    When you have multiple documents, you need to know whether
    two claims come from the same person or the same study.
    If they do, they shouldn't count as independent evidence.
    Without this field, ten articles all citing the same press
    release look like ten separate pieces of evidence. They're not.

  Why we skipped it:
    Right now we have one document. There's no second source to
    compare against, so the independence check can't fire.

  When to do it:
    Before ingesting a second document. It requires one LLM pass
    over all nodes (same pattern as scripts/tag_roles.py — batch
    of 10 nodes, ask "who is speaking here?", log the raw output).

  Code to reuse:
    scripts/tag_roles.py (same batch-and-tag pattern)
    scripts/event_store.py (log each tag as an event)


### TODO 2: Settle the 37 boundary claims
  What it is:
    During role tagging, two models (gpt-oss and Llama) disagreed
    on 101 of 590 nodes. We ran independent adjudicators on all 101.
    Result: gpt-oss right 61 times, Llama right 20 times,
    17 genuinely ambiguous (adjudicators themselves split),
    3 got malformed responses.

    The 20 where Llama was right: gpt-oss wrongly tagged real
    evidence as "conclusory" (judge's conclusions). Those 20 nodes
    are currently held out of scoring even though they're real
    evidence. That means the graph is missing 20 pieces of evidence.

    The 17 genuinely ambiguous ones need a human decision.

  Where the data is:
    artifacts/epistemic/role_adjudication.json — all 101 disputes
    with both adjudicators' verdicts and the exact deciding phrases.

  What to do:
    Go through the 20 Llama-was-right cases in role_adjudication.json.
    Flip their role from "conclusory" to "evidentiary" in
    claims_tagged.jsonl (or add a new event to events.jsonl
    changing the tag).

    For the 17 ambiguous ones: read the adjudicators' reasoning
    in role_adjudication.json, decide each one, log the decision.

  Why it matters:
    These 37 nodes affect what evidence enters the graph.
    A wrongly held-out piece of evidence is a missing card
    that could change scores.

  When to do it:
    Before freezing the golden set (the test queries + expected
    answers used to evaluate the system).


### TODO 3: Family deduplication
  What it is:
    When two nodes say essentially the same thing
    ("Drug X lowers blood pressure" and "Drug X reduces BP"),
    group them into one "family" with one canonical version.

    Corroboration (independent confirmation) should count
    families, not raw nodes. Otherwise near-duplicates inflate
    the support score.

  Where it would go:
    A "family_id" field on each node in nodes.jsonl, plus a
    families.json file mapping family_id to its canonical text
    and list of member node_ids.

  Why we skipped it:
    With one document, most claims appear once. Near-duplicates
    are rare (the extractor already atomized them).

  When to do it:
    Before ingesting documents that overlap in content
    (e.g. two papers about the same study). Use the embeddings
    from step B — nodes with cosine similarity > 0.95 are
    candidates for merging. Human review on borderline cases
    (the known failure: merging "lowers BP" with "lowers BP
    in adults over 65" — those are different claims).


### TODO 4: Fast/slow ingest split
  What it is:
    Right now the pipeline runs sequentially: extract, type,
    embed, pair, filter, label, score. That takes minutes to hours.

    For a live system where users are querying while a new
    document is being ingested, split into two paths:
      Fast (seconds): extract + type + embed + store.
        New content is immediately searchable.
      Slow (background): pair + filter + label + score.
        Relationships get built after, without blocking queries.

  Why we skipped it:
    No one is querying the system while we build it.
    The sequential pipeline is simpler and produces the same result.

  When to do it:
    Before any deployment where users send queries while
    new documents are being added.

  What to build:
    A job queue (even a simple file-based one) that tracks
    which nodes have been through the slow path and which haven't.
    The pairing funnel (step C above) already has the right
    property: it only pairs NEW nodes against the EXISTING graph,
    not all nodes against all nodes.


### TODO 5: Design 1 — Lifecycle tracking
  What it is:
    Give every node a status that changes as the graph evolves:
      extracted → corroborated → contested → settled

    "Corroborated" = multiple independent sources support it.
    "Contested" = it has strong attacks.
    "Settled" = high score, independent support, attacks defeated,
                no open questions pending.

    Also: if a source gets discredited, automatically flag
    everything that depended on it as "stale" so it gets re-checked.

  Where it would go:
    A "status" field on each node in nodes.jsonl.
    A "depends_on" edge from derived conclusions to their supporting
    evidence, so the system knows what to re-check when something changes.

  Why we skipped it:
    Needs a working scored graph first. Status transitions are
    triggered by scores and cards, which don't exist yet.

  When to do it:
    After cards.jsonl and scores.jsonl exist on real data.

  Reference:
    final-five-designs.md §2 (Design 1), lines 103-172.


### TODO 6: Design 2 — Calibration
  What it is:
    Every label the LLM produces (node types, card labels) has
    some error rate. Right now we don't measure it — we just
    trust the output.

    Calibration means: audit a sample of LLM outputs, measure
    how often each type of output is correct, and use that
    measured accuracy to adjust scores.

    Example: if the LLM labels "support" cards correctly 90%
    of the time but "attack" cards only 70% of the time, attack
    cards should carry less weight in scoring.

  Where it would go:
    A "calibration_class" field on each node and card, pointing
    to a record that stores the measured accuracy for that
    category of output.
    A "tau" (base score) field on each node, set from the
    calibration data instead of the current uniform 0.5.

  Why we skipped it:
    Needs enough cards to form meaningful categories.
    Can't measure accuracy of card labels if no cards exist.

  When to do it:
    After the first batch of cards exists. Audit 30-50 cards
    per category, measure accuracy, feed that into the scorer's
    base scores.

  Reference:
    final-five-designs.md §3 (Design 2), lines 176-235.


### TODO 7: Design 3 — Reusable rules
  What it is:
    When the same card pattern appears repeatedly
    ("X was industry-funded" + "industry trials overstate" → attack X),
    save it as a rule that fires automatically on new documents
    without needing an LLM call.

    Rules start in "shadow mode" (they log what they would do
    but don't actually create cards). Only activate after their
    accuracy is measured on enough examples.

  Why we skipped it:
    Needs at least two documents to detect repeating patterns.
    With one document there's nothing to generalize from.

  When to do it:
    After ingesting 2-3 documents that share topic areas.

  Reference:
    final-five-designs.md §4 (Design 3), lines 238-300.


### TODO 8: Design 4 — Investigation engine
  What it is:
    A formal way to compare competing hypotheses. Given a question:
      1. Generate competing hypotheses (+ a catch-all "other")
      2. Find all evidence relevant to each hypothesis
      3. Score how much each hypothesis is supported vs. attacked
      4. Rank by which hypothesis survives the most attack
      5. Red-team the winner: try to find the strongest attack
         against the leading hypothesis using only evidence in the graph
      6. Emit a verdict with a full dependency trail

  Why we skipped it:
    Needs the scored graph (cards + scores) to exist first.
    The investigation engine consumes the graph; it doesn't build it.

  When to do it:
    After the query layer (colleague's work) is functional and
    cards.jsonl + scores.jsonl exist on real data.

  Reference:
    final-five-designs.md §5 (Design 4), lines 303-369.


### TODO 9: Design 5 — Reality loop
  What it is:
    The system looks at its own graph and asks: "What single
    checkable fact, if verified, would most change my conclusions?"

    It ranks these by value: how much would the verdict move
    if this fact turned out to be true vs. false?

    When someone (or an automated agent) checks the fact and
    reports the result, the graph updates — and the accuracy
    record of the source/model that originally produced that
    claim also updates.

    Over time, the system learns which sources and which models
    to trust more.

  Why we skipped it:
    Needs the investigation engine (TODO 8) to produce verdicts
    worth checking.

  When to do it:
    After TODO 8 produces at least one verdict.

  Reference:
    final-five-designs.md §6 (Design 5), lines 371-430.


### TODO 10: HippoRAG comparison
  What it is:
    Run the same set of test queries through two systems:
      (a) Our argument graph (graph-guided retrieval + scoring)
      (b) HippoRAG's knowledge graph (topic-based retrieval)
    Compare which system surfaces better evidence, finds real
    disagreements, and identifies load-bearing links.

  What already exists:
    artifacts/hipporag2/ — HippoRAG's graph is already built
    (1,673 nodes, 2,169 triples).
    scripts/query_graph.py — existing HippoRAG query script.
    hipporag2-algorithm.md — detailed documentation of how
    HippoRAG's retrieval works.

  Why we skipped it:
    Needs both the golden set (test queries + expected answers)
    and the query layer to exist first.

  When to do it:
    After the query layer (colleague's work) can answer queries
    on real data.


## Colleague's idea: first-order logic structure

Her observation:
  The graph looks like a first-order logic proof graph where
  cards act as inference rules linking premises to conclusions.

Her proposal:
  Break the "question" layer into finer types:
    hypotheses, research questions, claims, conclusions.
  Use data and argument nodes as reasoning links (cards)
  between premises and conclusions, instead of as nodes
  with their own scores.

Why it's interesting:
  You could read the graph as a proof: "given this evidence
  and this reasoning, we conclude X." That's more interpretable
  than "this node has score 0.675."

Why to try it later, not now:
  It changes what's a node vs. what's a card. Right now all
  statements are nodes (with scores) and all relationships
  are cards. Her proposal would turn some statements into
  cards — and cards don't get independent scores in DF-QuAD.
  That means rethinking how scoring works.

Recommended path:
  Build the current architecture first (it's specced, the scorer
  works, the colleague is building against it). Then take the
  same 590 nodes, re-partition them under her structure, and
  compare side-by-side. If hers is clearly better, migrate.
  If it's a tradeoff, keep both as different views over the
  same underlying data.