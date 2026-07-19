WHAT WE'RE BUILDING
====================

We turn documents into a graph you can reason over —
not just search through.

Current retrieval systems (HippoRAG, GraphRAG) extract
things like "HSM — related to — outbreak." Fine for finding
which paragraph to read. But it can't answer:

  - "What's the evidence for X?"
  - "Where do sources disagree?"
  - "What's the weakest link in the argument for Y?"

Those questions need more than topic connections.
They need argument structure: what supports what,
what attacks what, and how strong each piece is.


THE GRAPH MODEL
================

The graph has three kinds of objects.


1. NODES — atomic statements pulled from source text

   Each node has a TYPE that says what kind of statement it is:

   Data nodes:       experiments, statistics, measurements
   Argument nodes:   claims, interpretations, rebuttals
   Question nodes:   hypotheses, conclusions

   Each node also carries:
   - a strength score from 0 to 1 (computed from the graph)
   - a pointer to the exact quote in the source document

   Example nodes from a drug trial document:

   [DATA]      C1: "A 2,000-person trial showed Drug X lowered BP"
   [DATA]      C2: "The trial was funded by the manufacturer"
   [DATA]      C3: "Industry-funded trials overstate benefits"
   [ARGUMENT]  C4: "The trial result is reliable"
   [ARGUMENT]  C5: "Drug X is effective"
   [DATA]      C6: "A separate 500-person trial found no effect"


2. CARDS — relationships between nodes

   A card connects one or more input nodes (premises)
   to one output node (target), with a label:
   support, attack, or outweigh.

   Example cards:

   Card 1:  C1 ——supports——> C4
            "The trial result supports reliability"

   Card 2:  C2 + C3 ——attack——> C4
            "Funding bias and industry pattern together
             undermine the trial's reliability"

   Card 3:  C4 ——supports——> C5
            "If the trial is reliable, Drug X is effective"

   Card 4:  C6 ——attacks——> C5
            "A second trial contradicts effectiveness"


3. EDGES — just wiring, no meaning of their own

   Node  → Card    (premise: this node feeds into this card)
   Card  → Node    (target: this card outputs to this node)
   Doc   → Node    (source: this node came from this document)


WHY CARDS?
===========

Look at Card 2. Neither C2 nor C3 attacks anything alone.

  "The trial was industry-funded"       — so what?
  "Industry trials overstate benefits"  — okay, in general.

But TOGETHER they attack C4. That's a joint argument.
Most interesting arguments in science, law, and policy
are joint — multiple pieces that only matter together.

A plain arrow (node → node) can't represent "together."
A card with two inputs can. That's the whole reason.


HOW WE BUILD THE GRAPH WITHOUT BLOWING UP
===========================================

The expensive step is labeling: you give an LLM two nodes
and ask "does A support, attack, or have no relation to B?"
One call per pair. So the question is: how many pairs
do we actually send to the LLM?

We set a hard budget: MAX_LLM_PAIRS = 1000 (configurable).
The pipeline's job is to pick the best 1000 pairs to check.

The funnel works in three stages:

  Stage 1 — Type-based blocking
    Data mostly targets arguments. Arguments mostly target
    hypotheses. Skip pairs that don't make structural sense
    (data-vs-data, hypothesis-vs-hypothesis).
    At 10k nodes this alone can cut 50M pairs to 2-3M.

  Stage 2 — Cheap local filter (free, runs on CPU)
    A small model (DeBERTa) scores whether two nodes
    are even talking about the same thing. Takes seconds
    over millions of pairs. Kills ~99%.

  Stage 3 — Rank and cap
    Score surviving pairs by confidence. Take the top
    MAX_LLM_PAIRS. Those go to the LLM for labeling.
    Each labeled pair with a real relationship becomes a card.

So the LLM cost is fixed regardless of corpus size.
590 nodes or 100k nodes — same budget, same cost.
What changes is how aggressively the funnel filters.


TWO OPERATIONS
===============

OPERATION 1: QUERY (most of the time)

  Documents are already in the knowledge base.
  No ingest, no new nodes. You just ask questions.

  1. Parse the question
  2. The graph finds relevant nodes using embeddings
     + argument structure (not just keyword matching)
  3. The graph does structural reasoning:
     scores, support/attack chains, load-bearing links
  4. Top-k relevant nodes' source text + graph structure
     get passed to the LLM
  5. LLM reads the source material, guided by the graph,
     and writes the answer

  Queries don't modify the graph.
  They read what's already been built.
  The 10th query about the same topic is just as cheap
  as the 1st — the cards are already there.


OPERATION 2: INGEST (when a new document arrives)

  This is occasional — adding a new paper, report,
  or dataset to the knowledge base. Two phases:

  FAST (at ingest time, seconds):
    - extract nodes from the new document
    - type them (data / argument / hypothesis)
    - embed them
    - store them in the existing graph
    The new content is now queryable immediately,
    though its connections to existing nodes are sparse.

  SLOW (background, runs later):
    - pair new nodes against the existing graph
    - label the top MAX_LLM_PAIRS candidates → create cards
    - re-score affected subgraph
    - detect new contradictions, update strength scores
    - learn patterns: if the same card shape keeps appearing
      ("industry funding attacks trial reliability"),
      promote it to a reusable rule that fires automatically
      on future documents without an LLM call

  The key property: everything is incremental.
  New documents don't re-process old ones. The slow path
  only pairs NEW nodes against the EXISTING graph.
  Cards from earlier runs persist and compound.

  So the graph gets richer over time:
    Document 1:  590 nodes, ~200 cards       (from scratch)
    Document 2:  +400 nodes, ~300 new cards  (pairs against existing)
    Document 10: +350 nodes, ~250 new cards  (but 50 are free
                 because learned rules fired automatically)

  Ingest is expensive but happens once per document.
  Queries are cheap because they read stored structure.


HOW SCORING AND QUERIES WORK
==============================

Once you have nodes and cards, a published formula
called DF-QuAD computes a strength score for every node.

No LLM involved here. It's pure arithmetic on the graph.

The idea is simple. Every node starts with a base score.
Then:
  - its supporters push it up
  - its attackers push it down
  - how much depends on how strong the supporters
    and attackers themselves are

The formula processes nodes in dependency order
(if A supports B, score A first, then use A's score
when computing B). One pass through the graph, done.


QUERY 1: "What's the evidence for X?"

  Pick a node. Follow all support cards inward.
  Each card points back to its premise nodes.
  Those are your evidence — already ranked by strength.

  "What supports Drug X being effective?"
    → Card 3 (C4 supports C5)
      → Card 1 (C1 supports C4)
        → C1: "2,000-person trial showed benefit"  [0.80]

  You get a chain with scores, not just a list of
  paragraphs that mention the same keywords.


QUERY 2: "Where do sources disagree?"

  Find any node that has BOTH support and attack cards.
  That's a contested claim. The scores tell you who's winning.

  C5 "Drug X is effective":
    Supported by: Card 3 (via trial result)      strength ~0.45
    Attacked by:  Card 4 (contradicting trial)   strength ~0.60

  The attack is currently stronger. The graph shows
  exactly what's in tension and which side has more weight.


QUERY 3: "What's the weakest link in the argument for Y?"

  Perturb each card one at a time. Remove it, re-run
  the formula, see how much the final score moves.
  The card whose removal moves the score the most
  is the load-bearing link.

  C5 "Drug X is effective" scores 0.30.

  Remove Card 2 (funding bias attack on the trial):
    → C4 rises from 0.45 to 0.70
    → C5 rises from 0.30 to 0.65
    → score moved 0.35  ← biggest swing

  Remove Card 4 (contradicting trial):
    → C5 rises from 0.30 to 0.45
    → score moved 0.15

  Card 2 is load-bearing. The funding argument is
  what's dragging the conclusion down the most.
  If you could resolve the funding concern,
  the whole picture changes.


All three queries work the same way:

  1. The GRAPH does the structural reasoning:
     which nodes are relevant, how they connect,
     what their scores are, what's load-bearing.

  2. The graph ranks and retrieves the top-k relevant
     nodes — like HippoRAG does with chunks, except
     our ranking uses argument structure and strength
     scores, not just topic similarity.

  3. Those nodes' source text (the original quotes
     from the document) gets passed to the LLM
     along with the graph's structural answer
     (the scores, the support/attack relationships).

  4. The LLM reads all of that and writes the answer.

So the LLM isn't just reading a table of scores —
it's reading the actual source material, but GUIDED
by the graph. The graph tells it what to focus on,
what order things matter in, and where the conflicts are.
The LLM brings comprehension and nuance.

Without the graph: the LLM gets a bag of chunks
and guesses what matters. (This is what HippoRAG does.)

Without the LLM: you get scores and structure
but no readable explanation.

Together: the graph does the reasoning,
the LLM does the reading and explaining,
and you can audit either layer independently.



THE PIPELINE
=============

1. EXTRACT NODES from document
   ✅ Done — 1,370 nodes (Nathan's improved-prompt extraction)
       with source quotes and byte spans

2. TYPE each node (data / argument / hypothesis)
   ✅ Done — 8 categories (evidence, estimate, hypothesis, etc.),
       mapped to the 3-layer vocabulary (data / argument / question)
       via layer_of() / map_layers.py.

3. TAG ROLES (leakage guard — not in the Slack message,
   but critical for our test case because the document
   contains its own verdict)
   ✅ Done — 1,370 tagged gpt-oss-only (evidentiary 1036 /
       conclusory 228 / procedural 106); leakage guard live
   ⚠️  boundary claims not human-adjudicated (deferred)

4. SOURCE ATTRIBUTION (who said each thing)
   ✅ Done — 1,370 attributed gpt-oss (adjudicator 225 / party 61 /
       cited_source 105 / witness 19 / unattributed 960); attribute_sources.py.
       Pays off most with multiple documents (corroboration / independence).

5. EMBED nodes
   ✅ Done — e5-large-instruct, 1,370 × 1024 (embed_nodes.py)

6. PAIRING FUNNEL
   ✅ Stage 1 (type-layer blocking) — built
   ⚠️  Stage 2 (DeBERTa NLI) — built, but slow on CPU;
       run cosine-only for now (GPU/xsmall to enable)
   ✅ Stage 3 (rank & cap at MAX_LLM_PAIRS=1000) — built
       (pairing_funnel.py)

7. LLM LABELING → CARDS
   ✅ Done — 429 cards (label_pairs.py, gpt-oss)

8. DF-QuAD SCORING
   ✅ Done — score_dfquad.py (+ ablate); fixture regression gate

9. QUERY LAYER (graph retrieval → LLM answer)
   ✅ Done — query_epistemic.py (3 query types + LLM prose);
       verified end-to-end, scorer cross-checked

10. FAST/SLOW PATH SPLIT
    ❌ Not built — everything above is still sequential


SUMMARY
========

   extract → type → role → embed → pair → filter →
   label → cards → score → query → [WE ARE HERE]

   Steps 1-3, 5-9:  done
   Step 4 (attribution) + step 10 (fast/slow split): not started

The graph is built and queryable end-to-end on the 1,370-claim
corpus: 1,370 nodes → 429 cards → DF-QuAD scores (34 contested),
leakage guard holding. It has stopped being a bag of nodes and
is now an argument graph you can query (evidence-for / contested /
weakest-link), verified against query_epistemic.py.

Remaining: source attribution (4), the fast/slow ingest split (10),
and enabling the DeBERTa NLI Stage-2 on a GPU.