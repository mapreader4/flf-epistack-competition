# Handoff — Query & Eval lane (pipeline step 9)

Hi — this is your slice of the epistemic-graph pipeline. It's **self-contained and
unblocked**: everything you consume already exists as fixtures, so you can build and
test end-to-end today without waiting on the graph-structure lane. Read `design.md`
(lines 191-277) for the vision; this doc is the concrete task + materials.

## The one-paragraph picture
We turn the judge's COVID-origins ruling into a **typed argument graph** — nodes
(atomic claims, typed), **cards** (reified support/attack/outweigh relations), and a
DF-QuAD **strength** in [0,1] on every node. Amir owns producing that graph + the
scores (lanes A+B). **You own the query layer**: given the graph + scores, answer the
three questions design.md promises, and evaluate the answers. The graph does the
structural reasoning; the LLM (gpt-oss) reads the top nodes' source quotes and writes
prose.

## What's ready for you right now
- **Fixtures** (`artifacts/epistemic/fixtures/`) — a 6-node "drug-trial" graph with
  cards, scores, and ablation output. Same shapes as the real 590-node artifacts, so
  code you write against these swaps to the real files with **zero changes**. See
  `fixtures/README.md`.
- **The scoring API** you call — `scripts/score_dfquad.py` — is built and passing its
  regression gate. You import `score_graph` and `ablate` from it.
- **Contracts** — `docs/contracts-steps-5-9.md` pins every file shape. Code against
  these; don't hardcode assumptions.
- **The real nodes + roles** already exist (`nodes.jsonl`, `claims_tagged.jsonl`).
  Only the real `cards.jsonl` / `scores.jsonl` are pending from Amir — the fixtures
  stand in until then.

## Your task, in two halves (split across the two of you if you like)

### C1 — Query traversal + LLM answers  → `scripts/query_epistemic.py` (new)
Implement the three canonical queries. Each: the **graph** selects/ranks nodes, then
those nodes' T1 source quotes + the structural facts go to gpt-oss, which writes the
answer. Load the graph with `read_nodes` / `read_cards` and scores from
`scores.jsonl`.

1. **"What's the evidence for X?"** — given a target node, walk **inward over RA
   (support) cards** (cards whose `target == X`, follow their `premises`), recursively.
   Return the premise nodes as an evidence chain **ranked by `strength`** (from
   contract 5). Worked fixture example: evidence-for `C5` → `card-3` (C4 supports C5)
   → `card-1` (C1 supports C4) → `C1` the 2,000-person trial.
2. **"Where do sources disagree?"** — nodes with **both** `in_supporters` and
   `in_attackers` non-empty (contract 5 gives you these directly). Report `vs` vs `va`
   and which side wins. In the fixture, `C4` and `C5` are contested. The structure is
   the answer; the LLM only writes prose.
3. **"What's the weakest link in the argument for Y?"** — call
   `ablate(nodes, cards, Y)`; it returns `{card_id: |Δ|}` sorted, largest first =
   load-bearing. Fixture: `ablate("C5")` → `card-3` (0.25) is most load-bearing.

**LLM call:** reuse the client pattern in `scripts/type_claims.py`
(`build_client()`, `temperature=0`). **Model = `openai/gpt-oss-120b`** (project
default — pass `max_tokens=4000`, it's a reasoning model that truncates otherwise).
Prompt = the structural answer (scores, support/attack sets, ablation ranking) + the
top-k nodes' `provenance[0].quote`. Log each call as an `llm_call` event
(`event_store.append_events`) with the verbatim output.

### C2 — Eval harness  → `answer_key.json` + a scoring script
Build the golden set from the **`conclusory`** claims only (the judge's held-out
verdict — contract 2). The judge's reasoning: favored hypothesis = **Zoonosis**; crux
influence ordering **HSM ≫ genetic/DEFUSE > secret** (|log-odds| ≈ 9.2 / 3 / 2.3).
Score three things:
- (a) Query-1 for the Zoonosis hypothesis surfaces the HSM/market evidence on top;
- (b) Query-3 weakest-link ordering vs the judge's |log-odds| ordering;
- (c) Query-2 flags the genuinely contested hypotheses.

The key is **eval-only** — never read it from any query/scoring path. The leakage
guard (funnel exclusion + the scorer's `conclusory` assert) keeps it out of the graph.

## Quickstart (works today, on fixtures)
```python
import sys; sys.path.insert(0, "scripts")
from epistemic_store import read_nodes, read_cards
from score_dfquad import score_graph, ablate

FX = "artifacts/epistemic/fixtures"
nodes = read_nodes(f"{FX}/sample_nodes.jsonl")
cards = read_cards(f"{FX}/sample_cards.jsonl")
scores = score_graph(nodes, cards, base=0.5)          # {node_id: ScoreRecord}

print(scores["C5"].strength)                          # 0.675
print(scores["C5"].in_supporters, scores["C5"].in_attackers)  # contested?
print(ablate(nodes, cards, "C5"))                     # Query 3 ranking
```
When Amir ships the real files, point the paths at
`artifacts/epistemic/cards.jsonl` and `scores.jsonl` (or just read the pre-written
`scores.jsonl` instead of recomputing). Nothing else changes.

## Environment
- `.venv/bin/python` (Python 3.11 — `python` is not on PATH).
- Secrets in `.env`: `TOGETHER_API_KEY`. All LLM/embedding calls go to Together's
  OpenAI-compatible API.
- Run the scorer's self-test to confirm your setup: `.venv/bin/python
  scripts/score_dfquad.py --selftest` → should print `SELFTEST: PASS`.

## Deliverables
- `scripts/query_epistemic.py` — the 3 queries, each returning a structured result +
  an LLM prose answer, every cited node traceable to a T1 span.
- `artifacts/epistemic/answer_key.json` + an eval script printing the three metrics.
- A short note on where graph-guided retrieval beats HippoRAG's chunk ranking (this
  is the competition's whole point — see `design.md` lines 245-277).

## Coordinate with Amir on
- The real `cards.jsonl` schema is frozen (contract 4) — if you need a field that
  isn't there, ping before assuming.
- Query 1's "target node" selection from a natural-language question (mapping a query
  to an entry hypothesis node) is a shared design question — worth a 10-min sync.

Questions → Amir (graph-structure lane). Contracts are in
`docs/contracts-steps-5-9.md`; the vision is in `design.md`.
