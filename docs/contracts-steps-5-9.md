# File-format contracts — pipeline steps 5-9

Frozen interfaces between the three lanes (see `design.md`). Each lane codes against
these shapes (and the fixtures that instantiate them) so nobody blocks anyone. **Do
not change a record shape without pinging the other lanes** — a consumer somewhere is
parsing it.

Join key across the two label systems: a node links to its role/claim via
`node.provenance[0].claim_id`. (Verified: 0 join misses across all 590 nodes.)

| # | Artifact | Producer | Consumers | Status |
|---|----------|----------|-----------|--------|
| 1 | `artifacts/epistemic/nodes.jsonl` | (exists) | all | ✅ 590 nodes |
| 2 | `artifacts/epistemic/claims_tagged.jsonl` | (exists) | funnel, scorer, eval | ✅ role field |
| 3 | `artifacts/epistemic/embeddings.npy` + `_index.json` | A (embed) | A (funnel) | ⏳ |
| 4 | `artifacts/epistemic/cards.jsonl` | **A (Amir)** | **B, C** | ⏳ |
| 5 | `artifacts/epistemic/scores.jsonl` | **B (Amir)** | **C** | ✅ engine + fixture |
| 6 | `artifacts/epistemic/answer_key.json` | **C** | C (eval only) | ⏳ |
| 7 | `artifacts/epistemic/events.jsonl` | all (append) | replay/audit | ✅ log exists |

Fixtures instantiating contracts 1/4/5 live in `artifacts/epistemic/fixtures/`.

---

## Contract 1 — nodes.jsonl  (exists; `epistemic_store.Node`)
```json
{"node_id": "n-00000", "type": "claim",
 "canonical_text": "...", "fingerprint": "fp-...",
 "provenance": [{"claim_id": "claim-...", "chunk_id": "chunk-...",
                 "section_number": "1", "quote": "...", "span": [239, 564], "tier": "T1"}],
 "payload": {}, "tier": "T1",
 "meta": {"type_tier": "T3", "type_source": "openai/gpt-oss-120b",
          "type_confidence": "high", "is_appendix": false, "is_excluded": false}}
```
`type` ∈ {hypothesis, evidence, assumption, estimate, rebuttal, verdict, background,
claim, subquestion, correlation_group}. `payload` may carry `number`, `attribution`.

## Contract 2 — claims_tagged.jsonl  (exists; leakage guard)
Keyed by `claim_id`; the operative field is top-level `role`:
```json
{"claim_id": "claim-...", "claim_text": "...", "role": "evidentiary", ...}
```
`role` ∈ {**evidentiary**, **conclusory**, **procedural**}. **`conclusory` = the
judge's own priors / Bayes factors / verdict = the answer key.** Never a scoring
input; reserved for eval (contract 6). Distribution: evidentiary 405, conclusory 141,
procedural 44.

## Contract 3 — embeddings.npy + embeddings_index.json  (A internal)
`.npy` = float32 array `[N × 1024]`; row *i* is the embedding of `index.json["node_ids"][i]`.
```json
// embeddings_index.json
{"model": "intfloat/multilingual-e5-large-instruct", "dim": 1024,
 "node_ids": ["n-00000", "n-00001", ...]}
```

## Contract 4 — cards.jsonl  (A → B, C; `epistemic_store.Card`)
A **reified relation**: premise node(s) → one target node, one label, one weight.
```json
{"card_id": "card-00000", "kind": "CA", "weight": 0.8,
 "premises": ["n-00042", "n-00311"], "target": "n-00107",
 "active": true, "tier": "T3",
 "provenance": {"labeler_model": "openai/gpt-oss-120b", "relation_label": "attack",
                "raw_confidence": 0.8, "llm_call_event": "ev-001234",
                "prompt_version": "label-v1"}}
```
`kind` ∈ {**RA** support, **CA** attack, **PA** outweighs}. `premises` is a list
(joint premises → one card). Load with `epistemic_store.read_cards(path)`; validate
with `validate_store(nodes, [], cards)` (checks kind, weight∈[0,1], premise/target
resolve, no dup ids).

## Contract 5 — scores.jsonl  (B → C; `score_dfquad.ScoreRecord`)
One record per node, DF-QuAD strength + the pieces that produced it:
```json
{"node_id": "n-00107", "strength": 0.31, "base": 0.5,
 "va": 0.55, "vs": 0.34,
 "in_supporters": ["card-00003"], "in_attackers": ["card-00001", "card-00012"],
 "scc_id": 4, "iterations": 1}
```
`strength`∈[0,1]. A node with both `in_supporters` and `in_attackers` non-empty is
**contested** (Query 2). `va`/`vs` are combined attacker/supporter pressure.

## Contract 6 — answer_key.json  (C, eval only — NEVER a pipeline input)
Derived from `conclusory` claims: the judge's favored hypothesis + the crux
influence ordering, used only to score query outputs. Shape is C's to define; keep it
in `artifacts/epistemic/` and never read it from any scoring/labeling path.

## Contract 7 — events.jsonl  (append-only; `event_store`)
Every mutation is also an event: `card_added` / `card_deactivated` / `score_computed`
/ `verdict_emitted` / `llm_call`. `event_store.fold(read_events(path))` re-derives the
cards + scores snapshot. Append with `append_events`; never rewrite prior lines.

---

## The public scoring API (contract between B and C)
`scripts/score_dfquad.py` exposes:
```python
score_graph(nodes, cards, roles=None, base=0.5, damping=0.5, max_iter=100)
    -> {node_id: ScoreRecord}
ablate(nodes, cards, target, **kw) -> {card_id: |Δ strength of target|}   # Query 3
roles_from_tagged(nodes, tagged_path) -> {node_id: role}                  # for the guard
```
`ablate` returns cards sorted by descending |Δ|; the first is the load-bearing link.
