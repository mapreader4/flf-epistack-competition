# End-to-end verification — argument graph on the 1370-claim extraction

*Run 2026-07-18. Source corpus: Nathan's improved-prompt extraction
(`artifacts/claims_improved_prompt/claims.json`, 1370 claims). Raw outputs live in
`artifacts/epistemic/query_examples/`.*

## Pipeline run
`tag_roles` (gpt-oss-only) → `type_claims` → `map_layers` → `embed_nodes` →
`pairing_funnel --no-nli` → `label_pairs --emit-events` → `score_dfquad`.

| | value |
|---|---|
| Nodes (typed) | **1370** |
| Roles (evidentiary / conclusory / procedural) | 1036 / 228 / 106 |
| Cards | **429** (328 support RA, 101 attack CA) |
| Contested nodes (both support & attack) | **34** |
| LLM cost (typing + labeling, gpt-oss) | ~$0.13 (+ role-tagging; total < $0.50) |

## Verification checks (all pass) — `query_examples/_verification.json`
| check | result |
|---|---|
| `validate_store` (dangling premises/targets, dup ids) | **0 errors** |
| Leakage guard: cards touching a `conclusory` node | **0** (must be 0) |
| `event_store.replay()` cards vs `cards.jsonl` | **429 = 429** |
| `score_dfquad --selftest` (fixture regression) | **PASS** |
| **Independent scorer cross-check** (our `score_dfquad` vs Arpita's `query_epistemic.score_graph`, all 1370 nodes) | **max Δ 5.0e-7**, mean Δ 1.7e-8 → agree |

The scorer cross-check is the strongest signal: two independently-written DF-QuAD
implementations converge to floating-point noise on the real graph.

## Query examples (via `scripts/query_epistemic.py` on the real artifacts)
Full JSON in `artifacts/epistemic/query_examples/`. Run with
`--data-dir artifacts/epistemic --prefix ""`.

**Q1 — evidence-for `n-00502`** ("The combination of these factors at HSM increases the
chance of an outbreak", strength **0.995**) → `q1_evidence-for_n-00502.json`
```
<- n-00545 str=0.762 §4.5.1  "To get 2^10 HSM cases within 5 doublings we need each resident to infect ~4 others"
<- n-00547 str=0.713 §4.5.1  "The total infectivity rate at HSM is 4 times that elsewhere"
<- n-00521 str=0.605 §4.5.1  "The visitor infects a resident at HSM"
```
Every premise resolves to a T1 span (claim_id + verbatim quote + byte span).

**Q2 — contested** → 34 nodes, ranked by |vs−va| → `q2_contested.json`
```
n-00168 vs=0.86 va=0.03 support  "Rootclaim's claim is that the SARS-CoV-2 outbreak was due to..."
n-00302 vs=0.81 va=0.17 support  "The goal of Rootclaim is to get a Bayes factor..."
n-00425 vs=0.72 va=0.17 support  "The number of vendors at HSM is rounded to 1200..."
```

**Q3 — weakest-link `n-00502`** (card-ablation) → `q3_weakest-link_n-00502.json`
```
card-00345 Δ=0.0100 RA   card-00262 Δ=0.0084 RA   card-00416 Δ=0.0055 RA
```

**LLM prose layer** (forced `--model openai/gpt-oss-120b`) →
`q3_weakest-link_n-00502_gptoss_answer.txt` — gpt-oss produced a coherent,
provenance-cited answer naming the load-bearing card, its premise text, and the exact Δ.

## Known integration note
`query_epistemic.py` currently defaults to `meta-llama/Llama-3.3-70B-Instruct-Turbo`
for the prose layer (`DEFAULT_MODEL`), which is against the gpt-oss-everything policy;
pass `--model openai/gpt-oss-120b` (as done here) until the default is changed. Its
`generate_prose` also omits `max_tokens`, which can truncate gpt-oss on larger results.
