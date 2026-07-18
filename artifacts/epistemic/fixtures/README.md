# Fixtures — the drug-trial worked example (design.md lines 38-68)

A tiny, hand-authored argument graph used to (a) let the scoring + query lanes start
before the real 590-node graph exists, and (b) act as the **DF-QuAD regression gate**
(`scripts/score_dfquad.py --selftest`). These are the same shapes as the real
artifacts, so code written against them swaps to the real files with zero changes.

## Files
| File | Contract | Contents |
|---|---|---|
| `sample_nodes.jsonl` | 1 (nodes) | 6 nodes C1-C6 |
| `sample_cards.jsonl` | 4 (cards) | 4 cards incl. the joint C2+C3→C4 attack |
| `sample_scores.jsonl` | 5 (scores) | DF-QuAD output for the 6 nodes |
| `sample_ablation.json` | — | `ablate()` output for C4 and C5 (Query 3) |

## The graph
```
C1 evidence  "2,000-person trial showed Drug X lowered BP"
C2 evidence  "trial was funded by the manufacturer"
C3 background "industry-funded trials overstate benefits"
C4 claim     "the trial result is reliable"
C5 hypothesis "Drug X is effective"        ← the conclusion Y
C6 evidence  "separate 500-person trial found no effect"

card-1  RA  w=1.0  C1        → C4   (support)
card-2  CA  w=1.0  C2 + C3   → C4   (JOINT attack — the whole point of cards)
card-3  RA  w=1.0  C4        → C5   (support)
card-4  CA  w=0.3  C6        → C5   (attack)
```

## Expected numbers (base v0 = 0.5, exact DF-QuAD)
Leaves C1,C2,C3,C6 = 0.5. `C4 = 0.5` (support 0.5 vs joint attack 0.5 cancel).
`C5 = 0.675`. `ablate(C5)` = `{card-3: 0.25, card-1: 0.125, card-2: 0.125, card-4: 0.075}`.

The selftest asserts these plus the **ordinal claim design.md makes**: removing
`card-2` (the funding attack, two hops up the chain) moves C5 more than removing
`card-4` (the direct attack) — `0.125 > 0.075`. `card-3`, the support link, is the
single most load-bearing (0.25): pull the support and the conclusion falls furthest.
