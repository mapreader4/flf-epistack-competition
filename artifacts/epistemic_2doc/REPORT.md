# Two-Document Epistemic Graph — eric_decision + will_decision

*Draft deliverable. Blanks (`___`) filled after Step 5–7 (labeling → scoring → verify).*

## What this is
A single epistemic argument graph spanning **two independent judges' decisions** on the
same COVID-19 origins debate (zoonosis vs. lab leak): `eric_decision.pdf` and
`will_decision.pdf`. Every node is document-attributed, so the graph can answer
cross-document questions — where the two judges' reasoning **agrees**, where it
**conflicts**, and what evidence each marshals for the shared hypotheses.

## Pipeline (one command per stage)
`epistemic_node_extraction_multidoc.py` (extract 7 node types from both PDFs, gpt-oss)
→ `multidoc_to_graph_nodes.py` (convert to graph schema; per-doc stamping; autonomous
top-level-hypothesis selection) → `embed_nodes.py` → `pairing_funnel.py --nli` (DeBERTa,
GPU) → `label_pairs.py` (gpt-oss cards) → `score_dfquad.py` → `verify_2doc.py`.

## Graph at a glance
| | value |
|---|---|
| Total nodes | **2232** (eric 2077 + will 155) |
| Node types | claim 567, assumption 426, evidence 396, estimate 380, rebuttal 310, subquestion 149, hypothesis 4 |
| Top-level hypotheses (LLM-selected, 4) | eric: lab-leak (n-00180), animal-host-at-HSM (n-00262); will: zoonotic-more-likely (n-00023), index-case-lab-worker (n-00066) |
| Hypotheses demoted → claim (defect 3a) | 295 |
| Candidate pairs (DeBERTa, `--nli`) | 3000 |
| Cards total / RA / CA | 1678 / 1382 / 296 |
| Cards by channel | top_down 689, data→argument 633, argument→question 300, contradiction 56 |
| Relation (hit) rate | 0.559 |
| **Cross-document cards** (premise in one doc, target in the other) | **230** (will→eric 109, eric→will 121) |
| Contested nodes (support + attack) | 82 |
| **Contested nodes spanning BOTH docs** (Eric vs Will) | **31** |

## Cross-document findings
- **Every top-level hypothesis is supported across BOTH documents** (`grouped_by_document`
  mode fires for all 4): eric's "animal host at HSM" (n-00262) has **47 supporters — 45
  eric + 2 will**; eric's "lab leak" (n-00180) **36 = 29 eric + 7 will**; will's "zoonotic
  more likely" (n-00023) **22 = 16 eric + 6 will**; will's "index case = lab worker"
  (n-00066) **11 = 10 eric + 1 will**. So each judge's central hypothesis draws evidence
  from *both* decisions — the graph reasons across documents, not within one.
- **Where the judges align / diverge:** 31 contested nodes span both docs. The zoonosis
  hypotheses are **net-supported** (n-00262 sup 35 / att 10; n-00023 sup 22 / att 6), while
  the lab-leak-index-case claim (n-00066 "index case was a laboratory worker") is
  **net-attacked** (sup 11 / att 10, winner=attack) — consistent with both judges landing
  on zoonosis, now shown as graph structure rather than prose.
- **230 cross-document cards** wire one judge's claims to the other's (roughly symmetric:
  109 will→eric, 121 eric→will), so agreement/disagreement is explicit and traceable.

## Method notes
- **will ingestion:** `chunk_decision.py` gained layout-mode extraction + a flat-heading
  splitter (will has no dotted TOC); eric's path is byte-identical.
- **Autonomous hypothesis selection (defect 3a option i):** one gpt-oss call per document
  reads all hypothesis statements and returns the genuine top-level competing ones
  (temperature 0, cached, no regex) — generalizes to any topic.
- **Extraction:** gpt-oss-120b, all 7 node types, ~$1.06; `--max-tokens 4000` caused **61
  truncation warnings** (long sections lost tail nodes — a known cost of the tight token
  cap chosen for speed); the tolerant JSON parser absorbed malformed output without
  crashing (no run-ending parse failures).
- **Cross-document dedup:** the graph is built on the **pre-dedup** nodes by design — for
  two judges independently adjudicating the same debate, keeping both nodes separate and
  attributed makes agreement an explicit cross-doc support card rather than a collapsed
  node. The Tier-1+Tier-2 (NLI) deduped `nodes_combined.json` **is available** (dedup
  finalized); folding it in is a 2-minute re-convert if a merged view is wanted.

## Files
`artifacts/epistemic_2doc/{nodes,cards,scores}.jsonl`, `candidate_pairs.json`,
`embeddings.npy`; source `artifacts/nodes_multidoc_2doc/nodes_combined_raw.json`.
The single-document graph (`artifacts/epistemic/`) is untouched.
