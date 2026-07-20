# Two-Document Graph — Handoff / "Where is everything" (for a new chat)

Paste-ready orientation for the eric_decision + will_decision two-document epistemic graph.
All paths are repo-relative. Numbers verified on disk. Read `docs/system-design.md` for the
full design; this file is just the map.

## TL;DR
We built one typed argument graph spanning **two independent judges' decisions** on the
COVID-19 origins debate (`eric_decision.pdf` + `will_decision.pdf`). It reasons *across*
documents: every top-level hypothesis is supported by evidence from both judges, and the
graph surfaces where they agree/disagree.

**Headline numbers (2-doc graph):**
- **2,232 nodes** (eric 2,077 + will 155); node types: claim 567, assumption 426,
  evidence 396, estimate 380, rebuttal 310, subquestion 149, hypothesis 4.
- **1,678 cards** (RA 1,382 / CA 296); channels top_down 689, data→argument 633,
  argument→question 300, contradiction 56; hit rate 0.559.
- **230 cross-document cards** (109 Will→Eric, 121 Eric→Will).
- **82 contested nodes, 31 span both documents.**
- **4 LLM-selected top-level hypotheses**, all fire `grouped_by_document`:
  `n-00023` will/zoonotic-more-likely, `n-00066` will/index-case-lab-worker,
  `n-00180` eric/lab-leak, `n-00262` eric/animal-host-at-HSM.

## The 2-doc graph (artifacts/epistemic_2doc/)
| file | what |
|---|---|
| `nodes.jsonl` | 2,232 typed nodes; each `provenance[].document` = eric_decision.pdf / will_decision.pdf |
| `embeddings.npy` + `embeddings_index.json` | (2232 × 1024) e5 embeddings, row-aligned |
| `candidate_pairs.json` | 3,000 DeBERTa/NLI candidate pairs (Arpita, GPU, `--nli`) |
| `cards.jsonl` | 1,678 support/attack cards (labeled by gpt-oss) |
| `scores.jsonl` | DF-QuAD strengths for all 2,232 nodes |
| `cards_summary.json` | card counts by kind/channel |
| `REPORT.md` | **the deliverable** — filled with all cross-document numbers + findings |

## Extraction source (artifacts/nodes_multidoc_2doc/)
| file | what |
|---|---|
| `nodes_combined_raw.json` | 2,232 pre-dedup Lineage-B nodes — **the converter input we used** |
| `nodes_combined.json` | Tier-1+Tier-2(NLI) **deduped** version (optional upgrade; re-convert to fold in) |
| `dedup_groups.json`, `combined_stats.json`, `by_doc/`, `run_config.json` | dedup groups, stats, per-doc/per-type outputs |

## Code (the pipeline)
| script | role |
|---|---|
| `scripts/extraction-variants/epistemic_node_extraction_multidoc.py` | multi-doc extraction (7 types) + two-tier dedup (mapreader4) |
| `scripts/chunk_decision.py` | PDF→chunks; **edited**: `extraction_mode="layout"` + `flat_sections()`/`heading_regex` so will (no dotted TOC) ingests |
| `scripts/multidoc_to_graph_nodes.py` | **converter** Lineage-B → graph Node schema; per-doc stamping; **autonomous LLM top-level-hypothesis selection** (1 gpt-oss call/doc, defect-3a) |
| `scripts/embed_nodes.py` | nodes → embeddings |
| `scripts/pairing_funnel.py` | retype + 4-channel candidates (`--nli` for DeBERTa) + label + merge |
| `scripts/label_pairs.py` | label candidate pairs → cards |
| `scripts/score_dfquad.py` | cards → DF-QuAD scores |
| `scripts/query_epistemic_single_multidoc.py` | queries (evidence-for / contested / weakest-link, document-aware) |
| `scripts/verify_2doc.py` | **all cross-document checks in one shot** |
| `config_2doc.json` / `config_2doc_smoke.json` | extraction configs (will first, eric 50 main-body sections) |

## Docs & report assets
| file | what |
|---|---|
| `docs/system-design.md` | full design & context reference (12 sections, all numbers) |
| `docs/2doc-handoff.md` | this file |
| `artifacts/epistemic_2doc/REPORT.md` | 2-doc deliverable with verified numbers + cross-doc findings |
| `report/graph_vs_hipporag.tex` | **report LaTeX**: HippoRAG-vs-ours comparison table + 3 grounded query examples + pipeline figure (needs `booktabs`, `tikz`) |

## How to reproduce / query / verify
```bash
# verify all cross-document checks (per-doc counts, grouped_by_document, contested, cards)
.venv/bin/python scripts/verify_2doc.py --data-dir artifacts/epistemic_2doc

# query a top-level hypothesis (structured JSON, no API needed)
.venv/bin/python scripts/query_epistemic_single_multidoc.py evidence-for n-00262 \
    --data-dir artifacts/epistemic_2doc --prefix "" --no-llm
.venv/bin/python scripts/query_epistemic_single_multidoc.py contested \
    --data-dir artifacts/epistemic_2doc --prefix "" --no-llm

# full rebuild pipeline (extraction -> convert -> embed -> pair -> label -> score -> verify)
.venv/bin/python scripts/extraction-variants/epistemic_node_extraction_multidoc.py \
    --documents config_2doc.json --model openai/gpt-oss-120b --max-tokens 4000 \
    --save-dir artifacts/nodes_multidoc_2doc --require-nli
.venv/bin/python scripts/multidoc_to_graph_nodes.py \
    --in artifacts/nodes_multidoc_2doc/nodes_combined_raw.json \
    --out artifacts/epistemic_2doc/nodes.jsonl \
    --authors '{"eric_decision.pdf":"Eric Stansifer ...","will_decision.pdf":"Will ..."}'
.venv/bin/python scripts/embed_nodes.py --nodes artifacts/epistemic_2doc/nodes.jsonl \
    --out artifacts/epistemic_2doc/embeddings.npy --index artifacts/epistemic_2doc/embeddings_index.json
.venv/bin/python scripts/pairing_funnel.py --nli --no-label --nodes artifacts/epistemic_2doc/nodes.jsonl \
    --emb artifacts/epistemic_2doc/embeddings.npy --emb-index artifacts/epistemic_2doc/embeddings_index.json \
    --out artifacts/epistemic_2doc/candidate_pairs.json --max-llm-pairs 3000        # GPU
.venv/bin/python scripts/label_pairs.py --pairs artifacts/epistemic_2doc/candidate_pairs.json \
    --nodes artifacts/epistemic_2doc/nodes.jsonl --out artifacts/epistemic_2doc/cards.jsonl \
    --cache outputs/epistemic_2doc/label_cache.json                                  # SEPARATE cache (id collision)
.venv/bin/python scripts/score_dfquad.py --nodes artifacts/epistemic_2doc/nodes.jsonl \
    --cards artifacts/epistemic_2doc/cards.jsonl --out artifacts/epistemic_2doc/scores.jsonl
```

## Single-document graph (reference, artifacts/epistemic/)
1,370 nodes → 1,253 cards → scores. Flagship hypothesis `n-00939` (HSM zoonosis): depth-2,
contested, evidence chain reaches raw §4.6.1 data after the FIX 1/2 depth fixes.

## Git state
- Branch `main`, at `17cfdb4` (origin). 2-doc results (nodes/pairs/cards/scores/REPORT,
  converter, verify_2doc, system-design) are committed.
- **Uncommitted:** `report/graph_vs_hipporag.tex` (new report asset) — commit when ready.

## Caveats (for honesty in the writeup)
- Built on **pre-dedup** nodes by design (both judges' nodes kept separate + attributed;
  agreement = explicit cross-doc support cards). Deduped `nodes_combined.json` is available.
- Extraction used gpt-oss `--max-tokens 4000` → ~61 truncation warnings (long sections lost
  tail nodes); tolerant JSON parser absorbed malformed output, no run-ending failures.
- Hypothesis selection is one gpt-oss call/doc (temperature 0, cached) — reproducible and
  topic-general (works for eggs/LHC unchanged); replaced an earlier fragile regex.
- 2-doc node ids reuse the `n-00000+` space → **label with a separate `--cache`** to avoid
  collisions with the single-doc graph's cache.
