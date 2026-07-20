#!/usr/bin/env bash
# Full query pipeline over the epistemic argument graph, walking every query
# type against one target so the whole surface area is exercised in one pass.
#
# H=n-00168 ("Rootclaim's claim... gain-of-function laboratory research") is
# the demo target: it's the richest-connected claim reaching real content on
# both the supporting and negating side across every NODE_TYPE, so nearly
# every command below returns real, non-empty results (verified by hand --
# see the per-section notes). Swap H to n-00939/n-00940 (the two top-level
# Z/LL hypotheses) for the "official" hypotheses instead -- they're real but
# sparser in the graph, so several commands legitimately return empty for
# them (not a bug -- e.g. their support chains don't reach an estimate-type
# node within 6 hops).
#H=n-00669
#H=n-00024
#H=n-00658
#H=n-00168
#H=n-00169
#H=n-00023 #multi-document node
#H=n-00180 
#D="artifacts/epistemic_priorOnes"
D="artifacts/epistemic_2doc"
Q="python scripts/query_epistemic_single_multidoc.py"

# ---------------------------------------------------------------------------
# 0. THE RESEARCH QUESTION, surfaced implicitly -- there's no literal
#    research-question node in this corpus, so the closest thing is ranking
#    all hypothesis-type nodes against each other.
# ---------------------------------------------------------------------------
$Q rank-hypotheses --data-dir "$D" --prefix "" --no-llm

# ---------------------------------------------------------------------------
# 1. THE RAW CHAIN -- support-only, any type, tree-shaped (groups by document
#    once it spans more than one source). This is the "no filter" baseline
#    every type-specific query below is a narrower view of.
# ---------------------------------------------------------------------------
#$Q evidence-for "$H" --data-dir "$D" --prefix "" --no-llm

# ---------------------------------------------------------------------------
# 2. EVERY REAL NODE_TYPE, split by polarity (supporting_X / negating_X).
#    Each of these walks BOTH the support and attack chain of $H, filtered to
#    one type, with polarity flipping correctly across an attack edge (see
#    _collect_polarized). Verified counts for n-00168 (--no-llm, checked by
#    hand): claims-for 3/1, evidence-only-for 4/3, background-for 1/1 all
#    return real results on both sides. assumptions-for and rebuttals-for are
#    genuinely 0/0 for THIS target -- real NODE_TYPES, just not reached by
#    n-00168's chain within 6 hops (not every type appears in every node's
#    neighborhood). verdicts-for is also usually empty (leakage guard mostly
#    keeps the judge's own conclusions out of the card graph, though not
#    perfectly -- a handful of verdict nodes do participate elsewhere in the
#    corpus). analysis-for / limitations-for are NOT included here at all:
#    "analysis" and "limitation" aren't NODE_TYPES in the current schema
#    (see their docstrings) -- they'd always return empty regardless of $H.
# ---------------------------------------------------------------------------
# $Q claims-for "$H" --data-dir "$D" --prefix "" --no-llm
$Q evidence-only-for "$H" --data-dir "$D" --prefix "" --no-llm
$Q background-for "$H" --data-dir "$D" --prefix ""
$Q assumptions-for "$H" --data-dir "$D" --prefix "" 
$Q rebuttals-for "$H" --data-dir "$D" --prefix "" --no-llm
$Q verdicts-for "$H" --data-dir "$D" --prefix "" --no-llm

# # ---------------------------------------------------------------------------
# 3. QUANTITATIVE -- the estimate-type nodes (priors, Bayes factors, the ~8%
#    of claims whose content IS a number). quantitative-backbone lists every
#    one in the whole graph, ungrouped; quantitative-evidence-for scopes
#    that down to just the ones bearing on $H, split by polarity.
# ---------------------------------------------------------------------------
#$Q quantitative-backbone --data-dir "$D" --prefix "" --no-llm
#$Q quantitative-evidence-for "$H" --data-dir "$D" --prefix "" --max-depth 15

# ---------------------------------------------------------------------------
# 4. POLARITY ONLY -- every node touched by the walk regardless of type,
#    split so you can ask for just one side. Same walk as section 2, just
#    with a match-everything predicate instead of a type filter.
# ---------------------------------------------------------------------------
$Q supporting-for "$H" --data-dir "$D" --prefix "" 
$Q negating-for "$H" --data-dir "$D" --prefix ""

# # ---------------------------------------------------------------------------
# # 5. ARGUMENTS -- where the graph disagrees with itself, and what's most
# #    load-bearing. weakest-link-scoped gives identical numbers to
# #    weakest-link but only tests cards inside $H's own dependency closure
# #    (~19 cards vs. 303 total in the corpus) -- much faster, same result.
# # ---------------------------------------------------------------------------
# $Q contested --data-dir "$D" --prefix "" --no-llm
# $Q weakest-link-scoped "$H" --data-dir "$D" --prefix "" --no-llm


#----------------------------------------------------------------------------

#####Older Pipeline:
# 1. THE RESEARCH QUESTION, surfaced implicitly — both competing hypotheses ranked
# python scripts/query_epistemic_single_multidoc.py rank-hypotheses \
#     --data-dir artifacts/epistemic --prefix "" --no-llm

# # 2. SUPPORTING hypothesis chain — lab-leak (n-00940), walks its claims/evidence
# python scripts/query_epistemic_single_multidoc.py evidence-for n-00940 \
#     --data-dir artifacts/epistemic --prefix "" --no-llm

# # 3. ALTERNATIVE hypothesis chain — zoonosis (n-00939), same shape, contrasting result
# python scripts/query_epistemic_single_multidoc.py evidence-for n-00939 \
#     --data-dir artifacts/epistemic --prefix "" --no-llm

# # 4. QUANTITATIVE RESULTS — the estimate-type nodes (priors, Bayes factors) and what they feed into
# python scripts/query_epistemic_single_multidoc.py quantitative-backbone \
#     --data-dir artifacts/epistemic --prefix "" --no-llm

# # 5. ARGUMENTS in tension — where the two hypotheses' evidence actually clashes
# python scripts/query_epistemic_single_multidoc.py contested \
#     --data-dir artifacts/epistemic --prefix "" --no-llm

# # 6. ARGUMENTS, load-bearing — what's propping up (or undermining) each hypothesis most
# python scripts/query_epistemic_single_multidoc.py weakest-link n-00940 \
#     --data-dir artifacts/epistemic --prefix "" --no-llm
# python scripts/query_epistemic_single_multidoc.py weakest-link n-00939 \
#     --data-dir artifacts/epistemic --prefix "" --no-llm
