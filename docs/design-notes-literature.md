# Design notes — literature menu for the epistemic layer

> **STATUS: reference menu, NOT a committed plan.** Produced by an automated
> multi-agent research + design pass (36 agents; 1 design proposal lost to a
> network error). We are deliberately **not** committing to any of this until we've
> validated the baseline empirically (see [WORKLOG.md](../WORKLOG.md)). Treat it as
> a well-organized menu of options to react to. **It contains unverified and one
> fabricated citation — see §Caveats before citing anything.**

Convergence worth noting: this literature pass independently lands on the same
moves as our hands-on ingestion audit (WORKLOG finding 2026-07-16-d) — *reify
relations so they can carry a Bayes factor* (fixes the collapsed-predicate defect),
*gate on the 590 provenance-anchored claims*, *make correlated-evidence first-class*.

---

## The 8 load-bearing design principles (the durable takeaway)

1. **Reify relations as typed nodes, selectively.** Adopt AIF's RA/CA/PA so an
   inference/conflict/preference link can carry a Bayes factor, scheme,
   independence flag, and provenance — and can itself be attacked. Reify *only*
   when there's a payload; a plain support stays a simple typed edge. (This is the
   headline upgrade over HippoRAG's flat, predicate-erased triples.)
2. **Type attacks three ways, anchored to fixed targets:** undermine (premise) /
   undercut (the inference's applicability) / rebut (conclusion), each with an
   explicit target id; anchor every support/stance edge to an explicit hypothesis
   node (a Bayes factor is only meaningful between mutually-exclusive hypotheses).
3. **Separate structure from aggregation.** LLM proposes the typed graph; a
   *deterministic* additive-log-odds engine computes the verdict. The explanation
   *is* the computation → reproducible and contestable (edit a prior/weight/polarity
   and re-run = built-in sensitivity analysis).
4. **Model independence/correlation as first-class structure, never an
   assumption.** Shared-source / common-cause nodes bundle correlated items so
   their log-BFs are discounted *before* summation. Directly targets FLF's
   "correlated evidence treated as independent."
5. **Encode hierarchy as a DAG, not a tree:** Issue/sub-question tier (IBIS) over
   hypothesis→evidence layers over atomic claims; priors as serial-linked *product*
   sub-trees; reuse `sections.json` for document structure. (One observation can
   bear on several hypotheses — a tree can't represent that.)
6. **Keep three quantities distinct:** evidential weight (signed log-BF +
   uncertainty on the edge) ≠ retrieval relevance (HippoRAG's PPR score) ≠ model
   confidence. Shrink high-variance product chains toward a base rate; separate the
   *computed* posterior from the *endorsed* belief.
7. **Preserve "similar-but-not-identical" as typed variant edges.** Canonicalize
   for *matching* (KPA / claim-normalization / CESI) but never flatten — keep both
   verbatim quotes and per-source uncertainty; attach rival estimates (Peter 1/50
   vs Saar 1/500) to the *same* node. Do not merge at a similarity threshold (the
   0.95 e5 mistake, one level up).
8. **Gate on deterministic provenance; treat inferred structure as
   lower-confidence, schema-driven, adversarially verified.** Keep the exact/fuzzy
   verbatim-span match (claims.json) as the trust floor and the gate on every edge.
   Prefer domain-general LLM extraction + verification over fine-tuned classifiers
   (they don't transfer across COVID/LHC/eggs). Use Walton **critical-questions as
   checklists** to generate FLF's "what's missing"; store unanswered CQs as live
   gaps.

---

## Compact literature map (borrow / avoid, by cluster)

**Node/claim typologies.** Borrow AIF's **I-node vs S-node** split (content vs
reified scheme) as the backbone; Toulmin's warrant/qualifier as optional
methodology metadata (warrant is the *least* extractable element — don't gate on
it); IBIS Issue/Position/Argument for the sub-question tier; Carneades premise
typology (ordinary/assumption/exception) and dialectical status
(asserted/contested/conceded/excluded); IAM's (claim, stance, evidence,
evidence-type) atomic record; SciFact's claim + minimal-rationale span; the
nanopublication assertion+provenance+who-asserted triple. *Avoid* heavyweight
RDF/OWL ceremony and Carneades' ordinal weights (can't sum likelihood ratios).

**Edge/relation typologies.** Borrow one **signed, weighted likelihood-ratio edge**
per evidence item that updates two hypotheses in opposite directions (sign =
supports/refutes, magnitude = log-BF) — never duplicated as "support-A + attack-B"
(double-counts); combination types serial-linked (multiply) vs convergent (add);
scheme-typed inference edges that pull in a fixed critical-question set;
evidence–evidence corroborates/contradicts/restates-with-caveat. *Avoid* inferring
relation *direction* from embedding similarity (two opposing-stance claims on one
topic look most similar yet are a conflict).

**Hierarchy.** Borrow IBIS sub-question tree + the `sections.json` TOC promoted to
real edges + hypothesis→evidence **DAG** + prior product-decomposition sub-trees;
GraphRAG/RAPTOR rollup summaries for global views. *Avoid* deriving hierarchy from
Leiden/embedding **communities** — they file a claim and its rebuttal together and
record no attack. Structure must be argumentative, not statistical.

**Probability × argument graph.** The native backbone is **I.J. Good's weight of
evidence**: prior log-odds + Σ(log-BF) = posterior log-odds — exactly
eric_decision's calculus. Additivity holds *only under conditional independence*,
so dependency must be an edge. Timmer et al.'s **two-phase** (structure graph, then
instantiate BFs) maps 1:1 onto FLF's structure-vs-assessment split. Hahn & Oaksford
give schemes a Bayesian reading → the mechanism for "rhetorical vs evidential"
weight. *Avoid* full Bayes-net CPT elicitation (we rarely have the numbers) and
gradual-semantics engines that saturate/double-count.

**"Same claim, different form."** IBM **Key Point Analysis** (many mentions → few
key points + prevalence), **claim normalization**, **CESI** (canonicalize
entities *and* relations — the named fix for HippoRAG's un-canonicalized OpenIE),
Dense-X proposition quality gates (distinct/minimal/self-contained). Store
multiplicity as *provenance count only, never independent support*.

**Extractable vs fragile (set expectations).** *Reliable:* claim/evidence span
detection (F≈0.85), stance to a fixed target, KPA clustering, deterministic
verbatim provenance. *Fragile:* relation identification **and direction** (F≈0.45 —
the weakest link), warrant recovery, argumentation-scheme classification (κ≈0.38).
Mitigation: LLM messy-text→graph, then a symbolic/Bayesian layer for the verdict;
per-edge negate/invert adversarial check; independent judge model (never
extract-and-judge with the same model); per-layer confidence so inferred structure
never masquerades as extracted fact.

---

## One candidate synthesis — "EpiStack" (top-ranked design pass)

A thin typed overlay keyed on the existing `claim_id` / `chunk_id` /
`section_number` id-space, with a deterministic weight-of-evidence ledger. Recorded
here as the most complete single proposal — **still a menu item.**

**Node types:** Claim (the provenance floor, reused verbatim), Source, SubQuestion
(IBIS), Hypothesis, Estimate (prior / sub-prior / likelihood / Bayes-factor /
posterior), Inference (selectively-reified support carrying the log-BF + scheme +
CQs), Assumption, CorrelationGroup, ExcludedItem, Verdict, Discrepancy.

**Edge types:** grounded-in (the provenance gate, tiered T1/T2/T3), part-of,
attributed-to, decomposes-into (product sub-trees), answers, premise-of,
**weighs-on** (the signed likelihood-ratio edge — the crown jewel), derived-from
(auditable numeric composition), attacks{undermine|undercut|rebut},
rests-on-assumption, in-correlation-group, variant-of / rival-of, aggregates-into.

**Layers (bottom→top):** L0 provenance floor (590 claims, deterministic) · L1
document structure · L2 discourse/sub-question tier · L3 hypothesis→evidence DAG ·
L4 probabilistic ledger · L5 aggregation + calibration · L6 assessment cross-cut
(cruxes via sensitivity, double-count audits, CQ-gaps, settled-vs-performed).

**A three-tier provenance model** is its main honesty device: **T1** byte-exact
claim span · **T2** chunk-level parse for the ~5 aggregation numbers that did *not*
survive atomization into claims.json (e.g. the "secret doesn't leak" 1/10, the §7.6
log-odds columns) · **T3** model-inferred edge direction/scheme. A number's tier is
displayed, so the posterior discloses how much rides on weak tiers.

**Minimal first slice it proposes:** build *only* the §7 Bayesian core (~50 claims)
end-to-end — 2 hypotheses, the prior product-chains, the 4 scoring inferences
(furin BF 20, ACE2 BF 2, location 1/20000→1/10000, secret 1/10), the signed
weighs-on edges, one CorrelationGroup, the deterministic log-odds roll-up
reproducing 0.07529% **as a labeled consistency check**, crux sensitivity (toggling
location → P(LL)≈88%), and two adversarial tests (poison the location sign;
duplicate a correlated item).

---

## Caveats — read before relying on any of this

- **Unverified / fabricated citations flagged by the pass itself:**
  - *Timmer et al. 2017 "QPN sign-propagation / signed ± edges"* — **fabricated** by
    an automated summary; the real method keeps magnitudes and an **unsigned**
    support graph. Do not cite.
  - Argument-mining cross-dataset degradation numbers (e.g. "F collapses below
    0.5") — thesis real, exact figures unverified; possibly conflates two papers.
  - Fenton/Lagnado 2013, Bench-Capon & Atkinson CQ lists, ArgRAG formulas, IBM
    Debater dataset sizes — reconstructed from mirrors/HTML; verify against
    primary sources before hard-coding.
- **Honest limits the design admits:** reproducing the printed posterior proves
  *transcription + 5-term arithmetic*, **not** argument-extraction fidelity — real
  correctness weight has to sit on span-grounding, adversarial checks, an
  independent judge, and human spot-checks. Only ~8% of the 590 claims are
  quantified, so the deterministic engine governs a ~5-row core while ~92% is a
  qualitative map. Independence detection is a **heuristic** (shared source ≠
  conditional dependence). The "pure overlay, no re-ingestion" claim is false — the
  aggregation numbers need a T2 parse pass.
- **Generality (why it's not COVID-only):** the schema is domain-independent
  ({hypotheses, decomposed prior, observation, likelihood-ratio, correlation-group,
  exclusion, aggregation, calibration} + Walton CQs). For **LHC**: hypotheses
  {catastrophic vs safe}, priors as a product chain, cosmic-ray safety arguments as
  bounding inferences, a CorrelationGroup where astrophysical bounds share a
  premise. For **eggs**: {raises-CVD-risk vs neutral}, each cohort/RCT an inference,
  and the independence machinery earns its keep bundling studies sharing a cohort or
  an egg-board funder. *But the "reproduce a known posterior" validation is
  COVID-specific* — LHC/eggs have no single author-supplied log-odds table.

Full raw output (litMap §1–8, all 3 designs, judge scores, 14 build steps, full
evaluation + risks) is in the workflow transcript if we want it later.
