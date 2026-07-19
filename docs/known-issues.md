# Known issues — argument-graph pipeline

The pipeline is a **chain** (extract → type → role → attribute → embed → funnel →
label → score → query). Each stage's error rate multiplies down the line, so a
tolerable-looking per-stage error compounds into a bad final answer. Two issues found
so far are the priorities; both have a clear root cause and fix.

---

## P0 — Graph coverage gap (found in the baseline comparison)

**Symptom.** In `docs/end-to-end-verification.md` / `query_examples/comparison.md`, the
naive "dump the whole document into gpt-oss" baseline gave a *better* answer than our
graph-guided path. Asked for the strongest evidence that COVID began at the Huanan
Seafood Market, the baseline surfaced the compelling record — 72 positive environmental
swabs, live virus isolated from 3, case clustering at specific stalls — while the
graph-guided answer surfaced only a narrow epidemiological *modeling* claim, and named
the **same card** as both strongest evidence and weakest link.

**Root cause.** The funnel under-covers the argument structure:
1. **Cosine-only ranking + a 1000-pair cap** over ~900k type-legal pairs (1370 nodes)
   means we label a tiny fraction, ranked by *topical similarity* — which favors
   near-duplicates over genuine support/attack links. The environmental-sampling
   evidence uses different vocabulary from the hypothesis node, so it never enters the
   top-cosine candidates → never labeled → **no support card → invisible to Query 1**.
2. **NLI Stage-2 is disabled** (too slow on CPU), so we lose the entailment/
   contradiction signal that would rank real relations above topical look-alikes.
3. **No question→node mapping** — the entry node was hand-picked (`n-00502`, a narrow
   sub-claim), not resolved from the question, so even the cards that exist weren't the
   right ones to walk.

**Impact.** Low recall: the graph can only cite what it linked, so when it misses the
key evidence the answer is thinner *and* less correct than a naive call. On a single
document that fits in one context window, the naive baseline is hard to beat — our edge
only appears once coverage is good and/or the corpus is too big to dump wholesale.

**Fix (top priority), in order:**
1. **Enable NLI Stage-2 on a GPU** (`pairing_funnel.py --device cuda`) — rank candidates
   by entailment/contradiction, not just cosine. (Handoff paragraph already drafted.)
2. **Raise `MAX_LLM_PAIRS`** to ~3–5k — labeling is ~$0.07/1k, so still cheap; more
   pairs = denser, higher-recall card set.
3. **Strengthen candidate channels** so key evidence reliably connects to hypotheses:
   embedding-kNN + section-adjacency + hypothesis-hub linking (not topical cosine alone).
4. **Build question→node mapping** (embed the question, pick the nearest hypothesis
   node) so queries start from the right entry.
5. **Re-measure** with `compare_vs_baseline.py` on the denser graph — this is the
   rematch where the graph-guided path should win.

---

## P1 — Source attribution defaults to "unattributed"

**Symptom.** `attribute_sources.py` returned **960 / 1370 claims as `unattributed`**,
including 144 claims that role-tagging marked `conclusory` (the judge's own verdict).
The attribution under-catches the document author's own voice.

**Root cause.** The model attributes from the **claim text alone**, with no knowledge of
who authored the document. When the judge writes impersonally ("the odds are therefore
~1/50", "this makes lab-leak less likely") there is no source phrase in the sentence, so
the model falls back to `unattributed` — even though, in a single-authored decision,
*the author is asserting it*. Role-tagging avoided this because it reasoned about the
document's structure, not just the sentence.

**Impact.** Corroboration/independence logic (the reason attribution exists) would treat
the author's own many assertions as source-less instead of as one voice — exactly the
failure mode attribution is meant to prevent. It also disagrees with the role layer,
and those disagreements compound downstream.

**Fix.** Default to the **document's author** rather than `unattributed`. Any claim that
doesn't explicitly credit a party, cited source, or witness is asserted by the author —
for this case, the **judge**. Pass the author identity into the prompt (a per-document
`--author` parameter, keeping the system domain-general); gpt-oss can then confidently
label impersonal claims as `adjudicator` / source_id = the author. This collapses the
960 `unattributed` into the author's voice and reconciles attribution with the role
layer. Re-run `attribute_sources.py` after the change.

---

## Cross-cutting: guard every stage against compounding error
Each stage should be validated against an independent sanity signal so errors are caught
where they enter, not after they've propagated:
- **Roles** — leakage guard (0 cards touch a conclusory node). ✅ enforced.
- **Scoring** — two independent DF-QuAD implementations agree to 5e-7. ✅ checked.
- **Attribution** — crosstab vs roles (cited_source/witness ⇒ evidentiary; adjudicator ⇒
  conclusory). Surfaced the P1 issue above.
- **Coverage** — the baseline comparison is the sanity signal for the graph itself;
  keep running it as the graph densifies.
