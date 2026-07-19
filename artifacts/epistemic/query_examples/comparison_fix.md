# Comparison — graph-guided vs naive direct LLM

*Model: openai/gpt-oss-120b (both). Question and file identical; only the context differs.*

**Question:** What is the evidence that COVID originated at the Huanan Seafood Market? Include environmental sampling, spatial analysis of early cases, and animal susceptibility.

**Target hypothesis:** `n-00939` — "LetZ is the hypothesis that covid emerged from a zoonotic spillover in HSM." (graph strength 0.503)

## A. Naive baseline — gpt-oss with the full document (53020k tokens), no structure

**Evidence that the first COVID‑19 infections clustered at the Huanan Seafood Market (HSM) comes from three complementary lines of data – environmental testing, the geography of the earliest human cases, and the presence of animal hosts that can carry a SARS‑like virus.**  

| What the data show | Where it is described in the report |
|--------------------|--------------------------------------|
| **Environmental sampling** – After the market was closed on 1 Jan 2020, 515 swabs were taken on the first day and a further 364 between 1 Jan and 2 Mar. 72 of the 871 total samples were PCR‑positive for SARS‑CoV‑2 RNA and three of those yielded live, culturable virus. The positives were not spread uniformly; they were concentrated in a few stalls in the west‑half of the market, especially shop 6:29 (5 positives out of 9 samples) and the nearby shop 4:26 (5 positives out of 9 samples). Drainage samples taken in late Jan and early Feb also repeatedly tested positive in the same area (Fig. 8). | 4.6.1 – Environmental testing of HSM (pp. 46‑47); 4.6.2 – Correlation of animal DNA with SARS‑CoV‑2 positivity (pp. 48‑49); Fig. 8 (p. 48). |
| **Spatial distribution of the earliest human cases** – The WHO’s 2021 investigation identified 174 cases with onset in December 2019. Of the 13 cases that have been sequenced, 12 were lineage B and **all 11 cases that could be linked to a market location were from the west‑half of HSM** (see the WHO case‑by‑date table, B.1, and the stall‑level map, B.2). The first documented cluster (a shrimp‑vendor) and the first household cluster (the “Chen” case) were both tied to HSM. This tight epidemiological clustering is highlighted in the background section (1.1.1) and the detailed inventory of early cases (B.1‑B.3). | 1.1.1 (HSM background, p. 6); 5.2 (Spatial distribution of lineages A and B, p. 51‑52); B.1 (WHO cases by date, p. 71); B.2 (HSM map, p. 72); B.3 (Vendor‑case table, p. 74‑75). |
| **Animal susceptibility at the market** – HSM housed ≈ 1 200 employees and ≈ 1 200 stalls, many of which sold live wildlife (including raccoon‑

## B. Graph-guided — gpt-oss with ranked evidence + ablation weakest-link

**Evidence that COVID‑19 originated at the Huanan Seafood Market (HSM)**  

| Aspect | Structured evidence (section / quote) |
|--------|----------------------------------------|
| **Environmental sampling** | • *Section 4.6.1* – “Besides the 33 positive samples in stores, other locations in and around HSM also tested positive for sars‑cov‑2.” 【n‑00717】<br>• *Section 7.4* – “the distribution of positive sars‑cov‑2 samples within HSM appears to generally point to the region where live animals were kept, or more specifically point to shop 6:29 which had been identified as a potential pandemic spillover location.” 【n‑01022】 |
| **Spatial analysis of early cases** | • *Section 7.6* – “the physical evidence of the covid pandemic starting in the immediate vicinity of live animals susceptible to sars‑like coronaviruses and the precedent of previous coronavirus epidemics starting in an analogous manner together point strongly in favor of the hypothesis of zoonotic spillover.” 【n‑01085】 |
| **Animal susceptibility** | • *Section 1.1.1* – “The market did have live animals that were susceptible to sars‑like coronaviruses and were imported from south China,” 【n‑00048】<br>• *Section 4.6.1* – “The latter store in particular had live raccoon dogs (one of the species suspected to be intermediary for sars‑cov‑2) and was specifically identified in 2014 as a potential location of a novel disease spillover [26, 39].” 【n‑00716】 |

**Weakest link in the argument**  
The ablation analysis identified the following premise as the weakest link:  

- *Card‑01395* – “Rootclaim's Bayesian calculation counts the location of the initial outbreak as favoring LL by a Bayes factor of 13.48.” (Δ = 0.0073, kind = CA)  

This premise contributes the smallest change to the overall strength when removed, indicating it is the most load‑bearing (i.e., weakest) component of the evidence chain.

## The graph-supplied structure (what B had that A didn't)

```json
{
  "hypothesis": {
    "node_id": "n-00939",
    "text": "LetZ is the hypothesis that covid emerged from a zoonotic spillover in HSM.",
    "strength": 0.503
  },
  "ranked_evidence": [
    {
      "node_id": "n-01085",
      "strength": 0.962,
      "section": "7.6",
      "quote": "the physical evidence of the covid pandemic starting in the immediate vicinity of live animals susceptible to sars-like coronaviruses and the precedent of previous coronavirus epidemics starting in an analogous manner together point strongly in favor of the hypothesis of zoonotic spillover.",
      "text": "The physical evidence of the COVID pandemic starting in the immediate vicinity of live animals susceptible to SARS-like coronaviruses favors the zoonotic spillover hypothesis."
    },
    {
      "node_id": "n-00048",
      "strength": 0.5,
      "section": "1.1.1",
      "quote": "The market did have live animals that were susceptible to sars-like coronaviruses and were imported from south China,",
      "text": "The market had live animals that were susceptible to sars-like coronaviruses and were imported from south China."
    },
    {
      "node_id": "n-00445",
      "strength": 0.5,
      "section": "4.2",
      "quote": "Notably, Saar observes [7] the possibility of zoonotic spillover at a restaurant that processes live animals",
      "text": "Saar observes the possibility of zoonotic spillover at a restaurant that processes live animals"
    },
    {
      "node_id": "n-01022",
      "strength": 0.953,
      "section": "7.4",
      "quote": "the distribution of positive sars-cov-2 samples within HSM appears to generally point to the region where live animals were kept, or more specifically point to shop 6:29 which had been identified as a potential pandemic spillover location.",
      "text": "The distribution of positive sars-cov-2 samples within HSM appears to generally point to the region where live animals were kept, or more specifically point to shop 6:29 which had been identified as a potential pandemic spillover location."
    },
    {
      "node_id": "n-00717",
      "strength": 0.5,
      "section": "4.6.1",
      "quote": "Besides the 33 positive samples in stores, other locations in and around HSM also tested positive for sars-cov-2.",
      "text": "Besides the 33 positive samples in stores, other locations in and around HSM also tested positive for sars-cov-2."
    },
    {
      "node_id": "n-00716",
      "strength": 0.5,
      "section": "4.6.1",
      "quote": "The latter store in particular had live raccoon dogs (one of the species suspected to be intermediary for sars-cov-2) and was specifically identified in 2014 as a potential location of a novel disease spillover [26, 39].",
      "text": "Store 6:29 had live raccoon dogs and was specifically identified in 2014 as a potential location of a novel disease spillover."
    },
    {
      "node_id": "n-01059",
      "strength": 0.75,
      "section": "7.6",
      "quote": "prior lean zoonotic",
      "text": "The prior evidence leans towards zoonosis."
    },
    {
      "node_id": "n-00445",
      "strength": 0.5,
      "section": "4.2",
      "quote": "Notably, Saar observes [7] the possibility of zoonotic spillover at a restaurant that processes live animals",
      "text": "Saar observes the possibility of zoonotic spillover at a restaurant that processes live animals"
    }
  ],
  "weakest_link_by_ablation": [
    {
      "card_id": "card-01395",
      "delta": 0.0073,
      "kind": "CA",
      "premise": "Rootclaim's Bayesian calculation counts the location of the initial outbreak as favoring LL by a Bayes factor of 13.48."
    },
    {
      "card_id": "card-01402",
      "delta": 0.0025,
      "kind": "CA",
      "premise": "The prior of 0.2321 has a Bayes factor of 0.2321/0.7679."
    },
    {
      "card_id": "card-01380",
      "delta": 0.0018,
      "kind": "CA",
      "premise": "The Bayes factor for the outbreak at HSM is -9.2."
    }
  ]
}
```
