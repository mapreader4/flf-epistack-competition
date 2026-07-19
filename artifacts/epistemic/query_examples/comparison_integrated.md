# Comparison — graph-guided vs naive direct LLM

*Model: openai/gpt-oss-120b (both). Question and file identical; only the context differs.*

**Question:** What is the evidence that COVID originated at the Huanan Seafood Market? Include environmental sampling, spatial analysis of early cases, and animal susceptibility.

**Target hypothesis:** `n-00939` — "LetZ is the hypothesis that covid emerged from a zoonotic spillover in HSM." (graph strength 0.545)

## A. Naive baseline — gpt-oss with the full document (53020k tokens), no structure

**Evidence that the pandemic’s first detectable cluster arose in the Huanan Seafood Market (HSM)**  

| Type of evidence | What the data show | Where it is described in the document |
|------------------|-------------------|---------------------------------------|
| **Environmental sampling of the market** | • After the market was closed (1 Jan 2020) a total of **515 samples** were taken on the first day and a further **364 samples** between 12 Jan and 2 Mar.  <br>• **72 of those samples tested PCR‑positive for SARS‑CoV‑2 RNA**; three of the strongest‑positive samples yielded live virus that could be cultured.  <br>• Positive samples were **highly clustered**: 22 stores were sampled ≥ 2 times; only two stores (4:26 and 6:29) had multiple positives (5/9 and 5/10 samples respectively).  <br>• Shop 6:29 (south‑west part of the west half) also had live raccoon‑dogs – a species that is a known susceptible intermediate – and the **drainage system downstream of that shop was repeatedly positive** (four positive drain samples, three of them from the same drain).  <br>• No animal samples (457 total, most from dead animals) were positive, but the **lack of positivity is expected** because only 16 samples came from plausible carriers (bamboo rats). | 4.6.1 Environmental testing of HSM (p. 46‑48); 4.6.2 Correlation with animal DNA (p. 49‑50); 4.6.3 Animal‑vendor testing (p. 50‑51) |
| **Spatial distribution of the earliest human cases** | • The WHO’s 2021 report lists **174 known cases with onset in Dec 2019**.  <br>• Of the **55 cases that the WHO links to HSM**, **30 were vendors** (i.e., “resident” market workers) and the rest were visitors or indirect contacts (B.3, p. 73‑75).  <br>• Early genome sequencing shows that **12 of the 13 earliest sequenced cases were lineage B**, and **all 11 of those with a known epidemiological link were HSM‑associated** (5.2, p. 52‑53).  <br>• The **first documented cluster** (a household case discovered by Dr Zhang) was later found to have an epidemiological connection to the market, and the **majority of the earliest confirmed cases (Dec 13‑23) were either market vendors or had visited the market** (B.1, p. 71‑73).  <br>• Mapping of the market (Fig. 12) shows that the **early cases are concentrated in the west half**, the same area where the environmental positives are found. | 5.2 Spatial distribution of lineages A & B (p. 52‑53); B.1 WHO cases by date & connection to HSM (p. 71‑73); B.2 Map of HSM, west half (p. 72‑73); B.3 Vendor cases (p. 74‑75) |
| **Animal susceptibility and market composition** | • HSM housed **≈ 1 200 employees** (vendors, staff, daily shoppers) and **sold a wide variety of live wildlife**, many of which are known to be susceptible to SARS‑like coronaviruses (p. 6‑7).  <br>• Species that were present or sold at the market include **raccoon‑dogs, civets, bamboo rats, pangolins and other wild mammals** – the same taxa that have been shown experimentally to support bat‑derived coronaviruses (1.1.1, p. 6).  <br>• Environmental sequencing detected a **bamboo‑rat betacoronavirus** in market samples, confirming that **wild‑animal coronaviruses were present in the market environment** (4.6.1, p. 46).  <br>• The **high foot‑traffic, permanent resident vendors, enclosed/poor‑ventilation conditions** and **dense animal stalls** identified by Rootclaim (4.5) make the market a plausible setting for a zoonotic spill‑over event. | 1.1

## B. Graph-guided — gpt-oss with ranked evidence + ablation weakest-link

**Evidence that COVID‑19 originated at the Huanan Seafood Market (HSM)**  

| Type of evidence | Structured‑graph citation | Exact wording (verbatim) |
|------------------|--------------------------|--------------------------|
| **Environmental sampling** | n‑01022 (section 7.4) | “the distribution of positive sars‑cov‑2 samples within HSM appears to generally point to the region where live animals were kept, or more specifically point to shop 6:29 which had been identified as a potential pandemic spillover location.” |
|  | n‑00717 (section 4.6.1) | “Besides the 33 positive samples in stores, other locations in and around HSM also tested positive for sars‑cov‑2.” |
|  | n‑00792 (section 5.2) | “A glove in HSM (shop 7:15,17) was tested for covid, and was the one environmental sample to be sequenced with lineage A.” |
| **Spatial analysis of early cases** | n‑01012 (section 7.3) | “or, more precisely, that the first significant covid cluster is centered among the residents in the west half of HSM.” |
|  | n‑01111 (section A.3) | “Rootclaim identified the possibility that the mahjong room in HSM may have been a nexus of early covid cases.” |
| **Animal susceptibility / potential intermediaries** | n‑00716 (section 4.6.1) | “Store 6:29 had live raccoon dogs and was specifically identified in 2014 as a potential location of a novel disease spillover.” |
|  | n‑00048 (section 1.1.1) | “The market had live animals that were susceptible to sars‑like coronaviruses and were imported from south China.” |
|  | n‑00730 (section 4.6.2) | “Several species in HSM are suspected as being potential sars‑cov‑2 interme‑diaries (such as raccoon dog, civet, and bamboo rat);” |
|  | n‑00978 (section 7.1) | “Biologist Eddie Holmes and the Wuhan CDC specifically called out HSM as a possible location for a future pandemic spillover.” |

**Weakest link in the argument**  
The ablation analysis identified the following premise as the most load‑bearing (i.e., the weakest link when removed):

- **Card 00523** – Δ = 0.0338, kind = CA

## The graph-supplied structure (what B had that A didn't)

```json
{
  "hypothesis": {
    "node_id": "n-00939",
    "text": "LetZ is the hypothesis that covid emerged from a zoonotic spillover in HSM.",
    "strength": 0.545
  },
  "ranked_evidence": [
    {
      "node_id": "n-01022",
      "strength": 0.945,
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
      "node_id": "n-00048",
      "strength": 0.5,
      "section": "1.1.1",
      "quote": "The market did have live animals that were susceptible to sars-like coronaviruses and were imported from south China,",
      "text": "The market had live animals that were susceptible to sars-like coronaviruses and were imported from south China."
    },
    {
      "node_id": "n-00978",
      "strength": 0.5,
      "section": "7.1",
      "quote": "biologist Eddie Holmes and the Wuhan CDC specifically called out HSM as a possible location for a future pandemic spillover [26, 39].",
      "text": "Biologist Eddie Holmes and the Wuhan CDC specifically called out HSM as a possible location for a future pandemic spillover."
    },
    {
      "node_id": "n-00445",
      "strength": 0.5,
      "section": "4.2",
      "quote": "Notably, Saar observes [7] the possibility of zoonotic spillover at a restaurant that processes live animals",
      "text": "Saar observes the possibility of zoonotic spillover at a restaurant that processes live animals"
    },
    {
      "node_id": "n-01059",
      "strength": 0.713,
      "section": "7.6",
      "quote": "prior lean zoonotic",
      "text": "The prior evidence leans towards zoonosis."
    },
    {
      "node_id": "n-00419",
      "strength": 0.5,
      "section": "4.2",
      "quote": "LetZ be the event that zoonotic spillover of covid took place at an east Asian wildlife market,",
      "text": "The event that zoonotic spillover of covid took place at an east Asian wildlife market is denoted as Z"
    },
    {
      "node_id": "n-01085",
      "strength": 0.5,
      "section": "7.6",
      "quote": "the physical evidence of the covid pandemic starting in the immediate vicinity of live animals susceptible to sars-like coronaviruses and the precedent of previous coronavirus epidemics starting in an analogous manner together point strongly in favor of the hypothesis of zoonotic spillover.",
      "text": "The physical evidence of the COVID pandemic starting in the immediate vicinity of live animals susceptible to SARS-like coronaviruses favors the zoonotic spillover hypothesis."
    },
    {
      "node_id": "n-00760",
      "strength": 0.5,
      "section": "5.1",
      "quote": "If true, and sars-cov-2 underwent multiple introductions into humans, then we could be all but certain that the zoonotic hypothesis is correct",
      "text": "If sars-cov-2 underwent multiple introductions into humans, then we could be all but certain that the zoonotic hypothesis is correct."
    },
    {
      "node_id": "n-00050",
      "strength": 0.5,
      "section": "1.1.1",
      "quote": "Thus when multiple people who were employed at HSM became sick with a virus resembling sars-cov-1,",
      "text": "Multiple people who were employed at HSM became sick with a virus resembling sars-cov-1."
    },
    {
      "node_id": "n-00052",
      "strength": 0.5,
      "section": "1.1.1",
      "quote": "indeed in early January, before there existed adequate capacity for testing for covid, having an epidemiological link to the market was one component of a diagnosis of covid.",
      "text": "Having an epidemiological link to the market was one component of a diagnosis of covid in early January."
    },
    {
      "node_id": "n-00792",
      "strength": 0.5,
      "section": "5.2",
      "quote": "A glove in HSM (shop 7:15,17) was tested for covid, and was the one environmental sample to be sequenced with lineage A",
      "text": "A glove in HSM was tested for covid and was the one environmental sample to be sequenced with lineage A"
    },
    {
      "node_id": "n-01111",
      "strength": 0.5,
      "section": "A.3",
      "quote": "Rootclaim identified the possibility that the mahjong room in HSM may have been a nexus of early covid cases.",
      "text": "Rootclaim identified the possibility that the mahjong room in HSM may have been a nexus of early covid cases."
    },
    {
      "node_id": "n-00468",
      "strength": 0.5,
      "section": "4.3",
      "quote": "In section 4.5 we will see Rootclaim's argument that HSM was a likely place for an earlier outbreak to arise; a key part of that argument is the observation that HSM has \"permanent residents\"",
      "text": "Rootclaim argues that HSM was a likely place for an earlier outbreak to arise, in part because HSM has permanent residents."
    },
    {
      "node_id": "n-00295",
      "strength": 0.5,
      "section": "3.1",
      "quote": "They estimate that 15% of contact with potential animal-sources of pandemics is at HSM",
      "text": "Rootclaim estimates that 15% of contact with potential animal-sources of pandemics is at HSM"
    },
    {
      "node_id": "n-00981",
      "strength": 0.5,
      "section": "7.1",
      "quote": "Overall this gives P (Z) = 1 20\u00b7 1 4\u00b7 1 2\u00b7 1 200 = 1 32000,",
      "text": "The probability of a pandemic akin to sars-cov-2 emerging from HSM is estimated to be 1/32000."
    },
    {
      "node_id": "n-00730",
      "strength": 0.5,
      "section": "4.6.2",
      "quote": "Moreover, there are several species in HSM suspected as being potential sars-cov-2 interme- diaries (such as raccoon dog, civet, and bamboo rat);",
      "text": "Several species in HSM are suspected as being potential sars-cov-2 intermediaries, such as raccoon dog, civet, and bamboo rat."
    },
    {
      "node_id": "n-01012",
      "strength": 0.5,
      "section": "7.3",
      "quote": "or, more precisely, that the first significant covid cluster is centered among the residents in the west half of HSM",
      "text": "The first significant covid cluster is centered among the residents in the west half of HSM."
    },
    {
      "node_id": "n-00655",
      "strength": 0.5,
      "section": "4.5.5",
      "quote": "Rootclaim estimates that there was a 10% chance of the first covid outbreak appearing in HSM",
      "text": "Rootclaim estimates that there was a 10% chance of the first covid outbreak appearing in HSM"
    },
    {
      "node_id": "n-00479",
      "strength": 0.5,
      "section": "4.4",
      "quote": "most notably that in the first half of 2020 January having an epidemiological link to HSM was required for diagnosis.",
      "text": "In the first half of 2020, having an epidemiological link to HSM was required for diagnosis."
    },
    {
      "node_id": "n-00098",
      "strength": 0.5,
      "section": "1.2.1",
      "quote": "However, different parts of the sars-cov-2 genome are closest to different wild viruses,",
      "text": "Different parts of the sars-cov-2 genome are closest to different wild viruses."
    }
  ],
  "weakest_link_by_ablation": [
    {
      "card_id": "card-00523",
      "delta": 0.0338,
      "kind": "CA",
      "premise": "The most likely explanation for the lab leak hypothesis involves an early case going to HSM and starting a cluster of covid cases."
    },
    {
      "card_id": "card-01162",
      "delta": 0.0335,
      "kind": "CA",
      "premise": "Rootclaim evaluates 80% of pandemics that first appear in Wuhan to be lab leaks."
    },
    {
      "card_id": "card-00832",
      "delta": 0.0194,
      "kind": "CA",
      "premise": "Covid is contagious outside of HSM because it continued to grow even after HSM was closed."
    }
  ]
}
```
