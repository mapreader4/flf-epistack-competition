# Comparison — graph-guided vs naive direct LLM

*Model: openai/gpt-oss-120b (both). Question and file identical; only the context differs.*

**Question:** What is the evidence that COVID originated at the Huanan Seafood Market? Include environmental sampling, spatial analysis of early cases, and animal susceptibility.

**Target hypothesis:** `n-00939` — "LetZ is the hypothesis that covid emerged from a zoonotic spillover in HSM." (graph strength 0.503)

## A. Naive baseline — gpt-oss with the full document (53020k tokens), no structure

**Evidence that the first COVID‑19 infections clustered at the Huanan Seafood Market (HSM)**  

| Type of evidence | What the document reports | Where it is described |
|------------------|--------------------------|-----------------------|
| **Environmental sampling** | • After the market was closed (1 Jan 2020) 515 surface swabs were taken on the first day and a further 364 between 12 Jan and 2 Mar.  <br>• A second round (18 Jan – 30 Mar) collected 457 animal specimens (mostly dead animals, a few live‑animal or fecal samples).  <br>• None of the animal specimens tested positive for SARS‑CoV‑2, which is unsurprising because most were taken from dead animals stored in cold‑storage and only 16 of the 457 came from species that could plausibly carry the virus (six bamboo‑rat samples).  <br>• Seventy‑two environmental samples were PCR‑positive for SARS‑CoV‑2 RNA; three of these yielded live virus in culture.  <br>• Positive samples were not spread uniformly – they clustered in the southwest part of the west half of the market.  Two stalls (shop 4:26‑28 and shop 6:29) accounted for 10 of the 15 positive samples (5/9 and 5/10 respectively).  The latter stall sold live raccoon‑dogs, a species that is a known candidate intermediate host.  <br>• Drainage‑pipe samples taken on 27‑29 Jan and again on 9‑15 Feb showed four positives, three of which came from the same drainage system that serves shop 6:29, reinforcing the geographic focus of the contamination. | **4.6.1** (p. 46‑48) – environmental and animal testing; **4.6.2** (p. 49‑50) – correlation with animal DNA; **4.6.3** (p. 50‑51) – animal‑vendor testing. |
| **Spatial distribution of the earliest human cases** | • The WHO “early‑case” dataset (174 cases with onset in Dec 2019) shows that the overwhelming majority of the earliest cases had an epidemiological link to HSM.  In the WHO annex (p. 71‑73) 30 cases are listed as “vendor”, 25 as “visitor” and only 6 as “unknown”; the rest are non‑HSM cases.  <br>• Of the 13 cases that have been sequenced from Dec 2019, 12 were linked to HSM and all 12 belong to lineage B.  The single lineage‑A case was from a patient who lived 2 km south‑east of the market.  <br>• The first documented cluster (the “index” cluster) was centred on the west half of the market; the earliest confirmed infections (e.g., the “household” cluster described by Dr Zhang on 8‑9 Dec) were later traced back to market exposure.  <br>• Mapping of vendor stalls (Appendix B.2, p. 72‑73) shows that most of the 30 vendor cases are concentrated in the west half, especially in alleys 4, 6, 7 and 9 – the same area where the positive environmental samples were found. | **1.1.1** (p. 6‑7) – description of the market and early case numbers; **5.2** (p. 51‑52) – lineage‑A vs‑B distribution; **B.1** (p. 71‑73) – WHO case table; **B.2** (p. 72‑73) – market map and stall locations. |
| **Animal susceptibility and market‑animal trade** | • HSM housed ~1 200 vendors (≈ 1 162 according to the WHO annex) and sold a wide variety of live wildlife, much of it imported from southern China – the region where the closest known bat coronavirus (banal‑20‑52) was found.  <br>• Although no bats were sold, the market did keep live animals that are known to be susceptible to SARS‑like coronaviruses (e.g., raccoon‑dogs, civets, bamboo‑rats).  <br>• Metagenomic testing of environmental samples detected DNA from several of these wildlife species, confirming that they were present in the market environment.  <br>•

## B. Graph-guided — gpt-oss with ranked evidence + ablation weakest-link

**Evidence that COVID‑19 originated at the Huanan Seafood Market (HSM)**  

| Type of evidence | Structured‑graph citation (section) | Verbatim quote |
|------------------|--------------------------------------|----------------|
| **Physical‑environment sampling** | 4.6.1 (n‑00703) | “Second, 72 of the environmental samples tested positive for the presence of sars‑cov‑2 RNA by PCR, and from 3 of them live virus was able to be extracted and cultured, which were the same 3 that tested most strongly for sars‑cov‑2;” |
|  | 4.6.1 (n‑00717) | “Besides the 33 positive samples in stores, other locations in and around HSM also tested positive for sars‑cov‑2.” |
|  | 4.6.1 (n‑00705) | “2 of the 3 came from stores with known covid cases” |
|  | 7.4 (n‑01022) | “the distribution of positive sars‑cov‑2 samples within HSM appears to generally point to the region where live animals were kept, or more specifically point to shop 6:29 which had been identified as a potential pandemic spillover location.” |
|  | 4.6.1 (n‑00716) | “The latter store in particular had live raccoon dogs (one of the species suspected to be intermediary for sars‑cov‑2) and was specifically identified in 2014 as a potential location of a novel disease spillover [26, 39].” |
|  | 4.6.1 (n‑00730) | “Moreover, there are several species in HSM suspected as being potential sars‑cov‑2 interme‑ diaries (such as raccoon dog, civet, and bamboo rat);” |
| **Spatial / epidemiological analysis of early cases** | 4.5.6 (n‑00682) | “whereas we instead observe that HSM cases are slightly frontloaded to earlier in the outbreak.” |
|  | 4.5 (n‑00492) | “Since residents at HSM are 0.01% of the population of Wuhan, this is equivalent to saying that residents at HSM have 1000 times the chance of being the index case than the average Wuhan resident.” |
|  | 4.5 (n‑00488) | “The simplistic model used in the previous subsection suggests that the early outbreak being observed at HSM gives a Bayes factor of 100 to 10000 in favor of Z over LL,” |
|  | 7.3 (n‑01014) | “For Z this is simply the west half of HSM” (referring to the smallest epidemiological circle that contains both the primary case and the index case). |
|  | 4.5.1 (n‑00528) | “at 10% ascertainment, this falls to 0.00014% chance.” (probability that none of the 128 pre‑HSM cases are found). |
| **Animal susceptibility & market context** | 1.1.1 (n‑00048) | “The market did have live animals that were susceptible to sars‑like coronaviruses and were imported from south China,” |
|  | 4.2 (n‑00436) | “for a Bayes factor of 100 to 1000 in favor of zoonotic spillover” (when considering live‑animal presence). |
|  | 7.6 (n‑01085) | “the physical evidence of the covid pandemic starting in the immediate vicinity of live animals susceptible to sars‑like coronaviruses and the precedent of previous coronavirus epidemics starting in an analogous manner together point strongly in favor of the hypothesis of zoonotic spillover.” |
| **Overall strongest support** | 7.6 (n‑01085) | Same quote as above, emphasizing “physical evidence … point strongly in favor of the hypothesis of zoonotic spillover.” |

---

### Weakest link in the argument

The ablation analysis identified the **card‑01395** as the weakest link:

- **Delta:** 0.0073 (largest change when removed)  
- **Kind:** CA (causal assumption)  
- **Premise:** “Rootclaim's

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
      "node_id": "n-00098",
      "strength": 0.5,
      "section": "1.2.1",
      "quote": "However, different parts of the sars-cov-2 genome are closest to different wild viruses,",
      "text": "Different parts of the sars-cov-2 genome are closest to different wild viruses."
    },
    {
      "node_id": "n-00703",
      "strength": 0.5,
      "section": "4.6.1",
      "quote": "Second, 72 of the environmental samples tested positive for the presence of sars-cov-2 RNA by PCR, and from 3 of them live virus was able to be extracted and cultured, which were the same 3 that tested most strongly for sars-cov-2;",
      "text": "72 of the environmental samples tested positive for the presence of sars-cov-2 RNA by PCR."
    },
    {
      "node_id": "n-00978",
      "strength": 0.5,
      "section": "7.1",
      "quote": "biologist Eddie Holmes and the Wuhan CDC specifically called out HSM as a possible location for a future pandemic spillover [26, 39].",
      "text": "Biologist Eddie Holmes and the Wuhan CDC specifically called out HSM as a possible location for a future pandemic spillover."
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
      "node_id": "n-00705",
      "strength": 0.5,
      "section": "4.6.1",
      "quote": "2 of the 3 came from stores with known covid cases",
      "text": "2 of the 3 samples from which live virus was extracted came from stores with known covid cases."
    },
    {
      "node_id": "n-00730",
      "strength": 0.5,
      "section": "4.6.2",
      "quote": "Moreover, there are several species in HSM suspected as being potential sars-cov-2 interme- diaries (such as raccoon dog, civet, and bamboo rat);",
      "text": "Several species in HSM are suspected as being potential sars-cov-2 intermediaries, such as raccoon dog, civet, and bamboo rat."
    },
    {
      "node_id": "n-01059",
      "strength": 0.75,
      "section": "7.6",
      "quote": "prior lean zoonotic",
      "text": "The prior evidence leans towards zoonosis."
    },
    {
      "node_id": "n-00972",
      "strength": 0.5,
      "section": "7.1",
      "quote": "in addition to population density, other important factors are \"forested tropical regions experiencing land-use changes and where wildlife biodiversity (mammal species richness) is high\".",
      "text": "Population density and wildlife biodiversity are important factors in the emergence of zoonotic diseases."
    },
    {
      "node_id": "n-00738",
      "strength": 0.5,
      "section": "4.6.3",
      "quote": "a significant association between people who worked with wild animals and seropositivity for SARS-1",
      "text": "There is a significant association between people who worked with wild animals and seropositivity for SARS-1."
    },
    {
      "node_id": "n-00745",
      "strength": 0.5,
      "section": "4.6.3",
      "quote": "It does say that 20 of 168 December cases self-report exposure to live animals",
      "text": "20 of 168 December cases self-report exposure to live animals, according to the WHO report."
    },
    {
      "node_id": "n-00682",
      "strength": 0.5,
      "section": "4.5.6",
      "quote": "whereas we instead observe that HSM cases are slightly frontloaded to earlier in the outbreak.",
      "text": "However, HSM cases are observed to be slightly frontloaded to earlier in the outbreak."
    },
    {
      "node_id": "n-00450",
      "strength": 0.725,
      "section": "4.2",
      "quote": "for a Bayes factor of 10000",
      "text": "The Bayes factor in favor of the zoonotic hypothesis is 10000 when restricting Z to HSM and LL to WIV"
    },
    {
      "node_id": "n-00449",
      "strength": 0.5,
      "section": "4.2",
      "quote": "For example, restricting Z to HSM and LL to WIV we get P (H|Z) = 1/1200 P (H|LL) = 1/(1.2\u00b7 107)",
      "text": "Restricting Z to HSM and LL to WIV gives P(H|Z) = 1/1200 and P(H|LL) = 1/(1.2\u00b710^7)"
    },
    {
      "node_id": "n-00318",
      "strength": 0.5,
      "section": "3.2",
      "quote": "Human ACE-2 5 1 .61",
      "text": "The evidence 'Human ACE-2' has a Bayes log-odds of 1.61."
    },
    {
      "node_id": "n-00492",
      "strength": 0.5,
      "section": "4.5",
      "quote": "Since residents at HSM are 0.01% of the population of Wuhan, this is equivalent to saying that residents at HSM have 1000 times the chance of being the index case than the average Wuhan resident.",
      "text": "Residents at HSM have 1000 times the chance of being the index case than the average Wuhan resident."
    },
    {
      "node_id": "n-00488",
      "strength": 0.597,
      "section": "4.5",
      "quote": "The simplistic model used in the previous subsection suggests that the early outbreak being observed at HSM gives a Bayes factor of 100 to 10000 in favor of Z over LL,",
      "text": "The simplistic model used in the previous subsection suggests that the early outbreak being observed at HSM gives a Bayes factor of 100 to 10000 in favor of Z over LL."
    },
    {
      "node_id": "n-01014",
      "strength": 0.5,
      "section": "7.3",
      "quote": "For Z this is simply the west half of HSM",
      "text": "For Z, the smallest epidemiological circle that contains both the primary case and the index case is the west half of HSM."
    },
    {
      "node_id": "n-00528",
      "strength": 0.5,
      "section": "4.5.1",
      "quote": "at 10% ascertainment, this falls to 0.00014% chance.",
      "text": "At 10% ascertainment, the chance that none of the 128 pre-HSM cases are found falls to 0.00014%."
    },
    {
      "node_id": "n-00315",
      "strength": 0.5,
      "section": "3.2",
      "quote": "location of first SSE 13 .48 2 .6",
      "text": "The evidence 'location of first SSE' has a Bayes log-odds of 2.6."
    },
    {
      "node_id": "n-00320",
      "strength": 0.5,
      "section": "3.2",
      "quote": "12 nucleotide insert 50 3 .91",
      "text": "The evidence '12 nucleotide insert' has a Bayes log-odds of 3.91."
    },
    {
      "node_id": "n-01035",
      "strength": 0.595,
      "section": "7.5",
      "quote": "I give these Bayes factors of 20 and 2 respectively; the latter cannot go above 14 as that was the number of species in the study",
      "text": "The judge gives a Bayes factor of 20 to the observation of the strange 12 nucleotide insert creating a furin cleavage site"
    },
    {
      "node_id": "n-00134",
      "strength": 0.5,
      "section": "1.2.2",
      "quote": "The RRAR subsequence constitutes an FCS.",
      "text": "The RRAR subsequence in the amino acid sequence of the reference genome for sars-cov-2, Wuhan-Hu-1, constitutes a furin cleavage site (FCS)."
    },
    {
      "node_id": "n-00436",
      "strength": 0.379,
      "section": "4.2",
      "quote": "for a Bayes factor of 100 to 1000 in favor of zoonotic spillover",
      "text": "The Bayes factor in favor of the zoonotic hypothesis is between 100 and 1000"
    },
    {
      "node_id": "n-00980",
      "strength": 0.342,
      "section": "7.1",
      "quote": "In toto I will assess this as 1 /200.",
      "text": "The author assesses the probability of HSM being the location of a pandemic spillover to be 1/200."
    },
    {
      "node_id": "n-00970",
      "strength": 0.5,
      "section": "7.1",
      "quote": "On this basis I will add another factor of 1/2.",
      "text": "The author adds another factor of 1/2 to the probability of a pandemic due to the uniqueness of sars-cov-2."
    },
    {
      "node_id": "n-01026",
      "strength": 0.202,
      "section": "7.4",
      "quote": "Arguably, this observation has a Bayes factor of 15 in favor of Z",
      "text": "This observation has a Bayes factor of 15 in favor of Z."
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
