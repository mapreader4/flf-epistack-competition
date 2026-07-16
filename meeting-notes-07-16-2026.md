# Meeting notes — 2026-07-16

1. Improving claim extraction
2. Adding evidence attribution to extraction to claim extraction. And possibly other entities later, like presumed facts(these can be other facts in the graph as well. useful later for graph construction)

## Possible continuation paths for structure

3. Working on the graph:
    1. starting from claims and sections as nodes in a simple graph.
    2. discovering possible hierarchical structures.
    3. focuse on structure useful for human interaction.

More concretely, some ideas we can try:

4. Use different node types for different things, not just claims — evidence, experiments/studies, datasets, hypotheses, and sources can each be their own kind of node. Then link them by how they relate: a piece of evidence supports a hypothesis, one claim contradicts another, an experiment produces a data point, two claims give a rival number for the same quantity. Example: "the furin site roughly doubles the odds of lab-leak" is a claim node with a "supports" link to the lab-leak hypothesis, and the study behind it is a separate experiment node. (Right now the graph's nodes are just things like "HSM" or "the virus", and the claim itself is lost.)

5. Decide the graph's shape by trying a couple and comparing — the choices that matter:
    1. links: typed and directed (A supports B, A contradicts B), instead of the current single unlabeled link that drops the relation.
    2. flat vs layered: one big graph, or a few layers over the same claims (document sections, sub-questions, hypothesis → evidence).
    3. how a claim is identified: a stable id plus a separate text-fingerprint for catching duplicates, instead of naming a claim by a hash of its text (which changes the moment the wording changes, and gives the same claim two ids when it is said two ways).

6. Find ways to find relevant information/claims. Build those links without reading all 83 pages at once: for each claim, pull a few claims that might be related (they mention the same thing, or answer the same sub-question), then ask the model about just those two at a time — are they the same claim? does one support or contradict the other? Every model call stays small.

7. Group claims by the sub-question they answer, e.g. "did the outbreak start at the market?", "was the furin site engineered?", "would a lab leak have stayed secret?"; so the argument can be read by question instead of by page. When a document has no table of contents (eggs, black holes), build these groups from the claims themselves.

8. Work out how believable each claim is from its links: adding a rebuttal automatically lowers the claim it attacks. For the few claims that carry real numbers (~8%, the Bayes factors), add them up the way the judge did; leave the rest as a plain map of what supports what.

9. Once the structure exists, flag what a human cares about: the crux (the one claim that, if it flipped, would change the verdict), evidence that looks independent but traces back to a single source, and questions that still have no evidence.

10. Evaluation: read the first outputs by hand before deciding how to score them; every claim and link points back to the exact sentence it came from; later, try to break it — flip a key claim, or feed in a duplicate, and check the graph reacts.

11. Add the other test cases. The competition names three case studies — COVID origins (our eric_decision), the LHC black-hole risk, and egg health — and judges across all of them. Get source material for the black-hole and egg cases and run the same pipeline (claims → graph) on each, so we test generality and don't tune everything to one document.
