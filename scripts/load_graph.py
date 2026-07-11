"""Load the published HippoRAG 2 graph and join it to the document hierarchy.

This is the reference example for working with `artifacts/hipporag2/`. It needs
only networkx — no HippoRAG install, no API keys.

    python scripts/load_graph.py
"""

import json
from collections import Counter
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "hipporag2"
DATA = ROOT / "data"


def load() -> tuple[nx.Graph, dict]:
    """Return the KG as a networkx graph, with section metadata already attached.

    Passage nodes carry the section fields from data/chunks.json; entity nodes
    carry their surface text. Every node has `content` and `type`.
    """
    nodes = json.loads((ART / "kg_nodes.json").read_text())
    edges = json.loads((ART / "kg_edges.json").read_text())
    chunks = {c["chunk_id"]: c for c in json.loads((DATA / "chunks.json").read_text())}

    graph = nx.Graph()
    for n in nodes:
        meta = chunks.get(n["id"], {})
        graph.add_node(
            n["id"],
            type=n["type"],
            content=n["content"],
            section_number=meta.get("section_number"),
            section_title=meta.get("section_title"),
            page=meta.get("page"),
        )
    for e in edges:
        graph.add_edge(e["source"], e["target"], weight=e["weight"])

    return graph, chunks


def main() -> None:
    graph, chunks = load()
    facts = json.loads((ART / "kg_facts.json").read_text())

    entities = [n for n, d in graph.nodes(data=True) if d["type"] == "entity"]
    passages = [n for n, d in graph.nodes(data=True) if d["type"] == "passage"]

    n_raw = len(json.loads((ART / "kg_edges.json").read_text()))
    print(f"nodes {graph.number_of_nodes():,}  "
          f"(entities {len(entities):,}, passages {len(passages):,})")
    # kg_edges.json stores each undirected edge once per direction; networkx
    # collapses them, so the two counts legitimately differ.
    print(f"edges {graph.number_of_edges():,} undirected "
          f"({n_raw:,} directed entries in kg_edges.json)   facts {len(facts):,}")

    # Every passage node should have joined against a chunk.
    unjoined = [p for p in passages if graph.nodes[p]["section_number"] is None]
    print(f"passages without section metadata: {len(unjoined)}")

    print("\nmost connected entities:")
    for node, deg in sorted(graph.degree(entities), key=lambda x: -x[1])[:10]:
        print(f"  {deg:4d}  {graph.nodes[node]['content']}")

    print("\npassages per top-level section:")
    # Titles come from sections.json: a chunk in §5.1 carries the *subsection*
    # title, so reading top-level titles off chunks would mislabel them.
    sections = json.loads((DATA / "sections.json").read_text())
    top_title = {s["number"]: s["title"] for s in sections if "." not in s["number"]}
    tops = Counter(str(c["section_number"]).split(".")[0] for c in chunks.values())
    for sec, n in sorted(tops.items()):
        print(f"  §{sec:<2} {n:3d} chunks   {top_title.get(sec, '?')}")

    # Walk from an entity to the passages that mention it — the basic retrieval
    # move HippoRAG makes, minus the PPR scoring.
    target = "huanan seafood market"
    hit = next((n for n in entities if graph.nodes[n]["content"] == target), None)
    if hit:
        print(f"\npassages linked to entity {target!r}:")
        linked = [nb for nb in graph.neighbors(hit) if graph.nodes[nb]["type"] == "passage"]
        for p in linked[:5]:
            d = graph.nodes[p]
            print(f"  §{d['section_number']} p{d['page']}  {d['content'][:70]}...")
        print(f"  ({len(linked)} passages total)")


if __name__ == "__main__":
    main()
