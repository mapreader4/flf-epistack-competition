"""Phase 1 + 2: run HippoRAG 2 OpenIE over the decision chunks and build its KG.

Backends. OpenIE (NER + triple extraction) runs on Together. Embeddings run on
Together too by default, or on OpenAI if OPENAI_API_KEY is set and EMBED_MODEL
is an OpenAI model.

Two things upstream does not support that we work around here:

1. HippoRAG builds *both* its LLM client and its embedding client as
   `OpenAI(...)` with no explicit api_key, so both read the same OPENAI_API_KEY
   env var. Two providers cannot share it. When the backends differ we let the
   embedding client take the env var and repoint the LLM client at Together with
   its own key. OpenIE holds a reference to the same llm_model object, so the one
   swap covers both.

2. `_get_embedding_model_class` only recognizes a fixed set of model names and
   asserts on anything else, so Together's `BAAI/...` embedding models are
   rejected. They speak the OpenAI embeddings protocol, so we monkeypatch the
   lookup to route them to OpenAIEmbeddingModel with Together as the base url.

Usage:
    python scripts/run_hipporag_index.py --limit 6     # smoke test
    python scripts/run_hipporag_index.py               # full index
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hippo" / "HippoRAG"))

load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_LLM = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
DEFAULT_EMBED = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# Together embedding models speak the OpenAI embeddings protocol.
TOGETHER_EMBED_PREFIXES = ("BAAI/", "intfloat/", "togethercomputer/")


def patch_embedding_lookup() -> None:
    """Let HippoRAG accept Together's embedding models instead of asserting on them."""
    import src.hipporag.HippoRAG as hippo_mod
    from src.hipporag.embedding_model import _get_embedding_model_class
    from src.hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel

    def lookup(embedding_model_name: str = ""):
        if embedding_model_name.startswith(TOGETHER_EMBED_PREFIXES):
            return OpenAIEmbeddingModel
        return _get_embedding_model_class(embedding_model_name)

    hippo_mod._get_embedding_model_class = lookup


def build_hipporag(save_dir: str, llm_name: str, embed_name: str, synonymy_threshold: float):
    patch_embedding_lookup()
    # Import the class from the module, not the package: importing the submodule
    # in patch_embedding_lookup() binds it as a package attribute, which shadows
    # the lazy `__getattr__` export of the class in src/hipporag/__init__.py.
    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig

    together_key = os.environ.get("TOGETHER_API_KEY")
    if not together_key:
        sys.exit("TOGETHER_API_KEY missing. Put it in .env")

    openai_key = os.environ.get("OPENAI_API_KEY") or ""
    embed_on_together = embed_name.startswith(TOGETHER_EMBED_PREFIXES)

    if not embed_on_together and not openai_key:
        sys.exit(f"EMBED_MODEL={embed_name} needs OPENAI_API_KEY. Put it in .env, "
                 f"or set EMBED_MODEL=intfloat/multilingual-e5-large-instruct to stay on Together.")

    # The embedding client reads OPENAI_API_KEY at construction, whichever
    # provider it is actually pointed at.
    if embed_on_together:
        os.environ["OPENAI_API_KEY"] = together_key

    config = BaseConfig(synonymy_edge_sim_threshold=synonymy_threshold, save_openie=True)

    hipporag = HippoRAG(
        global_config=config,
        save_dir=save_dir,
        llm_model_name=llm_name,
        llm_base_url=TOGETHER_BASE_URL,
        embedding_model_name=embed_name,
        embedding_base_url=TOGETHER_BASE_URL if embed_on_together else None,
    )

    # If embeddings went to OpenAI, the env var belongs to OpenAI, so the LLM
    # client needs its Together credential injected explicitly.
    if not embed_on_together:
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=16)
        hipporag.llm_model.openai_client = OpenAI(
            base_url=TOGETHER_BASE_URL,
            api_key=together_key,
            http_client=httpx.Client(limits=limits, timeout=httpx.Timeout(300, read=300)),
            max_retries=3,
        )

    return hipporag, ("Together" if embed_on_together else "OpenAI")


def preflight(hipporag) -> None:
    """Together rejects some params OpenAI accepts; fail here, not 195 chunks in."""
    msg, meta, _ = hipporag.llm_model.infer(
        messages=[{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}]
    )
    print(f"  preflight LLM ok -> {msg.strip()[:60]!r} "
          f"({meta.get('completion_tokens')} completion tokens)")

    vec = hipporag.embedding_model.batch_encode(["preflight embedding check"])
    print(f"  preflight embeddings ok -> dim {vec.shape[-1]}")


def export_graph(hipporag, out_dir: Path) -> dict:
    """Dump the igraph KG plus node surface text into plain JSON for later layers."""
    import pandas as pd

    graph = hipporag.graph
    working = Path(hipporag.working_dir)

    def store_parquet(namespace: str) -> Path:
        return working / f"{namespace}_embeddings" / f"vdb_{namespace}.parquet"

    content: dict[str, str] = {}
    for namespace in ("entity", "chunk"):
        parquet = store_parquet(namespace)
        if parquet.exists():
            df = pd.read_parquet(parquet)
            for hash_id, text in zip(df["hash_id"], df["content"]):
                content[hash_id] = text

    nodes = [{
        "id": v["name"],
        "type": "passage" if v["name"].startswith("chunk-") else "entity",
        "content": content.get(v["name"], ""),
    } for v in graph.vs]

    edges = [{
        "source": graph.vs[e.source]["name"],
        "target": graph.vs[e.target]["name"],
        "weight": e["weight"] if "weight" in e.attributes() else 1.0,
    } for e in graph.es]

    facts = []
    fact_parquet = store_parquet("fact")
    if fact_parquet.exists():
        df = pd.read_parquet(fact_parquet)
        facts = [{"id": h, "triple": c} for h, c in zip(df["hash_id"], df["content"])]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kg_nodes.json").write_text(json.dumps(nodes, indent=2))
    (out_dir / "kg_edges.json").write_text(json.dumps(edges, indent=2))
    (out_dir / "kg_facts.json").write_text(json.dumps(facts, indent=2))
    graph.write_graphml(str(out_dir / "kg.graphml"))

    node_type = {n["id"]: n["type"] for n in nodes}
    n_entities = sum(t == "entity" for t in node_type.values())
    # Entity-entity edges are fact edges plus synonymy edges; entity-passage
    # edges come from the passage pass. Splitting them makes runaway synonymy
    # density visible instead of hiding inside one edge count.
    ent_ent = sum(node_type[e["source"]] == "entity" and node_type[e["target"]] == "entity"
                  for e in edges)
    stats = {
        "nodes": len(nodes),
        "entities": n_entities,
        "passages": sum(t == "passage" for t in node_type.values()),
        "edges": len(edges),
        "entity_entity_edges": ent_ent,
        "entity_passage_edges": len(edges) - ent_ent,
        "facts": len(facts),
        "entity_graph_density": round(ent_ent / max(n_entities * (n_entities - 1), 1), 4),
    }
    (out_dir / "kg_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="index only the first N chunks")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--save-dir", default=None)
    # HippoRAG defaults this to 0.8, which is tuned for GritLM/OpenAI embeddings.
    # e5's cosine floor across unrelated entities is ~0.77 and its median ~0.84,
    # so 0.8 marks ~94% of all entity pairs as synonyms and collapses the graph
    # into a near-clique. Genuine synonyms here sit at >=0.96.
    ap.add_argument("--synonymy-threshold", type=float, default=0.95)
    args = ap.parse_args()

    chunks = json.loads((ROOT / "data" / "chunks.json").read_text())
    if args.limit:
        chunks = chunks[: args.limit]
    docs = [c["text"] for c in chunks]

    save_dir = args.save_dir or str(ROOT / "outputs" / ("smoke" if args.limit else "hipporag"))

    hipporag, embed_provider = build_hipporag(
        save_dir, args.llm, args.embed, args.synonymy_threshold)
    print(f"LLM (OpenIE):  {args.llm} @ Together")
    print(f"Embeddings:    {args.embed} @ {embed_provider}")
    print(f"Synonymy sim:  >= {args.synonymy_threshold}")
    print(f"Documents:     {len(docs)} chunks")
    print(f"Save dir:      {save_dir}\n")

    preflight(hipporag)

    print("\n=== Phase 1+2: OpenIE extraction and graph construction ===")
    hipporag.index(docs=docs)

    out_dir = Path(save_dir) / "graph"
    stats = export_graph(hipporag, out_dir)

    print("\n=== Knowledge graph built ===")
    for k, v in stats.items():
        print(f"  {k:>10}: {v:,}")
    print(f"\n  exported -> {out_dir}")


if __name__ == "__main__":
    main()
