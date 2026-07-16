"""Parse eric_decision.pdf into section-aware chunks for HippoRAG indexing.

The decision has a real table of contents, so we use it as the authoritative
section tree rather than guessing headings from the body. We then walk the body
in order, locating each heading in sequence, which is robust to the ligature and
spacing damage that pdf text extraction leaves behind.

Chunk ids are md5 of the exact chunk text with a "chunk-" prefix, which is the
same scheme HippoRAG's EmbeddingStore uses. That lets us join this metadata back
onto HippoRAG's graph nodes without threading anything through the library.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import tiktoken
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "eric_decision.pdf"
OUT_DIR = ROOT / "data"

TARGET_TOKENS = 300
MAX_TOKENS = 420
MIN_TOKENS = 40

ENC = tiktoken.get_encoding("cl100k_base")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling module
from span_match import clean, locate  # noqa: E402

# Chunk texts are a deterministic whitespace-collapsed transform of the section's
# own raw text (see pack_chunks/split_sentences below), so a real match should
# score at or near 1.0; this is a safety net against a genuine "cannot find it"
# case (e.g. a heavily word-split monster-sentence chunk), not a calibrated
# fidelity threshold like extract_claims.py's (different noise profile: that one
# tolerates a model's paraphrasing-adjacent copy errors, this one only tolerates
# our own whitespace flattening).
CHUNK_PAGE_MATCH_THRESHOLD = 0.85

TOC_LINE = re.compile(r"^([0-9]+(?:\.[0-9]+)*|[A-E](?:\.[0-9]+)*)\s+(.*?)\s*\.{3,}\s*([0-9]+)\s*$")
TOC_LINE_NODOTS = re.compile(r"^([0-9]+(?:\.[0-9]+)*|[A-E](?:\.[0-9]+)*)\s+(.+?)\s+([0-9]+)\s*$")
PAGE_MARK = re.compile(r"^--- PAGE (\d+) ---$")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


def norm_key(text: str) -> str:
    """Aggressive normalization for matching a TOC title against a body heading."""
    return re.sub(r"[^a-z0-9]", "", clean(text).lower())


def extract_pages() -> list[str]:
    reader = PdfReader(str(PDF))
    return [clean(p.extract_text() or "") for p in reader.pages]


def parse_toc(pages: list[str]) -> list[dict]:
    """Pull the canonical section list out of the table of contents pages."""
    toc: list[dict] = []
    seen: set[str] = set()
    for page in pages[1:4]:  # contents live on pages 2-3
        for raw in page.split("\n"):
            line = raw.strip()
            m = TOC_LINE.match(line) or TOC_LINE_NODOTS.match(line)
            if not m:
                continue
            number, title, page_no = m.group(1), m.group(2).strip(" ."), int(m.group(3))
            if not title or number in seen:
                continue
            seen.add(number)
            toc.append({"number": number, "title": title, "toc_page": page_no})
    return toc


def build_body(pages: list[str], first_body_page: int) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate body pages, dropping page markers and bare page-number lines.

    Returns the body text plus a list of (char_offset, page_number) so any chunk
    can be traced back to the page it came from.
    """
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for idx in range(first_body_page - 1, len(pages)):
        page_no = idx + 1
        lines = []
        for raw in pages[idx].split("\n"):
            line = raw.rstrip()
            # Drop the bare page-number line that sits alone at the page foot.
            if line.strip().isdigit() and len(line.strip()) <= 3:
                continue
            lines.append(line)
        text = "\n".join(lines).strip("\n")
        if not text:
            continue
        offsets.append((cursor, page_no))
        parts.append(text)
        cursor += len(text) + 1
    return "\n".join(parts), offsets


def page_for_offset(offsets: list[tuple[int, int]], pos: int) -> int:
    page = offsets[0][1] if offsets else 1
    for start, page_no in offsets:
        if start <= pos:
            page = page_no
        else:
            break
    return page


def locate_headings(body: str, toc: list[dict]) -> list[dict]:
    """Walk the body in TOC order, finding where each section actually starts.

    Sequential search (rather than one global regex) means a heading that got
    mangled by extraction can be skipped without derailing the sections after it.
    """
    lines = body.split("\n")
    line_offsets: list[int] = []
    cursor = 0
    for line in lines:
        line_offsets.append(cursor)
        cursor += len(line) + 1

    found: list[dict] = []
    search_from = 0
    for entry in toc:
        want_num = entry["number"]
        want_title = norm_key(entry["title"])
        hit = None
        for i in range(search_from, len(lines)):
            line = lines[i].strip()
            if not line.startswith(want_num):
                continue
            rest = line[len(want_num):].strip()
            rk = norm_key(rest)
            # Heading line is "<number> <title>", possibly with the title wrapped
            # onto the next line, so accept a prefix match in either direction.
            if rk and (rk == want_title or want_title.startswith(rk) or rk.startswith(want_title)):
                hit = i
                break
        if hit is None:
            continue
        found.append({**entry, "line": hit, "start": line_offsets[hit] + len(lines[hit]) + 1})
        search_from = hit + 1

    for i, sec in enumerate(found):
        sec["end"] = found[i + 1]["line"] and line_offsets[found[i + 1]["line"]] if i + 1 < len(found) else len(body)
    return found


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Split on sentence enders followed by whitespace + a capital, quote or digit.
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[]|\d)", text.replace("\n", " "))
    return [p.strip() for p in pieces if p.strip()]


def pack_chunks(text: str) -> list[str]:
    """Greedily pack sentences to ~TARGET_TOKENS, never crossing a section."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        st = n_tokens(sent)
        # A single monster sentence (tables, long quotes) becomes its own chunk.
        if st > MAX_TOKENS:
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_tokens = [], 0
            words = sent.split()
            step = max(1, len(words) * TARGET_TOKENS // max(st, 1))
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + step]))
            continue
        if cur_tokens + st > TARGET_TOKENS and cur_tokens >= MIN_TOKENS:
            chunks.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(sent)
        cur_tokens += st
    if cur:
        tail = " ".join(cur)
        # Fold a runt tail into the previous chunk rather than emitting a stub.
        if n_tokens(tail) < MIN_TOKENS and chunks:
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)
    return chunks


def main() -> None:
    pages = extract_pages()
    toc = parse_toc(pages)
    print(f"parsed {len(toc)} sections from table of contents")

    body, offsets = build_body(pages, first_body_page=4)
    sections = locate_headings(body, toc)
    print(f"located {len(sections)}/{len(toc)} section headings in the body")

    missing = {t["number"] for t in toc} - {s["number"] for s in sections}
    if missing:
        print(f"  WARNING unlocated sections: {sorted(missing)}")

    chunks = []
    page_match_tiers = {"exact": 0, "fuzzy": 0, "reject": 0}
    for sec in sections:
        sub = body[sec["start"]:sec["end"]]
        # raw = sub.strip() shifts the start offset by however much leading
        # whitespace .strip() removed; track that so per-chunk offsets below
        # convert back to real `body` coordinates correctly.
        lead = len(sub) - len(sub.lstrip())
        raw_start_in_body = sec["start"] + lead
        raw = sub.strip()
        if not raw:
            continue
        number = sec["number"]
        parts = number.split(".")
        parent = ".".join(parts[:-1]) if len(parts) > 1 else None
        search_from = 0  # chunks appear in raw in order; narrows the search window
        for order, text in enumerate(pack_chunks(raw)):
            if n_tokens(text) < 12:  # drop figure captions / stray fragments
                continue
            # pack_chunks() joins sentences with " " and flattens newlines, so
            # `text` is no longer a literal substring of `raw` -- the same
            # problem extract_claims.py solves for LLM-returned quotes. Reuse
            # that matcher (via span_match) rather than assuming every chunk in
            # a section starts at the section's own start offset, which is what
            # this used to do and is wrong for every chunk after the first in a
            # multi-chunk section.
            r = locate(text, raw[search_from:], CHUNK_PAGE_MATCH_THRESHOLD)
            page_match_tiers[r["tier"]] += 1
            if r["tier"] == "reject":
                # Rare (a heavily word-split monster-sentence chunk might miss
                # threshold); fall back to the section start rather than
                # crashing, and it's counted above so a regression is visible.
                pos = raw_start_in_body
            else:
                pos = raw_start_in_body + search_from + r["start"]
                search_from += r["end"]
            chunks.append({
                "chunk_id": "chunk-" + hashlib.md5(text.encode()).hexdigest(),
                "text": text,
                "section_number": number,
                "section_title": sec["title"],
                "section_depth": len(parts),
                "parent_section": parent,
                "is_appendix": number[0].isalpha(),
                "chunk_order_in_section": order,
                "page": page_for_offset(offsets, pos),
                "n_tokens": n_tokens(text),
            })

    print(f"  chunk->page offsets: {page_match_tiers['exact']} exact, "
          f"{page_match_tiers['fuzzy']} fuzzy, {page_match_tiers['reject']} "
          f"rejected (fell back to section start)")

    # A duplicate chunk text would collide on chunk_id and silently merge in the
    # graph, so surface it rather than letting it pass.
    ids = [c["chunk_id"] for c in chunks]
    dupes = len(ids) - len(set(ids))
    if dupes:
        print(f"  WARNING {dupes} duplicate chunk texts (will merge into one graph node)")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2))
    (OUT_DIR / "sections.json").write_text(json.dumps(
        [{k: s[k] for k in ("number", "title", "toc_page")} for s in sections], indent=2))

    tot = sum(c["n_tokens"] for c in chunks)
    print(f"\nwrote {len(chunks)} chunks, {tot:,} tokens total")
    print(f"  mean {tot // max(len(chunks),1)} tok/chunk, "
          f"max {max((c['n_tokens'] for c in chunks), default=0)}")
    print(f"  -> {OUT_DIR/'chunks.json'}")


if __name__ == "__main__":
    main()
