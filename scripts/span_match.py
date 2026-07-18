"""Shared tiered exact-then-fuzzy string matcher: locate a (possibly reformatted)
needle inside a haystack and return original-coordinate offsets.

Used by two problems in this pipeline that are actually the same problem --
reconciling a text fragment against a longer original text after some
whitespace-destroying transformation happened in between:

  - chunk_decision.py: pack_chunks() joins sentences with " " and flattens
    newlines, so a chunk's own text is no longer a literal substring of the
    section's raw text -- but the page a chunk lands on is looked up by finding
    the chunk's real start offset in that raw text.
  - extraction-variants/extract_claims_original.py (and its siblings): an
    LLM-returned "quote" is supposed to be verbatim, but minor
    whitespace/hyphenation/footnote-splicing noise from PDF extraction means
    straight substring search can still miss it.

Both need: exact substring match first, then a whitespace/case-normalized exact
match, then a rapidfuzz windowed fuzzy match as a last resort -- never fabricate
an offset, and always report which tier a match came from so callers can log or
threshold on it. No other module in this repo should reimplement this search.

This module has no dependency on chunk_decision.py or anything under
extraction-variants/ (they depend on it), so there is no import cycle:
chunk_decision -> span_match, and extraction-variants/* -> {chunk_decision, span_match}.
"""

import re
import unicodedata

# Ligatures and typographic characters that pdf extraction leaves in the text.
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}


def clean(text: str) -> str:
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    return unicodedata.normalize("NFKC", text)


_WS = re.compile(r"\s+")


def _norm_needle(s: str) -> str:
    """Full normalization for the search pattern (no offset map needed)."""
    return _WS.sub(" ", clean(s).lower()).strip()


def _norm_haystack(s: str) -> tuple[str, list[int]]:
    """Lowercase + collapse whitespace, keeping a map from each normalized char
    back to the original char index in s (so matches map back to real offsets).

    We do NOT run clean() on the haystack: callers pass text that is already
    ligature/NFKC-normalized (chunk_decision.py's build_body() runs clean() on
    every page up front, and clean() is idempotent), and re-cleaning here could
    shift the offset map if it weren't.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            for c in ch.lower():
                out.append(c)
                idx.append(i)
            prev_space = False
    return "".join(out), idx


def locate(needle: str, haystack: str, threshold: float) -> dict:
    """Find `needle` in `haystack`, returning original-coordinate offsets.

    Tiers: (1) exact substring; (2) whitespace/case-normalized exact substring
    (still scored 1.0, tier "exact" -- it is only a formatting difference);
    (3) windowed fuzzy match via rapidfuzz.partial_ratio_alignment.

    Returns a dict with tier in {"exact", "fuzzy", "reject"}. "reject" means the
    best score fell below `threshold`; start/end are absent in that case.
    """
    if not needle or not haystack:
        return {"tier": "reject", "score": 0.0}

    i = haystack.find(needle)
    if i >= 0:
        return {"start": i, "end": i + len(needle), "score": 1.0, "tier": "exact"}

    nn = _norm_needle(needle)
    if not nn:
        return {"tier": "reject", "score": 0.0}
    nh, idx = _norm_haystack(haystack)
    if not nh:
        return {"tier": "reject", "score": 0.0}

    p = nh.find(nn)
    if p >= 0:
        start = idx[p]
        end = idx[p + len(nn) - 1] + 1
        return {"start": start, "end": end, "score": 1.0, "tier": "exact"}

    from rapidfuzz.fuzz import partial_ratio_alignment

    al = partial_ratio_alignment(nn, nh)
    if al is None or al.dest_end <= al.dest_start:
        return {"tier": "reject", "score": 0.0}
    score = al.score / 100.0
    if score < threshold:
        return {"tier": "reject", "score": round(score, 4)}
    start = idx[al.dest_start]
    end = idx[min(al.dest_end, len(idx)) - 1] + 1
    return {"start": start, "end": end, "score": round(score, 4), "tier": "fuzzy"}
