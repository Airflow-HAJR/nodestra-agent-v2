"""
Retrieval over per-POI write-ups (menus, dietary info, hours, vibe).

The map graph knows where a place is; it knows nothing about what's on the
menu or whether the meat is halal. This module is the other half: a small
corpus of prose about each POI (`data/poi_knowledge.json`), embedded once and
searched by cosine similarity, so the agent answers questions about a venue
from retrieved text rather than from whatever it happens to believe about a
restaurant chain.

  * search       — top-k chunks for a free-text query, optionally scoped to
                   specific POIs (that scoping is what "check each of these
                   three places for halal" is built on)
  * summarize_poi — the one-line overview for a POI, for listing options

Degrades gracefully. If no embedding model is configured or the embedding call
fails, retrieval falls back to keyword overlap scoring over the same corpus —
noticeably worse, never broken.

The embedding index is cached to `data/poi_index.json` and keyed by a hash of
the corpus plus the model name, so editing the knowledge file re-embeds
automatically and an unchanged one costs nothing at startup.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CORPUS_PATH = _DATA_DIR / "poi_knowledge.json"
_INDEX_PATH = _DATA_DIR / "poi_index.json"

_lock = threading.Lock()
_docs: Optional[list[dict]] = None
_vectors: Optional[list[list[float]]] = None  # parallel to _docs; None when embeddings are unavailable


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_docs() -> list[dict]:
    """Every chunk in the corpus: [{poi_id, name, section, text}, ...]."""
    global _docs
    if _docs is not None:
        return _docs
    try:
        raw = json.loads(_CORPUS_PATH.read_text())
        _docs = [d for d in raw.get("docs", []) if d.get("text")]
    except Exception:
        logger.exception("poi_rag: could not read %s", _CORPUS_PATH)
        _docs = []
    return _docs


def _corpus_fingerprint(docs: list[dict], model: str) -> str:
    h = hashlib.sha256(model.encode())
    for d in docs:
        h.update(d["text"].encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def _build_or_load_vectors(docs: list[dict]) -> Optional[list[list[float]]]:
    """Embed the corpus, reusing the on-disk cache when it matches."""
    from agent.llm import build_embeddings, embedding_model_name

    model = embedding_model_name()
    if not model:
        return None
    fingerprint = _corpus_fingerprint(docs, model)

    try:
        cached = json.loads(_INDEX_PATH.read_text())
        if cached.get("fingerprint") == fingerprint and len(cached.get("vectors", [])) == len(docs):
            return cached["vectors"]
    except Exception:
        pass  # no cache yet, or it's stale/corrupt — rebuild below

    embeddings = build_embeddings()
    if embeddings is None:
        return None
    print(f"[poi_rag] embedding {len(docs)} POI documents with {model}...")
    vectors = embeddings.embed_documents([d["text"] for d in docs])
    try:
        _INDEX_PATH.write_text(json.dumps({"fingerprint": fingerprint, "model": model, "vectors": vectors}))
    except Exception:
        logger.exception("poi_rag: could not write index cache")
    return vectors


def ensure_index() -> tuple[list[dict], Optional[list[list[float]]]]:
    """Load the corpus and (once) its embeddings. Safe to call on every query."""
    global _vectors
    docs = load_docs()
    if not docs:
        return [], None
    with _lock:
        if _vectors is None:
            try:
                _vectors = _build_or_load_vectors(docs) or []
            except Exception:
                logger.exception("poi_rag: embedding failed — falling back to keyword search")
                _vectors = []
    return docs, (_vectors or None)


def warm_index() -> None:
    """Build the embedding index ahead of the first query, so no user turn pays
    for it. Called at server startup; safe to skip."""
    try:
        ensure_index()
    except Exception:
        logger.exception("poi_rag: warm_index failed")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


_WORD_RE = re.compile(r"[a-z0-9']+")


def _keyword_score(query: str, text: str) -> float:
    """Fallback when there's no embedding model: overlap of query terms with the
    document, normalized by query length."""
    q = set(_WORD_RE.findall(query.lower()))
    if not q:
        return 0.0
    t = set(_WORD_RE.findall(text.lower()))
    return len(q & t) / len(q)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(
    query: str,
    poi_ids: Optional[list[str]] = None,
    top_k: int = 5,
    per_poi: bool = False,
) -> list[dict]:
    """Retrieve chunks relevant to `query`.

    poi_ids  — restrict the search to these POIs. This is what makes a
               "does each of these three places serve halal food?" question a
               retrieval per venue rather than one blurred search.
    per_poi  — return the best `top_k` chunks *for each* POI in poi_ids,
               instead of `top_k` overall. Guarantees every candidate venue is
               represented in the evidence, so the agent can't answer for two
               of three places and quietly drop the third.
    """
    docs, vectors = ensure_index()
    if not docs:
        return []

    idxs = range(len(docs))
    if poi_ids:
        wanted = {p.strip() for p in poi_ids if p and p.strip()}
        idxs = [i for i in idxs if docs[i]["poi_id"] in wanted]
    if not idxs:
        return []

    if vectors:
        from agent.llm import build_embeddings
        try:
            qvec = build_embeddings().embed_query(query)  # type: ignore[union-attr]
            scored = [(i, _cosine(qvec, vectors[i])) for i in idxs]
        except Exception:
            logger.exception("poi_rag: query embedding failed — keyword fallback")
            scored = [(i, _keyword_score(query, docs[i]["text"])) for i in idxs]
    else:
        scored = [(i, _keyword_score(query, docs[i]["text"])) for i in idxs]

    def _hit(i: int, score: float) -> dict:
        return {**docs[i], "score": round(score, 4)}

    if per_poi and poi_ids:
        by_poi: dict[str, list[tuple[int, float]]] = {}
        for i, s in scored:
            by_poi.setdefault(docs[i]["poi_id"], []).append((i, s))
        out: list[dict] = []
        # Iterate poi_ids, not by_poi, so the evidence comes back in the order
        # the caller asked about the venues.
        for pid in poi_ids:
            hits = sorted(by_poi.get(pid, []), key=lambda t: t[1], reverse=True)[:top_k]
            out.extend(_hit(i, s) for i, s in hits)
        return out

    scored.sort(key=lambda t: t[1], reverse=True)
    return [_hit(i, s) for i, s in scored[:top_k]]


def summarize_poi(poi_id: str) -> Optional[str]:
    """The POI's 'overview' chunk, or its first chunk if it has no overview."""
    docs = load_docs()
    chunks = [d for d in docs if d["poi_id"] == poi_id]
    if not chunks:
        return None
    for d in chunks:
        if d.get("section") == "overview":
            return d["text"]
    return chunks[0]["text"]


def known_poi_ids() -> set[str]:
    return {d["poi_id"] for d in load_docs()}
