"""pplx snippets verb: batched query-relevant excerpts from N URLs.

Pure-local implementation (Perplexity's web-session API has no snippets-per-URL
endpoint; same finding as Step 6 for pplx fetch). Architecture:

  1. Concurrent-fetch each URL via curl_cffi + trafilatura (same path as
     pplx fetch's plain mode).
  2. Paragraph-split each page, embed paragraphs with fastembed (ONNX, no
     PyTorch), insert into SQLite (FTS5 for BM25 + sqlite-vec for vector KNN).
  3. For each URL, retrieve top-K candidates from both indices, merge via
     Reciprocal Rank Fusion (RRF), trim to per-URL and total token budgets.

The two retrieval signals complement each other: BM25 catches exact-term hits,
vector similarity catches paraphrasing. RRF (k=60) is the standard fusion —
rank-only, no score normalization needed.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..errors import NetworkError, SchemaError
from ..wire import Client

DEFAULT_MAX_TOKENS = 4000
DEFAULT_MAX_TOKENS_PER_PAGE = 1500
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, ~80MB
DEFAULT_K_PER_URL = 8  # candidates pulled from each index before RRF merge
RRF_K = 60  # standard RRF constant

# Approx tokens-per-word; LLM tokenizers vary, this is good enough for budgets.
_TOKEN_PER_WORD = 1.3


@dataclass
class Snippet:
    text: str
    score: float  # RRF score, higher = more relevant
    tokens: int


@dataclass
class UrlSnippets:
    url: str
    snippets: list[Snippet] = field(default_factory=list)
    error: str | None = None


@dataclass
class SnippetsResult:
    query: str
    results: list[UrlSnippets]
    # Aggregate, free-form messages (e.g. "fetched 3/5 URLs successfully").
    # Per-URL fetch/index errors live on each UrlSnippets.error instead.
    warnings: list[str] = field(default_factory=list)


def snippets(
    query: str,
    urls: list[str],
    *,
    client: Client | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_tokens_per_page: int = DEFAULT_MAX_TOKENS_PER_PAGE,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> SnippetsResult:
    """Fetch N URLs concurrently and extract query-relevant paragraphs from each.

    Returns SnippetsResult with one UrlSnippets per input URL (errors recorded
    on the per-URL object rather than raised, so one bad URL doesn't kill the
    whole call).

    `client` is accepted for API symmetry with `search` / `fetch` (both of
    which require a Client) and reserved for a future authenticated mode.
    The current implementation is pure-local — fetch_page uses a fresh
    curl_cffi session with no cookies — so passing None is the default and
    has the same behavior as passing a Client today.
    """
    _ = client  # currently unused; see docstring
    if not urls:
        return SnippetsResult(query=query, results=[])

    # Dedupe input while preserving order. Duplicates would (a) double-count
    # the same paragraphs in the FTS/vec index, skewing rankings, and (b)
    # share a single UrlSnippets object across the dup'd output slots,
    # causing snippets to appear N times.
    urls = list(dict.fromkeys(urls))

    pages = _fetch_all(urls)

    # Split → embed → index. Paragraphs from all URLs into one DB, with URL as a tag.
    rows: list[tuple[str, str, int]] = []  # (url, paragraph_text, word_count)
    for url, content, err in pages:
        if err is not None or not content:
            continue
        for para in _split_paragraphs(content):
            words = len(para.split())
            if words < 3:
                continue
            rows.append((url, para, words))

    if not rows:
        return SnippetsResult(
            query=query,
            results=[
                UrlSnippets(url=url, error=err or "no extractable paragraphs")
                for url, _, err in pages
            ],
        )

    # Generate embeddings in one batch (amortizes model cold-start across all rows).
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise SchemaError(f"fastembed is required for pplx snippets: {e}") from e

    embedder = TextEmbedding(model_name=embed_model)
    texts = [r[1] for r in rows]
    paragraph_vecs = list(embedder.embed(texts))
    query_vec = next(iter(embedder.query_embed([query])))
    dim = len(query_vec)

    conn = _build_index(rows, paragraph_vecs, dim)
    try:
        # Tokenize the FTS5 query once. Strip special chars to avoid syntax errors.
        fts_query = _fts5_escape(query)
        query_blob = _vec_to_blob(query_vec)

        by_url: dict[str, UrlSnippets] = {url: UrlSnippets(url=url) for url, *_ in pages}
        for url, _, err in pages:
            if err is not None:
                by_url[url].error = err
                continue

        total_tokens = 0
        for url in urls:
            if by_url[url].error is not None:
                continue
            budget = min(max_tokens_per_page, max_tokens - total_tokens)
            if budget <= 0:
                break
            ranked = _hybrid_retrieve(conn, fts_query, query_blob, url, k=DEFAULT_K_PER_URL)
            for para_text, score, words in ranked:
                tokens = int(words * _TOKEN_PER_WORD)
                if tokens > budget:
                    continue
                by_url[url].snippets.append(Snippet(text=para_text, score=score, tokens=tokens))
                budget -= tokens
                total_tokens += tokens
                if budget <= 0:
                    break

        return SnippetsResult(query=query, results=[by_url[url] for url in urls])
    finally:
        conn.close()


def _fetch_all(urls: list[str]) -> list[tuple[str, str, str | None]]:
    """Concurrent-fetch + extract content. Returns one row per URL preserving
    input order: (url, content_or_empty, error_or_None).

    Concurrency model: parallelize *across hosts*, serialize *within* a host,
    and reuse one curl_cffi Session per host so same-host URLs share a TCP
    connection (and benefit from HTTP/2 multiplexing when the server supports
    it). A naive `ThreadPoolExecutor(max_workers=6)` would fire 6 simultaneous
    requests at the same origin when the caller passes 6 URLs from one
    site — some hosts will rate-limit that, the user gets NetworkError per
    URL, and the verb under-delivers on otherwise valid input. Grouping
    by host lets us stay polite without losing cross-host parallelism, and
    sharing a Session per group makes the polite path also fast: 6 same-host
    URLs cost 1 handshake instead of 6.
    """
    from collections import defaultdict
    from urllib.parse import urlparse

    from curl_cffi import requests as cf_requests

    from ..verbs.fetch import fetch_page

    def one(
        url: str,
        session: cf_requests.Session[cf_requests.Response] | None = None,
    ) -> tuple[str, str, str | None]:
        try:
            result = fetch_page(url, domain="", max_chars=None, session=session)
        except NetworkError as e:
            return (url, "", str(e))
        except Exception as e:
            # any fetch failure is captured on the row, not raised
            return (url, "", f"fetch failed: {e}")
        return (url, result.content or "", None)

    # Group URLs by host, preserving each URL's original index so output
    # order matches input order regardless of which host group finishes first.
    by_host: dict[str, list[int]] = defaultdict(list)
    for i, url in enumerate(urls):
        host = urlparse(url).netloc.lower() or url
        by_host[host].append(i)

    results: list[tuple[str, str, str | None] | None] = [None] * len(urls)

    def fetch_host_serially(indices: list[int]) -> None:
        # One Session per host group: TCP reuse + HTTP/2 multiplexing when
        # the server offers it. Session closes after the group completes.
        with cf_requests.Session(impersonate="chrome") as session:
            for i in indices:
                results[i] = one(urls[i], session=session)

    max_workers = min(len(by_host), 6)
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # pool.map preserves group order; per-group serial means same-host
        # URLs never overlap. Inner `one` swallows exceptions, so the
        # iterator never raises.
        list(pool.map(fetch_host_serially, by_host.values()))
    finally:
        # cancel_futures=True drops any unstarted work so Ctrl-C / outer
        # exceptions don't block on already-queued URLs.
        pool.shutdown(wait=True, cancel_futures=True)

    # All indices were written by the host-serial pass; the cast is safe.
    return [r for r in results if r is not None]


def _split_paragraphs(content: str) -> list[str]:
    """Split markdown-ish content into paragraphs. Long paragraphs are not
    further split — FTS5 and embeddings both handle longer chunks fine, and
    keeping context together is better for retrieval quality.
    """
    blocks = re.split(r"\n\s*\n+", content.strip())
    out: list[str] = []
    for b in blocks:
        s = b.strip()
        if s:
            out.append(s)
    return out


def _build_index(
    rows: list[tuple[str, str, int]],
    vecs: list[Any],
    dim: int,
) -> sqlite3.Connection:
    """Build an in-memory SQLite database with FTS5 (BM25) + sqlite-vec (KNN)
    side-by-side. Both share the same rowid space so we can join by row.
    """
    import sqlite_vec

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        "CREATE TABLE paragraphs (id INTEGER PRIMARY KEY, url TEXT, text TEXT, words INTEGER)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE p_fts USING fts5(text, content='paragraphs', content_rowid='id')"
    )
    conn.execute(f"CREATE VIRTUAL TABLE p_vec USING vec0(embedding float[{dim}])")

    for i, ((url, text, words), vec) in enumerate(zip(rows, vecs, strict=True), start=1):
        conn.execute(
            "INSERT INTO paragraphs(id, url, text, words) VALUES (?, ?, ?, ?)",
            (i, url, text, words),
        )
        conn.execute("INSERT INTO p_fts(rowid, text) VALUES (?, ?)", (i, text))
        conn.execute(
            "INSERT INTO p_vec(rowid, embedding) VALUES (?, ?)",
            (i, _vec_to_blob(vec)),
        )
    return conn


def _hybrid_retrieve(
    conn: sqlite3.Connection,
    fts_query: str,
    query_blob: bytes,
    url: str,
    *,
    k: int,
) -> list[tuple[str, float, int]]:
    """RRF-merge BM25 (FTS5) and vector (sqlite-vec) ranks for one URL.

    Returns [(paragraph_text, rrf_score, word_count), ...] ordered by score desc.
    """
    # FTS5: text matching, scope to one URL via JOIN. The LIMIT is applied
    # after the URL filter, so we always get up to k URL-matching rows.
    bm25_rows = conn.execute(
        """
        SELECT p.id, p.text, p.words
        FROM p_fts JOIN paragraphs p ON p_fts.rowid = p.id
        WHERE p_fts MATCH ? AND p.url = ?
        ORDER BY bm25(p_fts)
        LIMIT ?
        """,
        (fts_query, url, k),
    ).fetchall()

    # Vector KNN: sqlite-vec MATCH is a GLOBAL top-K — it picks the k
    # nearest rows in the whole corpus, THEN the JOIN filters by URL. If
    # the URL is sparse (few rows) and those rows aren't in the global
    # top-k, we'd silently get zero vector hits for this URL. Scale k by
    # the ratio of total_rows / url_rows so the global KNN over-fetches
    # enough to leave ~k URL-matching rows after the join.
    total_rows, url_rows = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM paragraphs),
            (SELECT COUNT(*) FROM paragraphs WHERE url = ?)
        """,
        (url,),
    ).fetchone()
    if url_rows == 0 or total_rows == 0:
        vec_rows: list[tuple[int, str, int]] = []
    else:
        scaled_k = min(total_rows, k * max(1, total_rows // url_rows))
        vec_rows = conn.execute(
            """
            SELECT p.id, p.text, p.words
            FROM p_vec JOIN paragraphs p ON p_vec.rowid = p.id
            WHERE p_vec.embedding MATCH ? AND p.url = ? AND k = ?
            ORDER BY distance
            """,
            (query_blob, url, scaled_k),
        ).fetchall()

    # RRF merge by row id. score(d) = sum(1/(RRF_K + rank_i(d)))
    rrf: dict[int, float] = {}
    text_words: dict[int, tuple[str, int]] = {}
    for rank, (rid, text, words) in enumerate(bm25_rows, start=1):
        rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (RRF_K + rank)
        text_words[rid] = (text, words)
    for rank, (rid, text, words) in enumerate(vec_rows, start=1):
        rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (RRF_K + rank)
        text_words.setdefault(rid, (text, words))

    merged = [
        (text_words[rid][0], score, text_words[rid][1])
        for rid, score in sorted(rrf.items(), key=lambda kv: -kv[1])
    ]
    return merged


def _fts5_escape(query: str) -> str:
    """Strip FTS5 syntax chars and wrap each term as a phrase so user input
    like `: ; ( ) " * .` doesn't break the parse.
    """
    tokens = re.findall(r"[\w][\w'-]*", query)
    if not tokens:
        return '""'
    # Each token as a double-quoted phrase; OR-join them so any-term-matches.
    return " OR ".join(f'"{t}"' for t in tokens)


def _vec_to_blob(vec: Any) -> bytes:
    """sqlite-vec expects little-endian float32 bytes."""
    return struct.pack(f"<{len(vec)}f", *vec)
