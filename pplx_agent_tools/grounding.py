"""Grounding check: do an answer's figures and names appear in its cited sources?

Pure and local. The evidence is only what the ask stream carries per source
(title + snippet); pages are never fetched, so a figure that is on the cited
page but outside its snippet counts as unsupported. The verdict is a signal
for the caller, not a gate: it never changes the exit code.

Checkable terms:

- Figures: numerals normalized to a value, so "$1.2M", "1.2 million",
  "1,200,000" and "1200000" are one figure, and "4.0" matches "4". An answer
  figure matches evidence that equals it at the coarser of the two
  precisions ("1.2 million" and "1,234,567" support each other).
- Names: runs of two or more capitalized words ("Wine Spectator"). Single
  capitalized words are skipped, as are sentence-initial words, headings,
  bold labels, table headers and names made only of the query's own words.

Terms that also occur in the query are not checked: echoing the question is
not a claim.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

from .verbs._ask_common import Source

# At or below one in five terms supported, the answer counts as ungrounded.
# Snippets are ~200-character excerpts, so a genuine answer routinely carries
# terms absent from them (live answers measured 31-52% supported) and a
# stricter bar would flag real answers; but a single incidental overlap must
# not clear an otherwise unsupported five-term row of figures.
MIN_SUPPORTED_FRACTION = Decimal("0.2")

REASON_NO_TERMS = "no checkable figures or names"
REASON_NO_SOURCES = "no sources"
REASON_SITE_ROOTS = "every cited URL is a site root"

_SCALES = {
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "m": Decimal(1_000_000),
    "mn": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "bn": Decimal(1_000_000_000),
    "billion": Decimal(1_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
}

# Single-letter scales only when attached ("5k", "$1.2M"): "100 m" is meters.
_NUMBER_RE = re.compile(
    r"(?:(?P<cur>[$€£¥])\s?|(?<![\w.$€£¥]))"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<word>(?i:thousand|million|billion|trillion))\b|(?P<abbr>bn|mn|[kKMB])(?![\w]))?"
    r"(?P<pct>\s?%|\s?per\s?cent\b)?"
    r"(?![\w])"
)
_CITATION_RE = re.compile(
    r"\[(?:\^|web:)?\d+(?:\s*[,\u2013-]\s*\d+)*\]|\u3010\d+(?:[^\u3011\n]{0,40})?\u3011"
)
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_CODE_RE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_BOLD_LABEL_RE = re.compile(r"\*\*[^*\n]+?\*\*\s*:|\*\*[^*\n]+?:\*\*")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
# Clause boundaries; a name never spans one.
_CLAUSE_SPLIT_RE = re.compile(r"([.!?;:|()\[\]\"“”,\n]|\s[-\u2013\u2014]\s)")
_SENTENCE_END = {".", "!", "?", ":", "\n", "|"}
_CAPITALIZED_RE = re.compile(r"^(?:[A-Z][\w'\u2019.&-]*|[A-Z]{2,}\d*)$")
_NAME_CONNECTORS = {"of", "de", "di", "da", "del", "della", "der", "van", "von", "la", "le", "du"}
# Capitalized words that are not names (function words, calendar words).
# fmt: off
_NOT_NAME_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "i", "we", "you", "he", "she", "they", "our", "your", "their",
    "in", "on", "at", "for", "from", "by", "with", "as", "if", "but", "and", "or", "not", "no", "yes", "to", "of",
    "however", "also", "although", "while", "when", "where", "which", "who", "what", "why", "how",
    "according", "based", "note", "overall", "summary", "total", "average", "approximately", "about",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
# fmt: on


@dataclass(frozen=True)
class _Figure:
    text: str
    value: Decimal
    # Half the unit of the last written digit: the rounding window.
    tolerance: Decimal
    # A plain integer with no currency, scale, percent or decimals.
    bare: bool


@dataclass(frozen=True)
class Grounding:
    """`grounded` is None when the answer has nothing to check."""

    grounded: bool | None
    reasons: list[str] = field(default_factory=list)
    # Checkable terms found in no source's title or snippet, as written.
    unsupported: list[str] = field(default_factory=list)
    checked: int = 0


def _parse_figures(text: str) -> list[_Figure]:
    out: list[_Figure] = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group("num").replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        scale_word = m.group("word") or m.group("abbr")
        scale = _SCALES[scale_word.lower()] if scale_word else Decimal(1)
        unit = Decimal(1).scaleb(-decimals) * scale
        bare = not (m.group("cur") or scale_word or m.group("pct") or decimals)
        out.append(_Figure(m.group(0).strip(), value * scale, unit / 2, bare))
    return out


def _is_checkable_figure(fig: _Figure) -> bool:
    """Bare small integers ("3 reasons", "2 options") are counts, not claims."""
    return not (fig.bare and fig.value <= 10)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped.casefold())


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _fold(text)))


def _clean_markdown(answer: str) -> str:
    text = _CODE_RE.sub(" ", answer)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _CITATION_RE.sub(" ", text)
    return _LIST_MARKER_RE.sub("", text)


def _name_lines(text: str) -> list[str]:
    """Lines eligible for name extraction: headings and table headers are
    title-cased labels, not claims."""
    lines = text.split("\n")
    keep: list[str] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") or _TABLE_SEPARATOR_RE.match(line):
            continue
        if i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
            continue
        keep.append(_BOLD_LABEL_RE.sub(":", line).replace("*", "").replace("_", " "))
    return keep


def _names_in_clause(tokens: list[str], sentence_start: bool) -> list[str]:
    names: list[str] = []
    run: list[str] = []
    run_start = 0

    def flush() -> None:
        while run and run[-1].lower() in _NAME_CONNECTORS:
            run.pop()
        if run_start == 0 and sentence_start and run:
            run.pop(0)
        if sum(1 for t in run if t.lower() not in _NAME_CONNECTORS) >= 2:
            names.append(" ".join(run))

    for i, tok in enumerate(tokens):
        word = re.sub(r"['\u2019]s$", "", tok)
        is_cap = bool(_CAPITALIZED_RE.match(word)) and word.lower() not in _NOT_NAME_WORDS
        if is_cap or (run and word.lower() in _NAME_CONNECTORS):
            if not run:
                run_start = i
            run.append(word)
            continue
        flush()
        run = []
    flush()
    return names


def _extract_names(text: str) -> list[str]:
    names: list[str] = []
    for line in _name_lines(text):
        sentence_start = True
        for part in _CLAUSE_SPLIT_RE.split(line):
            if _CLAUSE_SPLIT_RE.fullmatch(part):
                sentence_start = part.strip() in _SENTENCE_END or part == "\n"
                continue
            tokens = part.split()
            if tokens:
                names.extend(_names_in_clause(tokens, sentence_start))
                sentence_start = False
    return names


def _is_site_root(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:  # e.g. an unclosed IPv6 bracket
        return False
    return parts.path in ("", "/") and not parts.query


def _figure_supported(fig: _Figure, evidence: list[_Figure]) -> bool:
    """Compare at the coarser of the two precisions, so "$269.91 billion"
    matches a source's "$269.9B". A bare integer in the evidence keeps the
    answer's precision: "14" in a snippet must not support "14.3"."""
    return any(
        abs(fig.value - e.value) <= (fig.tolerance if e.bare else max(fig.tolerance, e.tolerance))
        for e in evidence
    )


def _name_supported(name: str, evidence_text: str, evidence_words: set[str]) -> bool:
    """A name is supported when it appears verbatim, or when all its words do
    (sources often write "Spectator, Wine" or split a name across a title)."""
    folded = _fold(name)
    if folded in evidence_text:
        return True
    return _words(name) <= evidence_words


def check_grounding(answer: str, query: str, sources: Sequence[Source]) -> Grounding:
    """Verdict on whether `answer`'s figures and names appear in `sources`."""
    text = _clean_markdown(answer)
    query_values = {f.value for f in _parse_figures(query)}
    query_words = _words(query)

    figures: dict[Decimal, _Figure] = {}
    for fig in _parse_figures(text):
        if fig.value not in query_values and _is_checkable_figure(fig):
            figures.setdefault(fig.value, fig)
    names: dict[str, str] = {}
    for name in _extract_names(text):
        if not _words(name) <= query_words:
            names.setdefault(_fold(name), name)

    checked = len(figures) + len(names)
    if checked == 0:
        return Grounding(grounded=None, reasons=[REASON_NO_TERMS])
    all_terms = [f.text for f in figures.values()] + list(names.values())
    if not sources:
        return Grounding(False, [REASON_NO_SOURCES], all_terms, checked)

    evidence = " \n ".join(f"{s.title or ''} \n {s.snippet or ''}" for s in sources)
    # Small counts ("Top 4 wines") are everywhere in titles and would
    # support figures like "4.0" by accident.
    evidence_figures = [f for f in _parse_figures(evidence) if _is_checkable_figure(f)]
    evidence_text = _fold(evidence)
    evidence_words = _words(evidence)
    unsupported = [f.text for f in figures.values() if not _figure_supported(f, evidence_figures)]
    unsupported += [
        n for n in names.values() if not _name_supported(n, evidence_text, evidence_words)
    ]

    reasons: list[str] = []
    if all(_is_site_root(s.url) for s in sources):
        reasons.append(REASON_SITE_ROOTS)
    supported = checked - len(unsupported)
    if Decimal(supported) <= MIN_SUPPORTED_FRACTION * checked:
        reasons.append(
            f"{supported} of {checked} figures/names appear in a cited source's title or snippet"
        )
    return Grounding(not reasons, reasons, unsupported, checked)
