"""Unit tests for grounding.py — figure normalization, term extraction, verdicts."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pplx_agent_tools import grounding
from pplx_agent_tools.grounding import (
    Grounded,
    Grounding,
    Unchecked,
    Ungrounded,
    _clean_markdown,
    _extract_names,
    _figure_supported,
    _parse_figures,
    check_grounding,
)
from pplx_agent_tools.verbs._ask_common import Source


def _unsupported(g: Grounding) -> list[str]:
    return [] if isinstance(g, Unchecked) else list(g.ungrounded_terms)


# ---------- figure normalization ----------


@pytest.mark.parametrize("text", ["$1.2M", "1.2 million", "1,200,000", "1200000"])
def test_million_variants_share_a_value(text: str) -> None:
    [fig] = _parse_figures(text)
    assert fig.value == Decimal(1_200_000)


def test_lowercase_m_is_not_a_scale() -> None:
    assert _parse_figures("100m") == []
    assert [f.value for f in _parse_figures("100 m")] == [Decimal(100)]


@pytest.mark.parametrize(
    ("answer", "evidence"),
    [
        ("$1.2M", "1,200,000"),
        ("1,200,000", "$1.2M"),
        ("1200000", "1.2 million"),
        ("1.2 million", "1,234,567"),
        ("4.0", "rated 4 stars"),
        ("4", "rated 4.0 stars"),
        ("12%", "12 percent"),
        ("12 percent", "12 %"),
        ("$55", "55 USD"),
        ("5k", "5,000 reviews"),
        ("$3.5 billion", "3.5bn"),
        ("$254.5 billion", "Costco Reports $254.5 Billion Revenue"),
        ("1.2 million", "1.2 Million"),
        # A more precise answer matches a source rounded to its own precision.
        ("$269.91 billion", "$269.9B"),
        ("11,115,000", "11.1 million"),
        ("1,234,567", "1.2 million"),
    ],
)
def test_format_variants_match(answer: str, evidence: str) -> None:
    [fig] = _parse_figures(answer)
    assert _figure_supported(fig, _parse_figures(evidence))


@pytest.mark.parametrize(
    ("answer", "evidence"),
    [
        # A bare integer in a source keeps the answer's precision.
        ("4.3", "4"),
        ("14.3", "14"),
        ("$55", "$65"),
        ("$269.91 billion", "$268.9B"),
    ],
)
def test_distinct_figures_do_not_match(answer: str, evidence: str) -> None:
    [fig] = _parse_figures(answer)
    assert not _figure_supported(fig, _parse_figures(evidence))


def test_non_figures_are_not_parsed() -> None:
    assert _parse_figures("the 21st century, an A320, 3D printing") == []


def test_citation_markers_and_list_numbers_are_not_figures() -> None:
    text = _clean_markdown(
        "1. First point [1][2]\n2. Second point [3, 4] [^5] [web:6] \u30107\u3011 \u30108\u2020L1-L4\u3011"
    )
    assert _parse_figures(text) == []


def test_sign_is_part_of_the_value() -> None:
    [neg] = _parse_figures("a change of -5.0%")
    assert neg.value == Decimal(-5)
    assert not _figure_supported(neg, _parse_figures("up 5.0%"))
    assert _figure_supported(neg, _parse_figures("down \u22125.0% on the year"))
    [pos] = _parse_figures("up 5.0%")
    assert not _figure_supported(pos, _parse_figures("a change of -5.0%"))
    assert [f.value for f in _parse_figures("from 2019-2020, 95-100 points")] == [
        Decimal(2019),
        Decimal(2020),
        Decimal(95),
        Decimal(100),
    ]


def test_negative_bare_integers_are_checked() -> None:
    answer = "Net flows were -2,500, -7,300 and -9,100 in the quarter."
    g = check_grounding(answer, "q", [Source("https://x.test/a", "t", "flows were steady")])
    assert isinstance(g, Ungrounded)
    assert g.ungrounded_terms == ("-2,500", "-7,300", "-9,100")
    g = check_grounding("The balance moved by -3 overnight.", "q", [])
    assert isinstance(g, Ungrounded)
    assert g.checked_terms == ("-3",)


def test_absurdly_long_numerals_are_ignored() -> None:
    assert _parse_figures("Revenue was " + "9" * 1_000_001 + " units.") == []
    g = check_grounding("Revenue was " + "9" * 1_000_001 + " units.", "q", [Source("u", "t", "s")])
    assert g == Unchecked("no_checkable_terms")


# ---------- name extraction ----------


def _names(text: str) -> list[str]:
    return [name for name, _ in _extract_names(text)]


def test_names_skip_function_words() -> None:
    text = "The critics agree. Wine Spectator gave it 100 points, and the Wine Advocate agreed."
    assert _names(text) == ["Wine Spectator", "Wine Advocate"]


def test_sentence_initial_name_is_checked_with_a_fallback() -> None:
    assert _extract_names("Acme Labs sold 100 units.") == [("Acme Labs", None)]
    assert _extract_names("Critic James Suckling gave it 97.") == [
        ("Critic James Suckling", "James Suckling")
    ]
    assert _extract_names("It went to Critic James Suckling.") == [("Critic James Suckling", None)]


def test_names_skip_headings_bold_labels_and_table_headers() -> None:
    text = _clean_markdown(
        "## Key Findings Summary\n"
        "**Critic Score:** high\n"
        "| Critic Name | Point Score |\n"
        "|---|---|\n"
        "| x | reviewed by James Suckling |\n"
    )
    assert _names(text) == ["James Suckling"]


def test_names_keep_connectors_inside_only() -> None:
    assert _names("It is sold by the Bank of America in Italy.") == ["Bank of America"]


# ---------- name support ----------


def _name_verdict(answer: str, snippet: str) -> list[str]:
    return _unsupported(check_grounding(answer, "q", [Source("https://x.test/a", "t", snippet)]))


def test_name_words_must_be_adjacent() -> None:
    assert _name_verdict("It is sponsored by Red Bull here.", "red wine and a bull market") == [
        "Red Bull"
    ]
    for snippet in ("per Wine-Spectator", "WINE\nSPECTATOR", "the Wine  Spectator's list"):
        assert _name_verdict("It is rated by Wine Spectator here.", snippet) == []


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])
def test_possessive_inside_a_name_matches_a_verbatim_quote(apostrophe: str) -> None:
    answer = f"Last week Moody{apostrophe}s Investors Service downgraded the bond."
    for snippet in (
        "Moody's Investors Service downgraded the bond",
        "Moody\u2019s Investors Service downgraded the bond",
    ):
        assert _name_verdict(answer, snippet) == []
    assert _name_verdict(answer, "Moody Investors Service downgraded the bond") == []
    assert _name_verdict("It was rated by Wine Spectator's panel.", "the Wine Spectator list") == []


def test_name_does_not_match_across_sources() -> None:
    sources = [
        Source("https://x.test/a", "red", "ends with Red"),
        Source("https://x.test/b", "Bull", "s"),
    ]
    g = check_grounding("It is sponsored by Red Bull here.", "q", sources)
    assert _unsupported(g) == ["Red Bull"]


def test_sentence_initial_name_supported_by_its_fallback() -> None:
    assert _name_verdict("Critic James Suckling gave it 97.", "James Suckling: 97 points") == []
    assert _name_verdict("Acme Labs sold 100 units.", "Labs sold 100 units") == ["Acme Labs"]


# ---------- verdicts ----------

_CAPARZO_ANSWER = """## Caparzo Brunello di Montalcino 2019

| Wine | Rating | Ratings | Price | Critic |
|---|---|---|---|---|
| Caparzo Brunello 2019 | 4.0 | 1,234 ratings | $55 | Wine Spectator 100 points |

The Caparzo 2019 is rated 4.0 on Vivino by 1,234 users [1][2]. It costs about $55
and the Wine Spectator gave it 100 points [1]."""
_CAPARZO_QUERY = (
    "Caparzo Brunello di Montalcino 2019: Vivino rating, rating count, price, critic score"
)


def test_fabricated_row_cited_to_landing_pages_is_ungrounded() -> None:
    sources = [
        Source(
            "https://www.vivino.com/",
            "Vivino: Buy the Right Wine",
            "Discover and buy the right wine with the world's largest wine marketplace.",
        ),
        Source("https://www.vivino.com", "Vivino", "Wine app and marketplace"),
    ]
    g = check_grounding(_CAPARZO_ANSWER, _CAPARZO_QUERY, sources)
    assert isinstance(g, Ungrounded)
    assert g.reasons == ("site_roots", "low_support")
    assert sorted(g.ungrounded_terms) == sorted(
        ["2019", "4.0", "1,234", "$55", "100", "Wine Spectator"]
    )
    assert len(g.checked_terms) == 6


def test_fabricated_row_is_ungrounded_even_with_deep_links() -> None:
    sources = [Source("https://www.vivino.com/explore?q=caparzo", "Explore wines", "Find wine.")]
    g = check_grounding(_CAPARZO_ANSWER, _CAPARZO_QUERY, sources)
    assert isinstance(g, Ungrounded)
    assert "site_roots" not in g.reasons


def test_figure_in_snippet_in_another_format_is_grounded() -> None:
    answer = "Tokyo's metropolitan population is about 14.0 million people, per the Tokyo Metropolitan Government [1]."
    sources = [
        Source(
            "https://www.metro.tokyo.lg.jp/english/about/population",
            "Population of Tokyo - Tokyo Metropolitan Government",
            "As of 2024 the population of Tokyo was 14,047,594.",
        )
    ]
    g = check_grounding(answer, "What is the population of Tokyo?", sources)
    assert isinstance(g, Grounded)
    assert g.ungrounded_terms == ()
    assert len(g.checked_terms) == 2


def test_site_root_citations_alone_make_an_answer_ungrounded() -> None:
    answer = "The Eiffel Tower is 330 meters tall [1]."
    sources = [Source("https://www.toureiffel.paris/", "Eiffel Tower", "The tower is 330 m tall.")]
    g = check_grounding(answer, "how tall is it", sources)
    assert isinstance(g, Ungrounded)
    assert g.reasons == ("site_roots",)
    assert g.ungrounded_terms == ()


def test_no_sources_with_checkable_terms_is_ungrounded() -> None:
    g = check_grounding("It sold 4,500 units in 2023.", "sales", [])
    assert isinstance(g, Ungrounded)
    assert g.reasons == ("no_sources",)
    assert g.ungrounded_terms == ("4,500", "2023")


def test_no_checkable_terms_is_null_not_true() -> None:
    g = check_grounding("Yes, it is dynamically typed [1].", "is python dynamic?", [])
    assert g == Unchecked("no_checkable_terms")


def test_query_names_and_small_counts_are_not_checked() -> None:
    answer = "There are 3 reasons the Caparzo Brunello scores well [1]."
    g = check_grounding(answer, "Caparzo Brunello", [Source("https://x.test/a", "t", "s")])
    assert g == Unchecked("no_checkable_terms")


def test_figures_restated_from_the_query_are_checked() -> None:
    query, answer = "Was it $45 in 2024?", "Yes, it was $45 in 2024."
    g = check_grounding(answer, query, [])
    assert isinstance(g, Ungrounded)
    assert g.reasons == ("no_sources",)
    assert g.checked_terms == ("$45", "2024")
    g = check_grounding(answer, query, [Source("https://x.test/a", "Price", "It cost $45.")])
    assert isinstance(g, Grounded)
    assert g.ungrounded_terms == ("2024",)


def test_low_support_fraction_threshold() -> None:
    answer = "Figures: 101, 202, 303, 404 and 505."
    one_of_five = [Source("https://x.test/a", "t", "only 101 here")]
    g = check_grounding(answer, "q", one_of_five)
    assert isinstance(g, Ungrounded)
    assert g.reasons == ("low_support",)
    assert len(g.ungrounded_terms) == 4
    two_of_five = [Source("https://x.test/a", "t", "only 101 and 202 here")]
    assert isinstance(check_grounding(answer, "q", two_of_five), Grounded)


def _figures_answer(n: int) -> str:
    return "Figures: " + ", ".join(str(1000 + 7 * i) for i in range(n)) + "."


def _snippet_with(n: int) -> list[Source]:
    return [Source("https://x.test/a", "t", " ".join(str(1000 + 7 * i) for i in range(n)))]


def test_three_supported_terms_ground_a_long_answer() -> None:
    """Short snippets cannot hold every figure of a long table, so several
    confirmed terms ground it even below one in five."""
    g = check_grounding(_figures_answer(50), "q", _snippet_with(8))
    assert isinstance(g, Grounded)
    assert len(g.checked_terms) == 50
    assert len(g.ungrounded_terms) == 42
    assert isinstance(check_grounding(_figures_answer(50), "q", _snippet_with(3)), Grounded)


def test_two_supported_terms_do_not_ground_a_long_answer() -> None:
    g = check_grounding(_figures_answer(50), "q", _snippet_with(2))
    assert isinstance(g, Ungrounded)
    assert g.reasons == ("low_support",)
    assert len(g.ungrounded_terms) == 48


def test_one_incidental_small_number_does_not_ground_a_fabricated_row() -> None:
    sources = [Source("https://www.vivino.com/toplists/x", "Top 4 wines", "Buy wine online")]
    g = check_grounding(_CAPARZO_ANSWER, _CAPARZO_QUERY, sources)
    assert isinstance(g, Ungrounded)
    assert "4.0" in g.ungrounded_terms


@pytest.mark.parametrize("url", ["http://[::1", "https://[bad/", "::::", "http://[::1]:99999/x"])
def test_malformed_source_urls_do_not_raise(url: str) -> None:
    g = check_grounding("It costs $45 [1].", "price", [Source(url, "t", "$45")])
    assert isinstance(g, Grounded)


# Fragments that exercise the parsers' edge cases more often than random text.
_PIECES = [
    *"0123456789.,$%kKMB€£ \n|#*_-[]()^:;\"'`",
    "\u2019",
    "\u3010",
    "\u3011",
    "\u0661",
    "\U0001d7cf",
    "Wine ",
    "Of ",
    "million ",
    "Billion ",
    "**Label:** ",
    "|---|",
    "[1]",
]
_digit_run = st.integers(25, 400).map(lambda n: "9" * n)
_text = st.lists(st.one_of(st.sampled_from(_PIECES), _digit_run), max_size=120).map("".join)
# Past ~1e6 digits a scaled Decimal overflows rather than rounds.
_huge = _text.map(lambda t: f"{t}{'9' * 1_000_001} {t}")


@settings(max_examples=300, deadline=None)
@given(
    answer=st.one_of(st.text(max_size=300), _text, _huge),
    query=st.text(max_size=80),
    sources=st.lists(
        st.builds(
            Source,
            url=st.one_of(st.text(max_size=40), st.just("http://[::1")),
            title=st.one_of(st.none(), st.text(max_size=80), _text),
            snippet=st.one_of(st.none(), st.text(max_size=200), _text, _huge),
        ),
        max_size=4,
    ),
)
def test_check_grounding_is_total(answer: str, query: str, sources: list[Source]) -> None:
    assert isinstance(check_grounding(answer, query, sources), (Grounded, Ungrounded, Unchecked))


def test_only_one_function_constructs_a_term() -> None:
    """`Term` is a NewType, so the checker cannot stop a stray `Term(...)`;
    this keeps its one constructor the only call site in the package."""
    package = Path(grounding.__file__).parent
    calls = [
        (path.name, line.strip())
        for path in sorted(package.rglob("*.py"))
        for line in path.read_text().splitlines()
        if re.search(r"(?<![\w.])Term\(", line)
    ]
    assert calls == [("grounding.py", "return Term(text)")]


def test_verdicts_reject_contradictory_fields() -> None:
    """The variants cannot hold a combination their tag contradicts."""
    g = check_grounding("Figures: 101, 202, 303, 404 and 505.", "q", [])
    assert isinstance(g, Ungrounded)
    terms = g.checked_terms
    for bad in (
        lambda: Grounded(terms, terms),
        lambda: Grounded((), ()),
        lambda: Grounded(terms, (terms[0], terms[0])),
        lambda: Ungrounded((), terms, terms),
        lambda: Ungrounded(("low_support", "low_support"), terms, terms),
        lambda: Ungrounded(("low_support",), terms, ()),
        lambda: Ungrounded(("site_roots",), terms, terms),
        lambda: Ungrounded(("no_sources",), terms, terms[:1]),
        lambda: Ungrounded(("no_sources", "site_roots"), terms, terms),
    ):
        with pytest.raises(ValueError):
            bad()
    no_sources = check_grounding(_figures_answer(50), "q", [])
    assert isinstance(no_sources, Ungrounded)
    long = no_sources.checked_terms
    for bad in (
        # 3 of 50 supported: under one in five but at the absolute floor.
        lambda: Ungrounded(("low_support",), long, long[3:]),
        lambda: Grounded(long, long[2:]),
    ):
        with pytest.raises(ValueError):
            bad()
    assert Grounded(long, long[3:]).tag == "grounded"
    assert Ungrounded(("low_support",), long, long[2:]).tag == "ungrounded"
    assert Grounded(terms, terms[:1]).tag == "grounded"
    assert Ungrounded(("site_roots",), terms, ()).tag == "ungrounded"
    assert Unchecked("disabled").tag == "unchecked"
