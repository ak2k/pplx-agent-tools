"""Unit tests for grounding.py — figure normalization, term extraction, verdicts."""

from __future__ import annotations

from decimal import Decimal

import pytest

from pplx_agent_tools.grounding import (
    REASON_NO_SOURCES,
    REASON_NO_TERMS,
    REASON_SITE_ROOTS,
    _clean_markdown,
    _extract_names,
    _figure_supported,
    _parse_figures,
    check_grounding,
)
from pplx_agent_tools.verbs._ask_common import Source

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
    ],
)
def test_format_variants_match(answer: str, evidence: str) -> None:
    [fig] = _parse_figures(answer)
    assert _figure_supported(fig, _parse_figures(evidence))


@pytest.mark.parametrize(
    ("answer", "evidence"),
    [
        # A more precise answer figure is not supported by a rounded source.
        ("1,234,567", "1.2 million"),
        ("4.3", "4"),
        ("$55", "$65"),
    ],
)
def test_distinct_figures_do_not_match(answer: str, evidence: str) -> None:
    [fig] = _parse_figures(answer)
    assert not _figure_supported(fig, _parse_figures(evidence))


def test_non_figures_are_not_parsed() -> None:
    assert _parse_figures("the 21st century, an A320, 3D printing") == []


def test_citation_markers_and_list_numbers_are_not_figures() -> None:
    text = _clean_markdown("1. First point [1][2]\n2. Second point [3, 4] [^5]")
    assert _parse_figures(text) == []


# ---------- name extraction ----------


def test_names_skip_sentence_initial_and_function_words() -> None:
    text = "The critics agree. Wine Spectator gave it 100 points, and the Wine Advocate agreed."
    # "Wine Spectator" opens a sentence, so only "Spectator" would remain: not a name.
    assert _extract_names(text) == ["Wine Advocate"]


def test_names_skip_headings_bold_labels_and_table_headers() -> None:
    text = _clean_markdown(
        "## Key Findings Summary\n"
        "**Critic Score:** high\n"
        "| Critic Name | Point Score |\n"
        "|---|---|\n"
        "| x | reviewed by James Suckling |\n"
    )
    assert _extract_names(text) == ["James Suckling"]


def test_names_keep_connectors_inside_only() -> None:
    assert _extract_names("It is sold by the Bank of America in Italy.") == ["Bank of America"]


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
    assert g.grounded is False
    assert REASON_SITE_ROOTS in g.reasons
    assert any(r.startswith("0 of 5 ") for r in g.reasons)
    assert sorted(g.unsupported) == sorted(["4.0", "1,234", "$55", "100", "Wine Spectator"])
    assert g.checked == 5


def test_fabricated_row_is_ungrounded_even_with_deep_links() -> None:
    sources = [Source("https://www.vivino.com/explore?q=caparzo", "Explore wines", "Find wine.")]
    g = check_grounding(_CAPARZO_ANSWER, _CAPARZO_QUERY, sources)
    assert g.grounded is False
    assert REASON_SITE_ROOTS not in g.reasons


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
    assert g.grounded is True
    assert g.reasons == []
    assert g.unsupported == []
    assert g.checked == 2


def test_site_root_citations_alone_make_an_answer_ungrounded() -> None:
    answer = "The Eiffel Tower is 330 meters tall [1]."
    sources = [Source("https://www.toureiffel.paris/", "Eiffel Tower", "The tower is 330 m tall.")]
    g = check_grounding(answer, "how tall is it", sources)
    assert g.grounded is False
    assert g.reasons == [REASON_SITE_ROOTS]
    assert g.unsupported == []


def test_no_sources_with_checkable_terms_is_ungrounded() -> None:
    g = check_grounding("It sold 4,500 units in 2023.", "sales", [])
    assert g.grounded is False
    assert g.reasons == [REASON_NO_SOURCES]
    assert g.unsupported == ["4,500", "2023"]


def test_no_checkable_terms_is_null_not_true() -> None:
    g = check_grounding("Yes, it is dynamically typed [1].", "is python dynamic?", [])
    assert g.grounded is None
    assert g.reasons == [REASON_NO_TERMS]
    assert g.checked == 0


def test_query_terms_and_small_counts_are_not_checked() -> None:
    answer = "There are 3 reasons the Caparzo 2019 scores well [1]."
    g = check_grounding(answer, "Caparzo 2019", [Source("https://x.test/a", "t", "s")])
    assert g.grounded is None


def test_low_support_fraction_threshold() -> None:
    answer = "Figures: 101, 202, 303, 404 and 505."
    one_of_five = [Source("https://x.test/a", "t", "only 101 here")]
    assert check_grounding(answer, "q", one_of_five).grounded is True
    answer_six = answer + " Also 606."
    assert check_grounding(answer_six, "q", one_of_five).grounded is False
