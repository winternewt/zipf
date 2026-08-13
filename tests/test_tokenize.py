"""Tokenizer behaviour.

`assets/sample_reply.md` is an authored fixture with the structure of a real reply — fenced
blocks, inline spans, a table, a link, paths. The words that must NOT survive are spelled
`zzq*` so that a failure is unambiguous: those strings exist nowhere else in the project, so
if one appears in the token stream it came from the construct under test and nothing else.
"""

from __future__ import annotations

import pytest

from zipf.paths import ASSETS_DIR
from zipf.tokenize import iter_tokens, normalise, strip_html, strip_markdown, tokenize

FIXTURE = ASSETS_DIR / "sample_reply.md"


@pytest.fixture(scope="module")
def sample() -> str:
    if not FIXTURE.exists():
        pytest.skip(f"fixture {FIXTURE} is missing")
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sample_tokens(sample: str) -> list[str]:
    return tokenize(sample)


def test_fenced_code_is_removed(sample_tokens: list[str]) -> None:
    assert "zzqfenced" not in sample_tokens


def test_inline_code_is_removed(sample_tokens: list[str]) -> None:
    assert "zzqinline" not in sample_tokens


def test_prose_survives_stripping(sample_tokens: list[str]) -> None:
    # Words from ordinary sentences, each outside any code construct.
    for word in ("containers", "gateway", "misleading", "flakiness"):
        assert word in sample_tokens


def test_link_text_survives_but_target_does_not(sample_tokens: list[str]) -> None:
    assert "interface" in sample_tokens and "dump" in sample_tokens
    assert "example" not in sample_tokens
    assert "invalid" not in sample_tokens


def test_paths_and_urls_leave_no_fragments(sample_tokens: list[str]) -> None:
    for fragment in ("var", "syslog", "deb", "debian", "org", "http", "https"):
        assert fragment not in sample_tokens, f"{fragment!r} leaked from a path or URL"


def test_no_orphan_clitics(sample_tokens: list[str]) -> None:
    """F1: stripping `ufw` from ``the `ufw`'s default`` must not leave a bare ``s``."""
    for fragment in ("s", "t", "ll", "d", "m", "ve"):
        assert fragment not in sample_tokens


def test_contractions_stay_whole(sample_tokens: list[str]) -> None:
    for word in ("you're", "it's", "don't", "i'd"):
        assert word in sample_tokens


def test_hyphenated_compounds_stay_whole(sample_tokens: list[str]) -> None:
    """F2: ``re-checking`` must not emit a standalone ``re``."""
    assert "re-checking" in sample_tokens
    assert "well-known" in sample_tokens
    assert "re" not in sample_tokens


def test_single_letters_limited_to_real_words() -> None:
    tokens = tokenize("A thing *b* and c, I said. d e f")
    singles = {t for t in tokens if len(t) == 1}
    assert singles == {"a", "i"}


def test_digits_are_not_vocabulary() -> None:
    assert tokenize("version 3 shipped in 2021 with 42 fixes") == [
        "version",
        "shipped",
        "in",
        "with",
        "fixes",
    ]


def test_unicode_apostrophes_fold_to_ascii() -> None:
    assert tokenize("you’re right") == tokenize("you're right") == ["you're", "right"]


def test_normalise_is_idempotent(sample: str) -> None:
    once = normalise(sample)
    assert normalise(once) == once


def test_tokenizing_is_deterministic(sample: str) -> None:
    assert tokenize(sample) == tokenize(sample)


def test_iter_and_eager_forms_agree(sample: str) -> None:
    assert list(iter_tokens(sample)) == tokenize(sample)


def test_html_preprocessor_removes_code_and_tags() -> None:
    html = "<p>Real prose here.</p><pre><code>zzqhtml = 1</code></pre><p>More &amp; more.</p>"
    tokens = tokenize(html, preprocessor="html")
    assert "zzqhtml" not in tokens
    assert {"real", "prose", "here", "more"} <= set(tokens)


def test_unclosed_fence_does_not_swallow_nothing() -> None:
    """A fence opened and never closed runs to end of document, which is what markdown does."""
    text = "Before the fence.\n\n```python\nzzqunclosed = 1\n"
    tokens = tokenize(text)
    assert "zzqunclosed" not in tokens
    assert "before" in tokens


def test_document_that_is_entirely_code_yields_no_prose() -> None:
    assert tokenize("```\nzzqonly = 1\nprint(zzqonly)\n```\n") == []


def test_empty_document_is_empty_not_an_error() -> None:
    assert tokenize("") == []
    assert tokenize("   \n\n  ") == []


def test_strip_functions_are_pure(sample: str) -> None:
    before = sample
    strip_markdown(sample)
    strip_html(sample)
    assert sample == before


def test_asciidoc_listing_blocks_are_removed() -> None:
    """AsciiDoc fences with four or more hyphens, and literal blocks with dots."""
    text = (
        "Prose before.\n\n[source,console]\n----\n$ git commit -m zzqadoc\nzzqoutput\n----\n\n"
        "Prose between.\n\n....\nzzqliteral\n....\n\nProse after.\n"
    )
    tokens = tokenize(text, preprocessor="asciidoc")
    for fragment in ("zzqadoc", "zzqoutput", "zzqliteral", "source", "console"):
        assert fragment not in tokens
    assert {"prose", "before", "between", "after"} <= set(tokens)


def test_asciidoc_monospace_and_crossrefs_are_removed() -> None:
    tokens = tokenize("See <<ch02_zzqref>> and +zzqmono+ and `zzqtick` here.", preprocessor="asciidoc")
    for fragment in ("zzqref", "zzqmono", "zzqtick", "ch"):
        assert fragment not in tokens
    assert {"see", "and", "here"} <= set(tokens)


def test_markdown_setext_heading_is_not_treated_as_a_code_fence() -> None:
    """The reason AsciiDoc has its own preprocessor.

    A line of hyphens underneath text is a heading in markdown and a listing fence in AsciiDoc.
    If the markdown stripper learned the AsciiDoc rule, every setext heading would open a code
    block and swallow the document to the next one.
    """
    text = "A Heading\n----------\n\nzzqkept prose under the heading.\n"
    assert "zzqkept" in tokenize(text, preprocessor="markdown")
