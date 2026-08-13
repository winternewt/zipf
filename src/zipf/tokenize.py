"""The single tokenizer. Every corpus goes through this, without exception.

A baseline tokenized differently from the target does not measure style, it measures the
tokenizer, and the difference lands in the results looking exactly like a finding. That is
why there is one code path here and no per-corpus variants: only the *preprocessor* differs
(markdown vs HTML vs plain text), and each preprocessor's contract is the same — return prose
with code, markup and identifiers removed.

What is deliberately removed before counting, and why:

- fenced and inline code, and HTML ``<code>``/``<pre>`` — otherwise the comparison is
  dominated by whichever corpus contains more source code, which is a topic difference
- URLs and filesystem paths — high-entropy strings that fragment into junk tokens
- markdown link targets, keeping the link *text*, which is prose the author wrote
- HTML tags and entities

What is deliberately kept:

- contractions as single tokens (``you're``, ``don't``), because they are style-bearing and
  the planned n-gram extension needs them intact
- ordinary stopwords, because their rates are a real stylometric signal; filtering happens at
  report time, never at count time
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterator

# --- preprocessing patterns -------------------------------------------------------------

#: Fenced code blocks: ``` or ~~~ delimited, optionally with a language tag.
_FENCED_CODE = re.compile(r"(?ms)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]{0,3}\1[ \t]*$|\Z)")
#: Inline code spans, longest-delimiter-first so ``` inside text does not mis-pair.
_INLINE_CODE = re.compile(r"(?s)(`+)(.+?)\1")
#: HTML blocks whose contents are code, removed wholesale rather than tag-stripped.
_HTML_CODE = re.compile(r"(?is)<(code|pre|script|style)\b[^>]*>.*?</\1\s*>")
_HTML_TAG = re.compile(r"(?s)<[^>]{0,4000}>")
#: Markdown links and images: keep the visible text, drop the target.
_MD_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")
#: Reference-style link definitions, which are pure target.
_MD_LINKDEF = re.compile(r"(?m)^[ \t]{0,3}\[[^\]]+\]:[ \t]*\S+.*$")
_URL = re.compile(r"(?i)\b(?:https?://|www\.|ftp://)\S+")
#: Filesystem-ish tokens: anything containing a slash between word characters, or a bare
#: dotted filename with a plausible extension.
_PATH = re.compile(r"(?:[\w.~-]*/[\w./~-]+)|(?:\b[\w-]+\.[a-zA-Z]{1,6}\b(?=[\s,;:)\]]|$))")
#: Markdown table delimiter rows, which are punctuation-only but very numerous.
_TABLE_RULE = re.compile(r"(?m)^[ \t]{0,3}\|?[ \t:|-]+\|[ \t:|-]*$")
#: Setext/ATX heading markers and emphasis runs are handled by the word pattern itself.

#: A word: ASCII letters joined by internal apostrophes or hyphens — ``you're``, ``o'clock``,
#: ``re-run``, ``well-known``. Digits are excluded entirely; a version number and a row count
#: are not vocabulary.
#:
#: Hyphens join rather than split deliberately. Splitting them emits the bound prefix as a
#: standalone token, and ``re`` from ``re-run``/``re-read`` alone reached rank ~90 in the
#: Claude corpus. Because prefixed compounds are a register habit rather than a word, that
#: fragment would have ranked as a heavily overused "word" that nobody actually writes.
_WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")

#: An orphaned contraction or possessive: the apostrophe survives when the word it attached
#: to was removed as code, as in ``the `ufw`'s default`` -> `` 's default``. Left alone these
#: become bare ``s``/``t``/``ll`` tokens, and because they are produced by code density
#: rather than by prose they concentrate in exactly the corpus that contains most code.
_ORPHAN_CLITIC = re.compile(r"(?<![a-z])'(?:s|t|d|m|ll|re|ve)\b")

#: Single letters that are real English words. Every other one-letter token is a markup
#: artifact — an emphasis run, a list marker, a stray initial — not vocabulary.
_REAL_SINGLE_LETTERS = frozenset({"a", "i"})

#: Apostrophe-like codepoints normalised to U+0027 so ``you’re`` and ``you're`` are one token.
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "′": "'", "´": "'"})


def strip_markdown(text: str) -> str:
    """Remove code, markup and identifiers from markdown prose.

    Order matters: fenced blocks go before inline spans (a fence contains backticks), and
    link *targets* go before URL stripping (so the visible text survives).
    """
    text = _FENCED_CODE.sub(" ", text)
    text = _MD_LINKDEF.sub(" ", text)
    text = _MD_LINK.sub(r" \1 ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _HTML_CODE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _TABLE_RULE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PATH.sub(" ", text)
    return text


def strip_html(text: str) -> str:
    """Remove code, tags and entities from HTML prose (StackExchange bodies)."""
    text = _HTML_CODE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    # Unescaping can reveal a second layer of markup in double-encoded dumps.
    text = _HTML_CODE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PATH.sub(" ", text)
    return text


def strip_plain(text: str) -> str:
    """Remove only URLs and paths. For corpora that carry no markup (Gutenberg)."""
    text = _URL.sub(" ", text)
    text = _PATH.sub(" ", text)
    return text


#: Preprocessor by name. Additive: a new corpus format appends a key here.
PREPROCESSORS = {
    "markdown": strip_markdown,
    "html": strip_html,
    "plain": strip_plain,
}


def normalise(text: str) -> str:
    """Fold to a comparable form: NFKC, unified apostrophes, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_APOSTROPHES)
    return text.lower()


def iter_tokens(text: str, *, preprocessor: str = "markdown") -> Iterator[str]:
    """Yield prose tokens from one document.

    Trailing possessives are kept attached (``claude's`` stays one token) because splitting
    them would silently inflate the count of ``s``.
    """
    strip = PREPROCESSORS[preprocessor]
    cleaned = _ORPHAN_CLITIC.sub(" ", normalise(strip(text)))
    for match in _WORD.finditer(cleaned):
        # A leading or trailing apostrophe survives the pattern only via quoting artifacts.
        token = match.group(0).strip("'")
        if not token:
            continue
        if len(token) == 1 and token not in _REAL_SINGLE_LETTERS:
            continue
        yield token


def tokenize(text: str, *, preprocessor: str = "markdown") -> list[str]:
    """Eager form of :func:`iter_tokens`."""
    return list(iter_tokens(text, preprocessor=preprocessor))
