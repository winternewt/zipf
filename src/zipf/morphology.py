"""Fold inflected forms onto a lemma, cheapest layer first.

`gap`, `gaps` and `gap's` are one habit counted three times. Left unfolded they compete with
each other for rank, they split the evidence for a word that is individually rare, and they
fill the report with near-duplicates.

**Generate-and-verify, not strip-and-hope.** A classic stemmer (Porter, Snowball) works by
chopping suffixes, which over-merges (`verification` -> `verif`) and emits non-words as group
labels. This module runs the other way: from a candidate base it *generates* the forms English
inflection would produce, and keeps only those actually **attested in the corpora**. Every
group is therefore labelled by a real word, and no merge happens that the data does not
witness.

**The layers, cheapest first**, matching the ladder the user drew — `[[[gap, gaps, gap's],
gaped], agape]`:

``none``
    Exact types. What the first run reported.
``nominal``
    The innermost bracket only: base, plural, possessive. The safest merge, because the noun
    paradigm is nearly unambiguous.
``inflection``
    Adds the verbal paradigm: 3rd person, past, progressive. This is full inflectional
    lemmatization and the default.

**The outermost bracket is deliberately not implemented.** `agape` is not a form of `gap` in
any sense a frequency table should act on — it is a separate word (and separately, a Greek
noun for a love-feast). Derivational relatives like `verify`/`verification` are a real level
above this one, but they change part of speech and often meaning, so merging them is a
research decision rather than a normalisation. It is left for `ROADMAP.md`.

**No re-tokenization is needed.** Folding is a regroup of the `token -> count` tables that
already exist for every corpus, so adding a layer costs seconds rather than another pass over
240 million words.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping

logger = logging.getLogger(__name__)

FOLD_LEVELS: tuple[str, ...] = ("none", "nominal", "inflection")

VOWELS = frozenset("aeiou")

#: Shortest string allowed to act as a base. Two-letter bases generate catastrophes — `i` +s
#: would swallow `is`, `ha` + s would swallow `has` — and English has almost no genuine
#: two-letter lemmas with productive paradigms.
MIN_BASE_LENGTH = 3

#: A base must be attested at least this often across the pooled corpora to be a lemma. Filters
#: the long tail of typos and OCR fragments that a 240-million-word union inevitably contains.
MIN_BASE_COUNT = 50

#: A form may be at most this many times more frequent than its proposed base. Real English
#: lemmas are not thousands of times rarer than their own inflections, so this rejects a
#: fragment that merely looks like a stem. Set generously: `running` genuinely outnumbers `run`
#: in some corpora, and the guard must not break correct merges to catch wrong ones.
MAX_FORM_TO_BASE_RATIO = 25

#: Words that must never be folded into anything, in either direction. Every one is a function
#: word that a naive rule would capture: `its` looks like the plural of `it`, `does` like the
#: plural of `doe`, `was` like a verb form of `wa`. Function-word rates are a real stylometric
#: signal, so corrupting them to save a merge is a bad trade.
PROTECTED = frozenset(
    """
    is was as has his this its it us thus does goes less unless yes gas plus bus lens news
    means during being having does said says axis basis series species process access
    address progress press cross loss class pass mass glass grass guess dress stress
    """.split()
) | frozenset(
    # `X's` is ambiguous between a possessive and a contraction of `is`/`has`/`us`, and the
    # possessive rule cannot tell them apart. `let's` is `let us` and is a different
    # construction from `let` — folding it in would merge two habits the study is trying to
    # tell apart. Contractions are protected; genuine possessives like `agent's` still fold.
    """
    let's that's there's here's what's who's how's where's when's he's she's it's
    one's today's yesterday's tomorrow's
    """.split()
)

#: Irregular paradigms worth carrying by hand. Kept deliberately small: only forms frequent
#: enough in ordinary prose that leaving them unmerged would distort a rate.
IRREGULAR: dict[str, tuple[str, ...]] = {
    "be": ("am", "are", "been", "being"),
    "have": ("had", "having"),
    "do": ("did", "doing", "done"),
    "say": ("said", "saying", "says"),
    "make": ("made", "making", "makes"),
    "take": ("took", "taken", "taking", "takes"),
    "give": ("gave", "given", "giving", "gives"),
    "find": ("found", "finding", "finds"),
    "get": ("got", "gotten", "getting", "gets"),
    "keep": ("kept", "keeping", "keeps"),
    "leave": ("left", "leaving", "leaves"),
    "lose": ("lost", "losing", "loses"),
    "mean": ("meant", "meaning", "means"),
    "read": ("reading", "reads"),
    "run": ("ran", "running", "runs"),
    "see": ("saw", "seen", "seeing", "sees"),
    "set": ("setting", "sets"),
    "show": ("showed", "shown", "showing", "shows"),
    "tell": ("told", "telling", "tells"),
    "think": ("thought", "thinking", "thinks"),
    "write": ("wrote", "written", "writing", "writes"),
    "build": ("built", "building", "builds"),
    "hold": ("held", "holding", "holds"),
    "let": ("letting", "lets"),
    "put": ("putting", "puts"),
    "catch": ("caught", "catching", "catches"),
    "bring": ("brought", "bringing", "brings"),
    "child": ("children",),
    "person": ("people",),
    "man": ("men",),
    "woman": ("women",),
    "foot": ("feet",),
    "index": ("indices", "indexes"),
    "matrix": ("matrices",),
    "analysis": ("analyses",),
    "axis": ("axes",),
    "hypothesis": ("hypotheses",),
    "criterion": ("criteria",),
    "datum": ("data",),
}


#: Role recorded when the noun and verb paradigms generate the same surface form.
AMBIGUOUS_ROLE = "plural_or_third_person"


def _role_label(roles: set[str]) -> str:
    """One label for a form, keeping ambiguity visible instead of resolving it by accident."""
    if {"plural", "third_person"} <= roles:
        return AMBIGUOUS_ROLE
    return sorted(roles)[0]


def _sibilant(base: str) -> bool:
    return base.endswith(("s", "x", "z", "ch", "sh", "ss"))


def _consonant_y(base: str) -> bool:
    return len(base) > 2 and base.endswith("y") and base[-2] not in VOWELS


def _doubles(base: str) -> bool:
    """Approximate the consonant-doubling rule: stop -> stopped, run -> running."""
    if len(base) < 3:
        return False
    last, middle, first = base[-1], base[-2], base[-3]
    return (
        last not in VOWELS
        and last not in "wxy"
        and middle in VOWELS
        and first not in VOWELS
    )


def nominal_forms(base: str) -> dict[str, str]:
    """Plural and possessive. The innermost bracket."""
    forms: dict[str, str] = {}
    if _sibilant(base):
        forms[base + "es"] = "plural"
    elif _consonant_y(base):
        forms[base[:-1] + "ies"] = "plural"
    else:
        forms[base + "s"] = "plural"
    forms[base + "'s"] = "possessive"
    return forms


def verbal_forms(base: str) -> dict[str, str]:
    """Third person, past and progressive."""
    forms: dict[str, str] = {}
    if _sibilant(base):
        forms[base + "es"] = "third_person"
    elif _consonant_y(base):
        forms[base[:-1] + "ies"] = "third_person"
    else:
        forms[base + "s"] = "third_person"

    if base.endswith("e"):
        forms[base + "d"] = "past"
        forms[base[:-1] + "ing"] = "progressive"
    elif _consonant_y(base):
        forms[base[:-1] + "ied"] = "past"
        forms[base + "ing"] = "progressive"
    elif _doubles(base):
        forms[base + base[-1] + "ed"] = "past"
        forms[base + base[-1] + "ing"] = "progressive"
        forms[base + "ed"] = "past"
        forms[base + "ing"] = "progressive"
    else:
        forms[base + "ed"] = "past"
        forms[base + "ing"] = "progressive"
    return forms


def build_fold_map(
    counts: Mapping[str, int], *, level: str = "inflection"
) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Map every attested inflected form to its lemma.

    ``counts`` must be **pooled across every corpus**. Building it from the target alone would
    fold the target's forms while leaving the baselines' forms split, which inflates the
    target's apparent rate for exactly the words being tested — the comparison would be
    checking a convention against itself.

    Counts rather than a bare vocabulary, because attestation alone is not enough evidence that
    a string is a word. 240 million words of scraped text contain an enormous tail of typos,
    OCR damage and fragments, and any three-letter fragment that happens to prefix a real word
    will be generated into it: ``noth`` + ``ing`` swallowed ``nothing``, ``statu`` + ``s``
    swallowed ``status``, and ``sible`` and ``untouch`` became lemmas. The guard is
    :data:`MAX_FORM_TO_BASE_RATIO` — a real lemma is not thousands of times rarer than its own
    inflected form.

    Returns ``(form -> lemma, lemma -> [(form, role)])``. A form that more than one attested
    base could produce is left **unfolded** rather than assigned by guesswork: an ambiguous
    merge is a silent error, an unfolded form is merely a missed merge.
    """
    if level not in FOLD_LEVELS:
        raise ValueError(f"unknown fold level {level!r}; known: {FOLD_LEVELS}")

    if level == "none":
        return {}, {}
    attested = set(counts)

    def plausible(base: str, form: str) -> bool:
        """Is ``base`` frequent enough to be a real lemma for ``form``?"""
        base_count = counts.get(base, 0)
        if base_count < MIN_BASE_COUNT:
            return False
        return base_count * MAX_FORM_TO_BASE_RATIO >= counts.get(form, 0)

    # Propose every (form -> base) the rules generate, keeping all proposals so that a form
    # claimed by two different bases can be detected rather than silently resolved.
    proposals: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for base in attested:
        if len(base) < MIN_BASE_LENGTH or base in PROTECTED or "'" in base:
            continue
        # Roles are collected as a set rather than overwritten. `-s` is genuinely both a plural
        # and a third-person marker, and without a part-of-speech tagger there is no evidence
        # to choose between them — so the role is recorded as ambiguous rather than decided by
        # whichever rule happened to run last.
        candidates: dict[str, set[str]] = defaultdict(set)
        for form, role in nominal_forms(base).items():
            candidates[form].add(role)
        if level == "inflection":
            for form, role in verbal_forms(base).items():
                candidates[form].add(role)
        for form, roles in candidates.items():
            if form != base and form in attested and form not in PROTECTED:
                if plausible(base, form):
                    proposals[form].add((base, _role_label(roles)))

    if level == "inflection":
        # Irregular pairs are hand-verified, so they are exempt from the frequency guard. It
        # would reject several of them for the right reason and the wrong outcome: `data` really
        # is far more frequent than `datum`, and `people` than `person`.
        for base, forms in IRREGULAR.items():
            if base not in attested:
                continue
            for form in forms:
                if form in attested and form not in PROTECTED:
                    proposals[form].add((base, "irregular"))

    fold: dict[str, str] = {}
    ambiguous = 0
    for form, options in proposals.items():
        bases = {base for base, _ in options}
        if len(bases) != 1:
            ambiguous += 1
            continue
        fold[form] = next(iter(bases))

    # A form may point at a base that is itself a form of something else (readings -> reading
    # -> read). Resolve to the root, with a hop limit so a rule cycle cannot hang the build.
    roots: dict[str, str] = {}
    for form in fold:
        seen = {form}
        current = fold[form]
        for _ in range(8):
            nxt = fold.get(current)
            if nxt is None or nxt in seen:
                break
            seen.add(current)
            current = nxt
        roots[form] = current

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for form, lemma in roots.items():
        role = next((r for b, r in proposals[form] if b == fold[form]), "inflection")
        groups[lemma].append((form, role))
    for lemma in groups:
        groups[lemma].sort()

    logger.info(
        "fold level %s: %d forms folded into %d lemmas, %d ambiguous forms left unfolded",
        level,
        len(roots),
        len(groups),
        ambiguous,
    )
    return roots, dict(groups)


def apply_fold(counts: dict[str, int], fold: dict[str, str]) -> dict[str, int]:
    """Regroup a ``token -> count`` table under its lemmas.

    Token totals are preserved exactly: folding moves counts between keys and never creates or
    destroys any, so every rate computed downstream stays on the same denominator.
    """
    if not fold:
        return dict(counts)
    folded: dict[str, int] = defaultdict(int)
    for token, count in counts.items():
        folded[fold.get(token, token)] += count
    return dict(folded)
