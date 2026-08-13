"""Morphological folding.

The counts in these fixtures are chosen to exercise the frequency guards, so they are written
as small explicit tables rather than derived from a corpus: the point is the rule, not the data.
"""

from __future__ import annotations

import pytest

from zipf.morphology import (
    AMBIGUOUS_ROLE,
    FOLD_LEVELS,
    apply_fold,
    build_fold_map,
    nominal_forms,
    verbal_forms,
)


def counts(*words: str, base: int = 500) -> dict[str, int]:
    """Every word attested comfortably above the guards."""
    return dict.fromkeys(words, base)


def test_nominal_forms_follow_english_spelling() -> None:
    assert nominal_forms("gap")["gaps"] == "plural"
    assert nominal_forms("box")["boxes"] == "plural"
    assert nominal_forms("policy")["policies"] == "plural"
    assert nominal_forms("gap")["gap's"] == "possessive"


def test_verbal_forms_follow_english_spelling() -> None:
    assert verbal_forms("verify")["verified"] == "past"
    assert verbal_forms("verify")["verifying"] == "progressive"
    assert verbal_forms("use")["used"] == "past"
    assert verbal_forms("use")["using"] == "progressive"
    assert verbal_forms("stop")["stopped"] == "past"
    assert verbal_forms("stop")["stopping"] == "progressive"


def test_the_users_inner_brackets_fold_together() -> None:
    """`[[[gap, gaps, gap's], gaped]]` — the two inner brackets of the requested ladder."""
    fold, groups = build_fold_map(
        counts("gap", "gaps", "gap's", "gaped"), level="inflection"
    )
    # `gaps` is labelled ambiguous: without a POS tagger there is no evidence to say whether a
    # given occurrence is the plural noun or the third-person verb, and guessing would be a
    # silent claim. The *merge* is unambiguous either way — both roles share the base.
    assert set(groups["gap"]) == {
        ("gaps", AMBIGUOUS_ROLE),
        ("gap's", "possessive"),
        ("gaped", "past"),
    }
    assert fold["gaps"] == "gap"


def test_the_outer_bracket_does_not_fold() -> None:
    """`agape` is a different word. Folding it in would be a semantic error, not a merge."""
    fold, _ = build_fold_map(counts("gap", "gaps", "agape"), level="inflection")
    assert "agape" not in fold


def test_derivational_relatives_are_left_alone() -> None:
    """`verification` is a level above inflection and is deliberately out of scope."""
    fold, _ = build_fold_map(
        counts("verify", "verified", "verifying", "verification"), level="inflection"
    )
    assert "verification" not in fold
    assert fold["verified"] == "verify"


def test_nominal_level_excludes_verb_forms() -> None:
    """Cheapest layer first: the noun paradigm alone."""
    fold, _ = build_fold_map(
        counts("gap", "gaps", "gap's", "gaped", "gapping"), level="nominal"
    )
    assert fold["gaps"] == "gap"
    assert "gaped" not in fold
    assert "gapping" not in fold


def test_none_level_folds_nothing() -> None:
    fold, groups = build_fold_map(counts("gap", "gaps"), level="none")
    assert fold == {} and groups == {}


def test_unknown_level_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown fold level"):
        build_fold_map(counts("gap"), level="lemmatise")


def test_all_declared_levels_run() -> None:
    for level in FOLD_LEVELS:
        build_fold_map(counts("gap", "gaps"), level=level)


def test_rare_fragment_cannot_swallow_a_common_word() -> None:
    """The real defect: `noth` is a scrap of OCR damage, `nothing` is a word.

    Attestation alone made `noth` a lemma because a 240-million-word union contains almost any
    string. The frequency guard is what stops a fragment capturing the word it prefixes.
    """
    fold, groups = build_fold_map({"noth": 26, "nothing": 200_000}, level="inflection")
    assert "nothing" not in fold
    assert "noth" not in groups


def test_a_plausible_base_still_folds_at_realistic_ratios() -> None:
    """The guard must not be so tight that it breaks correct merges.

    `running` genuinely outnumbers `run` in some corpora, and that has to keep working.
    """
    fold, _ = build_fold_map({"run": 1_000, "running": 8_000, "runs": 2_000}, level="inflection")
    assert fold["running"] == "run"
    assert fold["runs"] == "run"


def test_contractions_are_not_treated_as_possessives() -> None:
    """`let's` is `let us`, a different construction from `let`."""
    fold, _ = build_fold_map(counts("let", "lets", "letting", "let's"), level="inflection")
    assert "let's" not in fold
    assert fold["lets"] == "let"


def test_function_words_are_protected_in_both_directions() -> None:
    """`i` + s would otherwise swallow `is`, and `doe` + s would swallow `does`."""
    fold, _ = build_fold_map(counts("i", "is", "doe", "does", "it", "its"), level="inflection")
    for word in ("is", "does", "its"):
        assert word not in fold


def test_ambiguous_forms_are_left_unfolded_rather_than_guessed() -> None:
    """Two plausible bases means no merge: a wrong merge is silent, a missed merge is not."""
    fold, _ = build_fold_map(counts("axe", "axes", "axis"), level="inflection")
    # `axes` is the plural of both `axe` and `axis`; neither may claim it.
    assert "axes" not in fold


def test_transitive_chains_resolve_to_the_root() -> None:
    fold, _ = build_fold_map(
        counts("read", "reading", "readings", "reads"), level="inflection"
    )
    assert fold["readings"] == "read"


def test_irregulars_are_exempt_from_the_frequency_guard() -> None:
    """`data` really is far more frequent than `datum`; the pair is hand-verified."""
    fold, _ = build_fold_map({"datum": 60, "data": 500_000}, level="inflection")
    assert fold["data"] == "datum"


def test_apply_fold_conserves_every_token() -> None:
    """Folding moves counts between keys; it must never create or destroy one.

    Every rate downstream is computed on the corpus total, so a fold that lost tokens would
    silently change every number in the report.
    """
    table = {"gap": 100, "gaps": 40, "gap's": 5, "unrelated": 7}
    fold, _ = build_fold_map(counts(*table), level="inflection")
    folded = apply_fold(table, fold)
    assert sum(folded.values()) == sum(table.values())
    assert folded["gap"] == 145
    assert folded["unrelated"] == 7


def test_apply_fold_with_an_empty_map_is_a_copy() -> None:
    table = {"gap": 3}
    result = apply_fold(table, {})
    assert result == table and result is not table
