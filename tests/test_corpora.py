"""Corpus readers and their date filters."""

from __future__ import annotations

import pytest

from zipf.corpora import TIERS, _normalise_date
from zipf.models import REFERENCE_TIERS


def test_iso_dates_pass_through() -> None:
    assert _normalise_date("2021-06-01T10:00:00", "iso") == "2021-06-01T10:00:00"


def test_day_first_dates_are_rearranged_before_comparison() -> None:
    """The bug this exists to prevent: `DD.MM.YYYY` compared as a string sorts by day.

    Under a naive compare, a 2023 commit dated `09.01.2023` sorts *below* a 2017 commit dated
    `17.01.2017`, so a cutoff of 2022 would admit the newer row and reject the older one — the
    exact inverse of the filter's purpose, while appearing to work.
    """
    old = _normalise_date("17.01.2017 12:29:54", "dmy")
    new = _normalise_date("09.01.2023 16:27:32", "dmy")
    assert old == "2017-01-17"
    assert new == "2023-01-09"
    assert old < "2022-01-01" <= new
    # And the failure mode itself, stated as a fact about the raw strings.
    assert not ("17.01.2017 12:29:54" < "2022-01-01" <= "09.01.2023 16:27:32")


def test_unreadable_dates_are_unknown_not_accepted() -> None:
    """A row whose date cannot be read is excluded; unknown is not 'in range'."""
    for value in (None, "", "not-a-date", "1.2.17", 20210601):
        assert _normalise_date(value, "dmy") is None


def test_unknown_date_format_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown date_format"):
        _normalise_date("2021-01-01", "julian")


def test_every_reference_tier_is_declared() -> None:
    for name in REFERENCE_TIERS:
        assert name in TIERS, f"{name} is in REFERENCE_TIERS but has no TierPlan"


def test_every_tier_declares_files_and_a_contamination_note() -> None:
    for name, tier in TIERS.items():
        assert tier.files, f"{name} declares no files"
        assert tier.spec.contamination_note.strip(), f"{name} has no contamination note"
        assert tier.reader in {"parquet", "jsonl", "jsonl_gz"}


def test_dated_tiers_declare_how_to_read_their_dates() -> None:
    for name, tier in TIERS.items():
        if tier.date_below is not None:
            assert tier.date_field, f"{name} filters by date but names no field"
            assert tier.date_format in {"iso", "dmy"}
