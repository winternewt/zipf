"""Pydantic models for everything crossing a file or CLI boundary.

Vocabularies that must grow additively across corpus additions are `frozenset[str]` plus a
validator rather than `Literal`, because a `Literal` in a model that is persisted to parquet
makes every future addition a breaking change to files already on disk.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Corpus identifiers. Additive: a new tier appends here and old manifests stay readable.
CORPUS_IDS: frozenset[str] = frozenset(
    {
        "claude_main",
        "claude_sidechain",
        "literature",
        "reddit",
        "technical",
        "web",
        "biomedical",
        "commits",
        "vcs",
    }
)

#: The four human reference tiers, in the order they are reported.
REFERENCE_TIERS: tuple[str, ...] = (
    "literature",
    "reddit",
    "technical",
    "web",
    "biomedical",
    "commits",
)

#: Register label, used to say which confound a tier controls for.
REGISTERS: frozenset[str] = frozenset(
    {
        "literary",
        "conversational",
        "technical",
        "web",
        "assistant",
        "biomedical",
        "engineering",
        "documentation",
    }
)


class CorpusSpec(BaseModel):
    """Provenance of one corpus. Written beside every token stream it produces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_id: str
    text_register: str
    source: str = Field(description="Human-readable origin, e.g. a HF dataset id or a URL.")
    licence: str
    date_cutoff: str | None = Field(
        default=None,
        description=(
            "Latest content date admitted, ISO-8601. None means 'unknown', which is NOT the "
            "same as 'no cutoff applied' — see contamination_note."
        ),
    )
    contamination_note: str = Field(
        description="How much LLM-generated text this corpus plausibly contains, and why."
    )

    @field_validator("corpus_id")
    @classmethod
    def _known_corpus(cls, value: str) -> str:
        if value not in CORPUS_IDS:
            raise ValueError(f"unknown corpus_id {value!r}; add it to CORPUS_IDS first")
        return value

    @field_validator("text_register")
    @classmethod
    def _known_register(cls, value: str) -> str:
        if value not in REGISTERS:
            raise ValueError(f"unknown register {value!r}; add it to REGISTERS first")
        return value


class CorpusStats(BaseModel):
    """Summary of a built token stream. The denominators every rate is computed against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_id: str
    documents: int
    parts: int = Field(description="Dispersion units: sessions for Claude, shards otherwise.")
    tokens: int
    types: int
    built_at: datetime

    @property
    def type_token_ratio(self) -> float:
        return self.types / self.tokens if self.tokens else 0.0


class OveruseRow(BaseModel):
    """One word's verdict against one reference tier.

    `unknown` is a first-class outcome: a tier that failed to build yields None for its
    statistics, which is not the same as a z-score of zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str
    reference: str
    claude_count: int
    reference_count: int
    claude_per_million: float
    reference_per_million: float
    log_odds_z: float | None = Field(
        default=None, description="Monroe log-odds with informative Dirichlet prior. None = unknown."
    )
    log_likelihood_g2: float | None = None
    dispersion_dp: float | None = Field(
        default=None, description="Gries DP over Claude sessions. 0 = perfectly even, 1 = one session."
    )
    sessions_present: int | None = None
    max_session_share: float | None = None


TokenCount = Annotated[int, Field(ge=0)]
