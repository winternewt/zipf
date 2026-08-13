"""The four human reference tiers: what they are, where they come from, and how to read them.

Each tier answers a different question, and the panel exists because no single one is
sufficient:

``literature``
    Edited literary English. The "wilderness" baseline. Maximally distant in register, which
    makes it good at catching genuinely odd word choice and bad at separating style from topic.
``reddit``
    Unedited conversational English. Catches the difference between how people write to each
    other and how an assistant writes to a user.
``technical``
    StackOverflow questions and answers: humans writing prose *about code*, with fenced code
    blocks, in the same markdown as the target corpus. This is the tier that holds topic
    constant, so what survives against it is style rather than subject matter.
``web``
    A dated Common Crawl snapshot. Broad modern written English, independent of the other three.

Every tier is dated to before the period when LLM text became common on the open internet.
That bias runs *toward the null*: any contamination makes a baseline more Claude-like and so
shrinks the measured difference. A surviving finding is therefore conservative.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from zipf.models import CorpusSpec
from zipf.paths import INPUT_DIR

logger = logging.getLogger(__name__)

#: Dispersion parts per reference corpus.
#:
#: Documents are assigned round-robin rather than in contiguous blocks of N. Blocks looked
#: reasonable and were not: document sizes differ by four orders of magnitude between tiers —
#: a Gutenberg row is a whole book, a Reddit row is a sentence — so a fixed block size put an
#: entire 60-million-token literary corpus into a single part, and any dispersion computed
#: over it would have been undefined while reporting as valid. Round-robin gives balanced
#: parts whatever the document size.
REFERENCE_PARTS = 200


@dataclass(frozen=True, slots=True)
class TierPlan:
    """How to acquire and read one reference tier."""

    corpus_id: str
    repo_id: str
    files: tuple[str, ...]
    reader: str
    text_fields: tuple[str, ...]
    preprocessor: str
    spec: CorpusSpec
    date_field: str | None = None
    date_below: str | None = None
    #: How to read `date_field`. "iso" compares the string directly, which is only valid for
    #: ISO-8601. "dmy" parses `DD.MM.YYYY ...` first — a lexicographic compare on that format
    #: sorts by day-of-month and would silently admit the wrong years.
    date_format: str = "iso" 

    @property
    def directory(self) -> Path:
        return INPUT_DIR / self.corpus_id


TIERS: dict[str, TierPlan] = {
    "literature": TierPlan(
        corpus_id="literature",
        repo_id="manu/project_gutenberg",
        files=(
            "data/en-00000-of-00052-7cda8f63c262acf8.parquet",
            "data/en-00001-of-00052-5c2b3fd5e60f0124.parquet",
        ),
        reader="parquet",
        text_fields=("text",),
        preprocessor="plain",
        spec=CorpusSpec(
            corpus_id="literature",
            text_register="literary",
            source="Hugging Face manu/project_gutenberg, English shards 0-1",
            licence="Project Gutenberg licence; works are public domain in the US",
            date_cutoff="1970-01-01",
            contamination_note=(
                "Effectively zero. Public-domain works, overwhelmingly pre-1930. The opposite "
                "risk applies instead: archaic vocabulary and orthography inflate the apparent "
                "novelty of ordinary modern words, so this tier over-reports rather than "
                "under-reports."
            ),
        ),
    ),
    "reddit": TierPlan(
        corpus_id="reddit",
        repo_id="sentence-transformers/reddit-title-body",
        files=(
            "reddit_title_text_2010.jsonl.gz",
            "reddit_title_text_2011.jsonl.gz",
            "reddit_title_text_2012.jsonl.gz",
        ),
        reader="jsonl_gz",
        text_fields=("body", "selftext", "text", "title"),
        preprocessor="markdown",
        spec=CorpusSpec(
            corpus_id="reddit",
            text_register="conversational",
            source="Hugging Face sentence-transformers/reddit-title-body, years 2010-2012",
            licence="Reddit user content, redistributed for research",
            date_cutoff="2012-12-31",
            contamination_note=(
                "Zero. These years predate large language models entirely, so no part of this "
                "tier can contain generated text."
            ),
        ),
    ),
    "technical": TierPlan(
        corpus_id="technical",
        repo_id="mikex86/stackoverflow-posts",
        files=(
            "stackoverflow-posts-00000-of-00058.parquet",
            "stackoverflow-posts-00001-of-00058.parquet",
            "stackoverflow-posts-00002-of-00058.parquet",
        ),
        reader="parquet",
        text_fields=("Body",),
        preprocessor="markdown",
        date_field="CreationDate",
        date_below="2022-01-01",
        spec=CorpusSpec(
            corpus_id="technical",
            text_register="technical",
            source="Hugging Face mikex86/stackoverflow-posts, shards 0-2, CreationDate < 2022",
            licence="CC BY-SA (StackExchange content licence)",
            date_cutoff="2022-01-01",
            contamination_note=(
                "Near zero. Shards are ordered by post id, so these are the earliest posts "
                "(2008 onward), and the date filter is applied on top. StackOverflow banned "
                "generated answers in late 2022, after this window closes."
            ),
        ),
    ),
    "biomedical": TierPlan(
        corpus_id="biomedical",
        repo_id="MedRAG/pubmed",
        files=tuple(f"chunk/pubmed23n{i:04d}.jsonl" for i in range(1, 21)),
        reader="jsonl",
        text_fields=("content",),
        preprocessor="plain",
        spec=CorpusSpec(
            corpus_id="biomedical",
            text_register="biomedical",
            source="Hugging Face MedRAG/pubmed, chunks 1-20 (oldest PMIDs)",
            licence="PubMed abstracts, US National Library of Medicine terms",
            date_cutoff="1990-01-01",
            contamination_note=(
                "Zero. Chunks are ordered by PMID, so chunk 1 begins at PMID 21 — records from "
                "the 1970s. This tier exists to control the topic confound the other four cannot: "
                "the target corpus is bioinformatics-heavy, so domain nouns like 'chromosome' and "
                "'annotation' score against every general-English baseline while being purely "
                "topical. Measured against biomedical writing they stop scoring, without anyone "
                "hand-curating a list of which words count as domain terms."
            ),
        ),
    ),
    "commits": TierPlan(
        corpus_id="commits",
        repo_id="JetBrains-Research/commit-chronicle",
        files=(
            "data/train-00000-of-00061-2a7ccc8e843f5f5b.parquet",
            "data/train-00001-of-00061-def39c7ec3091950.parquet",
            "data/train-00002-of-00061-dfcab59179577c46.parquet",
            "data/train-00003-of-00061-f93845bb3aadee0b.parquet",
            "data/train-00004-of-00061-d476f7d1228697e8.parquet",
            "data/train-00005-of-00061-e56b53608ef6ce93.parquet",
            "data/train-00006-of-00061-a9529faebc3924cc.parquet",
            "data/train-00007-of-00061-138c495790e8fd92.parquet",
            "data/train-00008-of-00061-9932c9ea4d85ff58.parquet",
            "data/train-00009-of-00061-15d88ff4993b6ef9.parquet",
            "data/train-00010-of-00061-e07036c37bcee6fd.parquet",
            "data/train-00011-of-00061-2eec95047f9f79e9.parquet",
            "data/train-00012-of-00061-ed6de29bbde599b4.parquet",
            "data/train-00013-of-00061-43599ab6ed492d67.parquet",
        ),
        reader="parquet",
        text_fields=("message",),
        preprocessor="markdown",
        date_field="date",
        date_below="2022-01-01",
        date_format="dmy",
        spec=CorpusSpec(
            corpus_id="commits",
            text_register="engineering",
            source="Hugging Face JetBrains-Research/commit-chronicle, commit messages before 2022",
            licence="Permissively licensed source repositories; see the dataset card",
            date_cutoff="2022-01-01",
            contamination_note=(
                "Low. The corpus runs 1998-2023 and is filtered to before 2022, which removes "
                "the window in which generated commit messages became plausible. This tier "
                "exists because software-collaboration vocabulary - commit, bump, revert, "
                "stale, upstream, deprecate - is a register, not a style, and no general "
                "corpus and no Q&A corpus contains it at its natural rate."
            ),
        ),
    ),
    # Not in REFERENCE_TIERS: this is an instrument, not a gate. See scripts/fetch_vcs_corpus.py
    # for why a sub-million-word corpus cannot be one of the tiers a word must clear.
    "vcs": TierPlan(
        corpus_id="vcs",
        repo_id="local",
        files=("vcs_docs.jsonl",),
        reader="jsonl",
        text_fields=("text",),
        preprocessor="asciidoc",
        spec=CorpusSpec(
            corpus_id="vcs",
            text_register="documentation",
            source=(
                "Pro Git (Chacon & Straub, progit/progit2) plus git's own Documentation/*.txt "
                "at tag v2.43.0; built by scripts/fetch_vcs_corpus.py"
            ),
            licence="Pro Git: CC BY-NC-SA 3.0. git documentation: GPL-2.0. Not redistributed.",
            date_cutoff="2023-11-20",
            contamination_note=(
                "Effectively zero: both are long-standing human-authored technical documents "
                "under version control, and the git tag is pinned. This corpus exists to answer "
                "one question — whether a word is simply the vocabulary of version control — "
                "and is deliberately not one of the tiers a word must clear, because at under a "
                "million words too many candidate words occur in it zero times, and a zero "
                "reference count collapses the log-odds z whatever the effect size (F10)."
            ),
        ),
    ),
    "web": TierPlan(
        corpus_id="web",
        repo_id="HuggingFaceFW/fineweb",
        files=("data/CC-MAIN-2021-04/000_00000.parquet",),
        reader="parquet",
        text_fields=("text",),
        preprocessor="plain",
        spec=CorpusSpec(
            corpus_id="web",
            text_register="web",
            source="Hugging Face HuggingFaceFW/fineweb, Common Crawl snapshot CC-MAIN-2021-04",
            licence="ODC-By 1.0; underlying pages retain their own terms",
            date_cutoff="2021-01-31",
            contamination_note=(
                "Very low. The snapshot was crawled in January 2021, before ChatGPT and before "
                "generated text was common on the open web. GPT-3 output existed but was rare."
            ),
        ),
    ),
}


def fetch_tier(tier: TierPlan) -> list[Path]:
    """Download one tier's files, skipping any already present.

    Prints its target and byte total before starting, so an interrupted run can be diagnosed
    without repeating it.
    """
    if tier.repo_id == "local":
        raise ValueError(
            f"tier {tier.corpus_id!r} is built locally, not downloaded. "
            "Run `uv run python scripts/fetch_vcs_corpus.py`."
        )
    tier.directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in tier.files:
        logger.info("fetching %s :: %s", tier.repo_id, filename)
        local = hf_hub_download(
            repo_id=tier.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=tier.directory,
        )
        path = Path(local)
        logger.info("  -> %s (%.0f MB)", path, path.stat().st_size / 1e6)
        paths.append(path)
    return paths


def _normalise_date(value: object, date_format: str) -> str | None:
    """Return an ISO-comparable date string, or None when the value cannot be read.

    `DD.MM.YYYY` must be rearranged before comparison. Comparing it as-is sorts by day of the
    month, so a cutoff of "2022-01-01" would admit or reject rows essentially at random while
    looking like it worked.
    """
    if not isinstance(value, str) or not value:
        return None
    if date_format == "iso":
        return value
    if date_format == "dmy":
        head = value.split(" ", 1)[0]
        parts = head.split(".")
        if len(parts) != 3:
            return None
        day, month, year = parts
        if not (len(year) == 4 and year.isdigit() and month.isdigit() and day.isdigit()):
            return None
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    raise ValueError(f"unknown date_format {date_format!r}")


def _first_text(row: dict, fields: tuple[str, ...]) -> str | None:
    """The first populated text field. Missing everywhere means an unusable row, not empty text."""
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _iter_parquet(path: Path, tier: TierPlan) -> Iterator[str]:
    """Stream text out of a parquet shard without materialising it.

    A FineWeb shard is 2.1 GB on disk; reading it whole is a bug, not a style choice.
    """
    columns = [c for c in (*tier.text_fields, tier.date_field) if c]
    handle = pq.ParquetFile(path)
    available = set(handle.schema_arrow.names)
    columns = [c for c in columns if c in available]
    for batch in handle.iter_batches(batch_size=2000, columns=columns):
        for row in batch.to_pylist():
            if tier.date_field and tier.date_below:
                stamp = _normalise_date(row.get(tier.date_field), tier.date_format)
                # A row whose date we cannot read is excluded rather than assumed in range:
                # "date unknown" is not "date acceptable".
                if stamp is None or stamp >= tier.date_below:
                    continue
            text = _first_text(row, tier.text_fields)
            if text is not None:
                yield text


def _iter_jsonl_gz(path: Path, tier: TierPlan) -> Iterator[str]:
    """Stream text out of a gzipped JSON-lines shard."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            text = _first_text(row, tier.text_fields)
            if text is not None:
                yield text


def _iter_jsonl(path: Path, tier: TierPlan) -> Iterator[str]:
    """Stream text out of a plain JSON-lines shard."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                text = _first_text(row, tier.text_fields)
                if text is not None:
                    yield text


_READERS = {"parquet": _iter_parquet, "jsonl_gz": _iter_jsonl_gz, "jsonl": _iter_jsonl}


def iter_documents(tier: TierPlan, paths: list[Path]) -> Iterator[tuple[str, str]]:
    """Yield ``(part_id, text)`` for a tier, assigning documents to dispersion parts.

    Part ids are derived from position, not from content, so two runs over the same files
    produce the same partition.
    """
    reader = _READERS[tier.reader]
    index = 0
    for path in sorted(paths):
        for text in reader(path, tier):
            yield f"{path.name}:{index % REFERENCE_PARTS:03d}", text
            index += 1
