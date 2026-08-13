#!/usr/bin/env bash
# Rebuild every result from scratch.
#
#   bash scripts/reproduce.sh
#
# Reads the local Claude Code transcript store, downloads the reference corpora, counts them,
# runs the comparison, the null test, the phrase stage and the domain analysis, and writes the
# report and the HTML artifact.
#
# It is idempotent. Downloads are skipped when the files are already present, and every step
# overwrites its own outputs, so a re-run after an interruption resumes rather than duplicates.
#
# What it needs, and what it costs:
#
#   disk      ~7 GB under data/input for the reference corpora
#   network   the corpora come from Hugging Face; HF_TOKEN is optional and only raises limits
#   time      roughly 60-90 minutes on 16 cores, almost all of it tokenizing
#
# The one input this cannot fetch is the target corpus: `zipf harvest` reads
# ~/.claude/projects, which is the machine's own Claude Code history. On a machine with no
# transcripts the harvest step refuses rather than producing an empty comparison, and every
# later step refuses in turn. That is deliberate — see docs/METHODOLOGY.md.

set -euo pipefail

cd "$(dirname "$0")/.."

# Keep the Hugging Face cache beside the corpora rather than in ~/.cache, which usually sits on
# a smaller filesystem than the one chosen for data/.
export HF_HOME="${HF_HOME:-$PWD/data/input/.hf_cache}"

FOLD="${ZIPF_FOLD:-inflection}"

step() { printf '\n\033[1m=== %s\033[0m\n' "$1"; }

step "1/8  install"
uv sync

step "2/8  harvest the local Claude Code transcripts"
uv run zipf harvest

step "3/8  fetch the reference corpora (skips what is already downloaded)"
uv run zipf fetch --tier all
# Two tarballs rather than a dataset, so it has its own builder.
uv run python scripts/fetch_vcs_corpus.py

step "4/8  count every corpus"
# Claude first: the reference tiers need its vocabulary to size their per-part tables.
uv run zipf count --corpus claude_main
uv run zipf count --corpus claude_sidechain
for tier in literature reddit technical web biomedical commits vcs; do
    uv run zipf count --corpus "$tier"
done

step "5/8  null test — the false-positive floor that decides whether any of this means anything"
uv run zipf calibrate

step "6/8  compare, unfolded and folded"
uv run zipf compare --top 0
uv run zipf compare --fold "$FOLD" --top 0

step "7/8  phrases, then the domain split and the recalibrated threshold"
uv run zipf chains --top 0
uv run zipf domain --fold "$FOLD" --top 0

step "8/8  write the report, the corpus doc and the HTML artifact"
uv run zipf report
uv run python scripts/build_corpora_doc.py
uv run python scripts/build_artifact.py

printf '\n\033[1mdone\033[0m — results in data/output/, corpus provenance in docs/CORPORA.md\n'
