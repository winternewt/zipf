# zipf

Measures which words Claude Code overuses.

It harvests the prose half of local Claude Code transcripts, tokenizes it identically to five
human-written reference corpora, and ranks the vocabulary by how far each word's rate departs
from **all** of them.

**This is not a detector.** It compares aggregate rates between corpora. There is no
per-document verdict, and nothing here can tell you whether a particular text was written by a
model.

## Why five baselines

A single baseline cannot separate *style* from *topic*. Compare assistant prose to Victorian
novels and the top of the list is `file`, `function`, `commit` — true, and useless, because it
describes what the text is about rather than how it is written.

| Tier | Source | Controls for |
|---|---|---|
| `literature` | Project Gutenberg, English | modern register generally |
| `reddit` | Reddit 2010–2012 | edited vs unedited writing |
| `technical` | StackOverflow, pre-2022 | **topic** — same subject, same markdown, same fenced code |
| `web` | Common Crawl `CC-MAIN-2021-04` | any single-source quirk |
| `biomedical` | PubMed abstracts, oldest PMIDs | **the target's own subject** — bioinformatics vocabulary |

The ranking statistic is the **minimum** z across tiers, so a word cannot be carried by one
extreme baseline. Every tier predates the period when generated text became common, which
biases results toward the null — surviving findings are conservative.

## Run it

```bash
uv sync
uv run zipf harvest              # ~/.claude/projects -> documents table
uv run zipf fetch --tier all     # download the reference corpora (~5.5 GB)
uv run zipf count --corpus all   # tokenize and count everything
uv run zipf compare              # rank the vocabulary
uv run zipf calibrate            # the null test: does it find nothing when there is nothing?
uv run zipf chains               # extend to 2/3/4-gram phrases
uv run zipf compare --fold inflection   # fold gap/gaps/gap's/gapped into one entry
uv run zipf domain --fold inflection    # split style from domain, recalibrate the threshold
uv run zipf report               # write data/output/report.md
```

`uv run pytest -vvv` runs the suite.

## Reading a result

Every reported word clears six gates: a minimum count after folding, a log-odds z above an
empirically calibrated threshold against *every* tier, agreement in sign from Dunning's G², a
session-level bootstrap lower bound above zero, dispersion gates requiring the word to be spread
across sessions, and a specialisation score low enough that it is not simply the subject matter.

Two things to keep in mind, both measured rather than assumed:

- The null test — the corpus split in half by session and compared against itself — reports a
  **2.71%** false-positive floor. The signal is roughly 11x that floor.
- That floor is made of **domain nouns**, not statistical noise, which is why the panel includes
  a biomedical tier and why every word carries a specialisation score.
- The significance threshold is **read off the null distribution** (z ≥ 4.69 at 1% FPR), not
  assumed. The conventional 3.00 is 56% too lenient on a corpus with this much topical spread.
- Successive corrections take 2,355 candidates → 735 → 477 → 431 → **352 style words**.

`docs/METHODOLOGY.md` says what each statistic is blind to. `docs/CORPORA.md` carries the
provenance, licence and contamination note for every corpus.
