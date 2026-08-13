# Changelog

What actually shipped, newest first.

## Unreleased

### Added — second round: morphology and domain control

- **Morphological folding** (`morphology.py`, `zipf compare --fold`). Generate-and-verify rather
  than suffix stripping: forms are generated from a candidate base and kept only where attested,
  so every group is labelled by a real word. Three layers, cheapest first — `none`, `nominal`
  (plural, possessive), `inflection` (+ third person, past, progressive). Folding stops before
  derivation: `verification` is not merged into `verify`, and `agape` is not merged into `gap`.
  No re-tokenization — it regroups the count tables that already exist.
- **A fifth reference tier: PubMed abstracts** (oldest PMIDs, 1970s, zero contamination). Because
  the ranking statistic is the minimum z across tiers, bioinformatics vocabulary now fails
  structurally rather than by a hand-written jargon list.
- **`domain.py` and `zipf domain`.** Specialisation — `log2(specialist rate / general rate)`,
  computed from the reference corpora only, so it interprets the ranking without contaminating
  it. Project-level dispersion, the only instrument that sees a private vocabulary no public
  corpus contains. And `empirical_threshold`, which reads the significance cut off the null
  distribution instead of assuming one.

### Results after the second round

- The threshold a word must clear is **z ≥ 4.69** at a 1% false-positive rate, not the
  conventional 3.00 — the constant was 56% too lenient for a corpus with this much topical
  spread.
- Successive corrections: **2,355** candidates → **735** (four general baselines) → **477**
  (biomedical tier + folding) → **431** (recalibrated threshold) → **352** style words.
- Domain vocabulary now has a slightly *lower* median z than style vocabulary (1.32 against
  1.40), which is the sign the skew is absorbed rather than merely labelled.
- Measured domain effects: `annotation` 397x general-English rate, `chromosome` 121x,
  `variant` 96x, `gene` 41x.
- Two of the strongest surviving words, `guard` and `gate`, have *negative* specialisation —
  general English uses them more than specialist writing does.
- 82 tests, up from 57.

### Fixed — second round

- **F7** — a junk fragment became a lemma: `noth` (26 occurrences) swallowed `nothing` (~200k).
  Attestation in 240M words of scraped text is nearly free; frequency guards were added.
- **F8** — the possessive rule folded `let's` into `let`. It is `let us`, and `let` is the
  study's single strongest finding.
- **F9** — the role label was decided by dict ordering rather than evidence. `-s` is genuinely
  ambiguous between plural and third person, and is now recorded as such.

### Added

- **The pipeline.** `zipf harvest / fetch / count / compare / calibrate / chains / report`.
  Harvests assistant prose from the local Claude Code transcript store, tokenizes it identically
  to four human reference corpora, and ranks the vocabulary by how far each word's rate departs
  from all of them.
- **One tokenizer for every corpus** (`tokenize.py`). Strips fenced and inline code, HTML
  `<code>`/`<pre>`, URLs, filesystem paths and link targets while keeping link text. Contractions
  and hyphenated compounds stay whole. A baseline tokenized differently from the target measures
  the tokenizer, not the style.
- **Statistics** (`stats.py`): Monroe log-odds with an informative Dirichlet prior and z-scores,
  Dunning's G² as an independent cross-check, Gries' DP for dispersion, a **session-level**
  bootstrap, `dispersion_excess` for a frequency-neutral dispersion gate, and a Zipf slope fit as
  a corpus sanity check.
- **The null test** (`nulltest.py`, `zipf calibrate`). Splits the target corpus in half by session
  and compares it against itself. Whatever it reports is the method's false-positive floor.
- **N-gram chains** (`ngrams.py`, `zipf chains`) — 2, 3 and 4 grams. Gated on the unigram result
  per the user's instruction, not built alongside it. See RM1 in `ROADMAP_HISTORY.md`.
- **Docs generated from run output rather than typed**: `scripts/build_corpora_doc.py` writes
  `docs/CORPORA.md` from the counting step's metadata, and `scripts/build_artifact.py` renders the
  HTML report from the parquet. No corpus number or result in the docs is hand-transcribed.
- 57 tests. Statistics are checked against closed-form values substituted by hand, not against a
  second copy of the implementation.

### Results from the first full run

- Corpora built: 428,453 words of Claude main-agent prose (128 sessions, 15 projects) and 153,674
  of subagent prose, against 240 million words across four human tiers.
- **735 of 2,355 candidate words** are over-used against all four baselines and well dispersed.
- The **null test floor is 2.71%** (35 of 1,292) against 31.2% in the real run — roughly 11x.
- The headline result is a construction rather than a vocabulary: `let me` at 11,661 per million,
  and every surviving four-gram is a variant of it.
- Of the words predicted in advance, `gap`, `gaps` and `instinct` qualify; `churn` is rejected by
  the dispersion gates despite a very large rate ratio; `imagining` occurs three times, far below
  the floor, so the predicted phrase is not measurable at any chain length.

### Fixed

Four defects found by using the pipeline rather than by testing it. Full write-ups with evidence
in `docs/previous_issues.md`.

- **F1** — stripping an inline code span left its possessive behind, minting a bare `s` at rank 32.
- **F2** — hyphen splitting minted `re` (from `re-run`, `re-read`) as a 797-occurrence "word".
- **F5** — the flat dispersion ceiling was a frequency filter in disguise: Spearman's rho between
  log count and DP is −0.768, and the 0.75 ceiling rejected 72% of words occurring 20–30 times
  against 0% of words occurring over 300 times.
- **F6** — the Dirichlet prior was normalised over the candidate vocabulary instead of the corpus,
  inflating every pseudo-count. It ranked the ordinary phrase *the rest of the* (a 1.2x rate
  ratio) above *let me read the* (roughly 22,000x). Unigrams were barely affected; the n-gram
  tables were invalid and were recomputed.

The common thread, recorded because it will recur: **this project's defects are systematically
shaped like its findings.** Every one of F1, F2 and F6 produced a plausible-looking overused
token. An implausible result checked against the corpus caught what no unit test did.

### Decisions recorded

- **Four reference tiers, not two.** A literature-and-Reddit panel cannot separate style from
  topic, so software vocabulary would have dominated the ranking without being LLM-ish at all.
- **Main-agent replies are the target**; subagent output is a separately labelled stratum, so
  audience effects are not misattributed to the model.
- **No consumer inbox.** Nothing depends on this repo, so per charter §4 a `docs/FEEDBACK.md`
  would be permanently empty.
- **Both dispersion gates are published** rather than replacing the flat one with the better
  statistic, because the flat gate's bias was found after a ranking existed. See RM6 for the
  fixed-in-advance criterion that should settle it.

### Known at scaffold time

- Assistant `thinking` blocks in the local store are signature-only, so reasoning prose cannot be
  analysed. Only `text` blocks carry usable prose.
