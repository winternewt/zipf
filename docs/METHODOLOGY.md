# Methodology

What each number means, what it is blind to, and which confound it controls for. Read this
before changing anything that ends up in a result: a metric changed without its doc is a claim
that moved silently.

---

## 1. The question

Which words does Claude Code use, in prose addressed to a user, at a rate that human writers
do not?

That question is narrower than it sounds, and the narrowing is the whole design:

- **Prose, not output.** Code, paths, URLs and markup are removed before counting. Otherwise
  the comparison is dominated by whichever corpus contains more source code, which is a
  difference of topic and format, not of style.
- **Rate, not presence.** Every word here is an ordinary English word. The claim is never
  "Claude uses a word humans don't"; it is "Claude uses it several times more often".
- **Aggregate, not per-document.** This is not a detector. There is no verdict on any single
  text, and the method cannot produce one.

## 2. Why six baselines

A single baseline cannot separate *style* from *topic*. Compare assistant prose to Victorian
novels and the top of the list is `file`, `function`, `commit` — true differences, and
completely uninformative, because they describe what the text is about rather than how it is
written.

Each tier removes a different confound, and only a word that clears all six is reported:

| Tier | Controls for | Blind to |
|---|---|---|
| `literature` | modern register generally | topic, era, medium — it is maximally distant |
| `reddit` | edited-vs-unedited writing | subject matter; almost no software prose |
| `technical` | **topic** — same subject, same markdown, same fenced code | its own register habits (Q&A voice, terse answers) |
| `web` | any single-source quirk; broad written English | nothing in particular, which is why it is the control on the other three |
| `biomedical` | **the target's own subject matter** — PubMed abstracts, so bioinformatics vocabulary stops scoring | anything outside biomedicine; a private vocabulary no public corpus contains |
| `commits` | **software-collaboration register** — commit messages, where `commit`, `bump`, `revert`, `stale` and `upstream` live at their natural rate | issue and pull-request *discussion*, which is longer-form and not covered here |

Two tiers carry most of the weight, and they control different things.

`technical` is StackOverflow: humans writing prose *about code*, in markdown, with fenced
blocks — the same shape as the target corpus. But it is Q&A about broken code, which is not the
same register as software *work*. Its vocabulary is `error`, `function`, `exception`; it is thin
on `commit`, `bump`, `revert`, `stale`, `upstream`.

`commits` closes exactly that gap. Commit messages are the register in which engineering work
is narrated, and they are where the vocabulary of version control lives at its natural rate.
Without this tier, ordinary developer register scores as assistant style: a word like `commit`
is over-used against Victorian novels, Reddit and PubMed for the obvious reason, and
StackOverflow does not rescue it. **This tier was added specifically to test whether the
project's own top results were measuring a habit or a job.**

**The ranking statistic is the minimum z across tiers, not the mean.** A mean lets one extreme
baseline carry a word; `commit` scores enormously against Gutenberg and near zero against
StackOverflow, and averaging would promote it. Taking the minimum states exactly the claim
intended: over-used against *every* human baseline tried, including the two that share its
subject matter.

## 3. The statistics

### 3.1 Log-odds with an informative Dirichlet prior — the headline number

Monroe, Colaresi & Quinn (2008), *Fightin' Words*. For each word:

```
delta = log( (y_t + a_w) / (n_t + a_0 - y_t - a_w) )
      - log( (y_r + a_w) / (n_r + a_0 - y_r - a_w) )

var   = 1/(y_t + a_w) + 1/(y_r + a_w)
z     = delta / sqrt(var)
```

`y` are counts, `n` corpus totals, `a_w = prior_mass * background_w / sum(background)` and
`a_0 = prior_mass`.

**Why not a frequency ratio.** A ratio carries no variance. A word seen three times in the
target and once in a baseline scores 3x and outranks a word seen thirty thousand times against
ten thousand. Ratios sort hapax noise to the top. The z-score divides by the uncertainty, so
rare words have to be *much* more skewed to rank.

**Where the prior comes from, and why it matters.** The background is the pooled counts of the
target and every reference tier. It must never be derived from the target alone: a prior taken
from the corpus being tested makes the test compare a convention against itself, and it agrees
perfectly. `prior_mass` defaults to 5000 pseudo-tokens, roughly 1% of the target's size —
enough to tame hapaxes, small enough to leave real effects intact.

**Blind to:** everything about *where* in the corpus the occurrences fall. A word produced
entirely by one long session gets exactly the same z as one spread evenly. That gap is what
§3.3 exists to close.

### 3.2 Dunning's G² — the independent cross-check

The standard corpus-linguistics keyness test, signed by direction. It rests on different
assumptions from the log-odds z (a likelihood-ratio test on the contingency table, with no
prior at all). It is **not** a second opinion from the same method, which is the only reason
including it is worth anything.

Where G² and the log-odds z disagree in sign, neither is reported. That is a deliberate loss of
recall in exchange for not shipping a word two methods cannot agree about.

**Blind to:** the same thing the log-odds is blind to, plus it is well known to over-reward
very high-frequency words. It is used only as an agreement filter, never as a ranking.

### 3.3 Gries' Deviation of Proportions — the dispersion gate

```
DP = 0.5 * sum_i | v_i/v  -  s_i/s |     normalised by (1 - min_i s_i/s)
```

`v_i` is the word's count in part `i`, `s_i` that part's token total. 0 means the word is
spread exactly in proportion to part sizes; 1 means every occurrence sits in one part.

For the target corpus a **part is a session**. This is the gate that separates a habit from one
long conversation about one thing. A domain noun from a single bioinformatics task has a large
z-score and a DP near 1; a genuine verbal tic is spread across most sessions.

Reported and gated alongside it:

- **`sessions_present`** — how many sessions contain the word at all.
- **`max_session_share`** — the largest fraction any one session holds. A word can be present
  in many sessions and still be 80% one of them; DP and share fail differently, so both gate.

A word with **no** occurrences gets `nan`, not 0. "No occurrences" has no dispersion, and
returning 0 would read as "perfectly even" — the exact inversion of the truth.

### 3.4 The session-level bootstrap — the honest interval

Sessions are resampled with replacement 500 times and the 5th percentile of the resulting
z-distribution is reported.

**Resampling is over sessions, not tokens, and this is not a detail.** Tokens within a session
are heavily correlated: an author who writes "gap" once writes it again three lines later. A
token-level bootstrap treats those as independent observations, produces intervals several
times too narrow, and confidently certifies single-session artifacts. The unit of resampling
has to be the unit of correlation.

### 3.5 Zipf fit — a corpus sanity check, not evidence

Least-squares slope of log-frequency against log-rank. A natural corpus sits near −1. It says
nothing about any word; it is a check that a corpus has not been filtered, deduplicated or
truncated in a way that would also distort the comparison.

## 3.6 Morphological folding

`gap`, `gaps`, `gap's` and `gapped` are one habit counted four times. Unfolded they compete for
rank, they split the evidence for a word that is individually rare, and they fill the report
with near-duplicates.

Folding is **generate-and-verify**, not strip-and-hope. A stemmer chops suffixes, over-merges
(`verification` -> `verif`) and labels groups with non-words. This runs the other way: from a
candidate base it generates the forms English inflection would produce and keeps only those
**attested in the corpora**, so every group is labelled by a real word and no merge happens
that the data does not witness.

Three layers, cheapest first:

| level | merges | note |
|---|---|---|
| `none` | nothing | exact types |
| `nominal` | plural, possessive | the safest paradigm |
| `inflection` | + third person, past, progressive | the default |

**Where it stops, and why.** Derivational relatives (`verify` / `verification`) change part of
speech and often meaning, so merging them is a research decision rather than a normalisation.
`agape` is not a form of `gap` in any sense a frequency table should act on. Both are excluded.

Two guards do most of the work, and both exist because of observed failures:

- **A base must be frequent enough to be a word.** 240 million words of scraped text contain an
  enormous tail of typos and OCR damage, and any three-letter fragment that prefixes a real word
  will be generated into it. `noth` + `ing` swallowed `nothing`; `statu` + `s` swallowed
  `status`. A form may be at most 25x more frequent than its proposed base.
- **Ambiguity is left unfolded.** `axes` is the plural of both `axe` and `axis`, so neither
  claims it. A wrong merge is silent; a missed merge is not.

Roles keep the same three-valued discipline as everything else: `-s` is genuinely both a plural
and a third-person marker, and without a part-of-speech tagger there is no evidence to choose,
so the role is recorded as ambiguous rather than decided by whichever rule ran last.

**Blind to:** part of speech entirely. A noun and a verb sharing a lemma are merged, so
`meeting` (the event) folds into `meet`. At this corpus size that is the right trade, but it is
a trade.

## 3.7 Domain vocabulary, and the yardstick it moves

The target corpus is bioinformatics-heavy. `annotation` runs at 397x general-English rate,
`chromosome` at 121x. These are real differences and they are not style — they are what the
sessions were about. They do two separate kinds of damage, which need two separate corrections.

**They occupy the ranking.** The fix is architectural rather than statistical: add a baseline
that *shares the confound*. A biomedical tier (PubMed abstracts, oldest PMIDs) makes domain
nouns fail the minimum-across-tiers gate structurally, without anyone hand-writing a list of
which words count as jargon — which would be the measurement being tuned until it agreed with
its author.

**They move the yardstick.** A corpus with a heavy topical component has a wider spread of
log-odds, so a z of 3 does not mean here what it would mean in a topic-matched comparison.
:func:`domain.empirical_threshold` reads the threshold off the *null* distribution — the corpus
compared against itself, where every value is a false positive by construction. On this corpus
it gives **z >= 4.69** at a 1% false-positive rate against the conventional 3.00, so the
constant was 56% too lenient.

Two instruments interpret the result, and they fail differently:

- **Specialisation** — `log2(specialist rate / general rate)`, where specialist is biomedical
  plus StackOverflow and general is literary plus Reddit plus web. **Computed from the reference
  corpora only.** A domain score derived partly from the corpus under test would be the
  measurement checking itself, and every domain word would helpfully confirm it was one.
- **Project dispersion** — Gries DP over the 15 projects rather than the 128 sessions. This is
  the only instrument that sees a *private* vocabulary, one no public corpus contains and which
  specialisation therefore cannot score at all.

## 3.8 The version-control instrument

Two questions look alike and are not. *Is this word part of the register of software work?* is
answered by a corpus of that register — commit messages — used as a baseline in the ordinary
way. *Is this word simply the subject matter of version control?* is not, because commit
messages use git's vocabulary without explaining it, at rates set by what people happen to be
committing.

The documentation answers the second question directly: Pro Git plus git's own reference manual,
315,979 tokens of prose. A word is set aside as version-control vocabulary when **both** hold:

1. the manual uses it at least four times more often than general English does — it is
   *distinctive* to writing about version control; and
2. Claude does not out-use the manual, its rate being under twice the manual's.

**Both conditions are load-bearing.** Without the first, the filter flags any word the manual
uses at a similar rate to Claude, which is most ordinary words, because git documentation is
dense technical prose — the first version removed `the`, `one` and `only`. Without the second, a
word the manual mentions once and Claude says constantly would be dismissed.

**Why this is a rate comparison and not a seventh baseline.** At 316k tokens the corpus is far
too small for the minimum-z rule. A candidate word it happens never to use would have its
z collapse through the zero-count mechanism in F10, and the word would drop for lack of evidence
while appearing to have been controlled for. A rate ratio has no such failure mode: a word the
manual never uses scores infinity, which is exactly the right verdict.

**Blind to:** anything git's documentation does not cover. Vocabulary from other tools —
containers, CI, package managers — is not tested by this instrument at all.

## 4. What the method is blind to

Stated plainly, because a check that cannot run is not a check that passed:

- **Topic inside the target.** The Claude corpus spans a handful of projects, several
  bioinformatics. Dispersion controls for single-session artifacts but not for a word common to
  several sessions *because they share a domain*. See `ROADMAP.md` RM3.
- **Corpus size asymmetry.** The reference tiers are two orders of magnitude larger than the
  target. The Dirichlet prior handles the variance of the estimate; it does not make vocabulary
  sizes comparable, so rank-based comparisons across corpora are not directly interpretable.
  See RM2.
- **Prompt effects.** Claude Code's output is shaped by a system prompt, by tool results, and
  by the user's own phrasing. This method cannot distinguish a model's habit from a habit the
  harness induced, and nothing here should be read as a claim about the model in general.
- **Register beyond the four tiers.** No baseline here is *this user's own technical prose*,
  which would control for topic, domain, era and register at once. It is the most informative
  missing tier. See RM4.
- **One author's transcripts.** The target is one person's sessions on a handful of projects.
  Whether these rates hold for Claude Code generally is untested and unclaimed.

## 5. Contamination, and which way it pushes

Every human corpus dated after roughly 2022 contains some generated text. Each tier here is
dated before that: Gutenberg is pre-1930, Reddit is 2010-2012, StackOverflow is filtered to
before 2022, the Common Crawl snapshot is January 2021.

Where contamination exists it makes a baseline **more** Claude-like, which **shrinks** the
measured gap. The bias runs toward the null, so a surviving finding is conservative. This is
the good direction, but it must be stated rather than assumed — an unstated bias is
indistinguishable from an unexamined one.

The literary tier carries the opposite risk and it is worth naming: archaic vocabulary makes
ordinary modern words look novel. That tier over-reports, which is precisely why a word must
clear all six rather than any one.

## 6. The null test

`uv run zipf calibrate` splits the target corpus in half **by session** and runs the full
comparison of one half against the other. Same author, same models, same register, largely the
same topics — so the honest answer is that nothing is over-used.

Whatever that test reports is the method's false-positive floor. If it returns a long list,
every number the pipeline produces is worthless, and **no unit test would have said so**: the
code would be working exactly as written and the claim would still be wrong.

The split is by session rather than by document on purpose. Documents within a session share a
topic, so a document-level split leaks the same conversation into both halves and makes them
look far more similar than two independent samples — the test would pass for the wrong reason.

## 7. Decisions that were answered rather than guessed

Recorded here because an answered question that is not written down gets re-asked:

- **Four tiers, not two.** A literature-and-Reddit panel cannot separate style from topic;
  software vocabulary would have dominated the ranking without being LLM-ish at all.
- **Main-agent replies are the target; subagent output is a separate labelled stratum.** Same
  models, different audience. Pooling them would misattribute audience effects to the model.
- **Unigrams first; n-grams gated on the unigram result.** Phrase-level tells are the
  motivating example, but the extension is expensive and only worth building once the unigram
  signal is shown to be real. See RM1.
