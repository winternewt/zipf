# Dogfooding — open findings

Layer 2. Findings carry stable `F#` IDs and **move** to `previous_issues.md` when resolved
here — they are never duplicated. Read this before touching the CLI or the statistics surface.

Ids are never reused, not even for a finding closed as a non-issue. Compute the next id from
this file **and** `previous_issues.md`: once resolved items move out, the highest id visible
here is not the highest that has existed.

---

## F3 — the null test's false positives are topic, not noise, and the report does not say so

*found:* first null-calibration run · *status:* open · *class:* **surface it, not fix it**

`uv run zipf calibrate` splits the target corpus in half by session and compares the halves.
It reports **35 of 1292 candidate words** surviving every gate — a 2.71% false-positive floor
against 31.4% in the real run, so the headline signal is roughly 11x the floor and stands.

The floor's *composition* is the finding. The survivors are:

> report, my, mb, human, session, wrong, step, mine, chr, something, annotation, whether,
> restart, was, claim, that's, comment, which, commits, rebuilt, helpers, same, trust, empty,
> came, difference, know, evidence, uncommitted, payload, worth, contradicts, assumed,
> invariant

These are overwhelmingly **project-domain nouns** — `chr`, `annotation`, `payload`,
`invariant`, `uncommitted`, `helpers`, `rebuilt`, `mb`. Two random halves of the session pool
do not contain the same projects in the same proportion, so the residual is topic leakage
between session groups, not statistical noise.

That has a direct consequence the report currently does not state: the same confound is
present in the real ranking, so a **domain noun in the results is far more likely to be a
false positive than a discourse word at the same z-score**. `manifest`, `catalog`, `tier` and
`variant` should be read with more suspicion than `let`, `now`, `rather` or `gap`, even though
the statistics do not distinguish them.

**Why this is surfaced rather than fixed, and why each obvious repair is wrong:**

- *Split by project instead of by session.* Makes the null test honest but does not fix the
  real comparison, and it shrinks the null test to a handful of unbalanced groups — the
  largest project is 32.8% of the corpus, so a project-level split cannot be balanced.
- *Add a software-vocabulary stop-list.* Would suppress the domain nouns, but choosing which
  words count as "domain" is exactly the judgement the measurement is supposed to make. A
  hand-curated exclusion list fitted to the observed output is the pipeline being tuned until
  it agrees with its author.
- *Weight sessions to equalise projects.* Defensible, and it changes what the corpus *is*
  rather than how it is measured — a different research question, not a bug fix.
- *Report a per-word leave-one-project-out stability score.* The most promising option and the
  most expensive. It is `ROADMAP.md` RM3.

The actionable part today is presentational: the report must say that domain nouns carry a
higher false-positive rate than discourse words, and must not present a single ranked list as
though every row carried the same credibility.

## F4 — `churn` is a real effect that the dispersion gate correctly refuses to report

*found:* testing the user's named predictions · *status:* open · *class:* **surface it**

`churn` occurs 58 times per million in the target against 0.8–2.3 in the four baselines — a
25–72x ratio, larger than most of what the report does publish. It is rejected because its
Gries DP is 0.82 against a 0.75 ceiling and it appears in only 10 sessions.

Both readings are defensible and they contradict each other:

- The gate is right. Ten sessions out of 128 is a topic, and the word is being counted as a
  habit because a few conversations were about the same thing.
- The gate is wrong *for rare, distinctive words*. A word used 25 times total cannot be spread
  evenly across 128 sessions no matter how habitual it is, so DP structurally penalises
  low-count words and the ceiling is doing frequency filtering under a dispersion name.

The second reading was testable, and **it has now been tested and confirmed** — Spearman's rho
between log count and DP is −0.768, and the repair is recorded as F5 in `previous_issues.md`.

**That did not change this finding's verdict.** Under the frequency-neutral gate `churn` has a
DP of 0.82 against an expected 0.80 for words of its frequency, an excess of +0.019 — still
rejected, now for a reason that survives the frequency confound. The first reading is
therefore the right one: ten sessions out of 128 is a topic, not a habit.

This entry stays open because the *reporting* question is unresolved, not the statistical one.
`churn` has a 25–72x rate ratio, larger than most of what the report publishes, and suppressing
it entirely tells the reader less than showing it with its rejection reason. The open question
is whether the report should carry a "high ratio, insufficient spread" section rather than a
single pass/fail list — which is a presentation decision, and the same one F3 raises.

## F10 — the log-odds z collapses exactly where the effect is largest

*found:* adding the commits tier dropped qualifying 4-grams to zero · *status:* open ·
*class:* **surface it, not fix it**

Adding a sixth corpus took the four-word chains from 13 qualifying to **none**, which looked at
first like the new tier doing its job. It is not. The mechanism is in the statistic.

Monroe's variance is `1/(y_target + a_w) + 1/(y_reference + a_w)`. The prior mass is 5,000
pseudo-tokens spread across a background of ~253 million, so a rare chain gets
`a_w ≈ 0.006`. When a chain occurs **zero** times in a reference corpus, the second term
becomes `1/0.006 ≈ 170`, the standard error swamps the effect, and the z-score collapses toward
1 no matter how extreme the rate difference is.

Measured, on real chains:

| chain | Claude /M | closest human /M | ratio | z_min | tiers agreeing |
|---|---:|---:|---:|---:|---:|
| `let me` | 11,661 | 226 | 52x | **19.0** | 6 of 6 |
| `let me check` | 1,666 | 0.31 | ~5,400x | **1.6** | 3 of 6 |
| `let me verify` | 953 | 0.03 | ~27,800x | **1.2** | 2 of 6 |
| `let me read the` | 751 | 0.03 | ~22,000x | **1.0** | 3 of 6 |

The ordering is inverted: **the more distinctive the phrase, the worse it scores.** Single words
are unaffected, because any word frequent enough to be a candidate is attested in every corpus.
This bites only where the reference count is zero, which is precisely the regime that phrase
evidence lives in.

**Why this is surfaced rather than fixed, and why each obvious repair is wrong:**

- *Raise `prior_mass` for the n-gram stage.* It would work, and it is the single most suspect
  change that could be made here, because the defect was found by noticing which phrases a
  larger prior would rescue — all of them `let me` variants, the project's headline result.
  Tuning a prior until it certifies the author's favourite finding is indistinguishable from
  fitting, whatever the justification attached.
- *Use the ratio instead.* A ratio against a smoothed zero is a statement about the smoothing
  constant, not about the corpus. `let me verify` at "27,800x" is really "not once in 253 million
  words", which is worth saying in words but is not a measurement.
- *Drop the min-across-corpora rule for phrases.* It would certify them, by abandoning the one
  rule that makes the unigram result trustworthy. Two standards, one report.
- *Switch to an exact test.* The right answer, and a real piece of work: for zero-count
  references the honest statistic is a one-sided Poisson or binomial bound on the rate ratio,
  which stays finite and orders these chains correctly. It is a different estimator with a
  different interpretation, so it needs its own METHODOLOGY entry and its own null calibration.

Until then the report states the limitation in the Limits section and declines to claim the
four-word chains, while still reporting `let me` itself, which clears all six corpora on its own
evidence at z 19.

## F11 — the instrument turned on its own author, from a cartoon

*found:* a user-drawn cartoon labelling a dinner table with the assistant's own tics ·
*status:* open · *class:* **observation, with one real limitation**

The charter's standing probe is "turn the tool on the work you just did". The occasion arrived
unprompted: a drawing of *Full Picture Restaurant*, in which every object is captioned with a
phrase from these very sessions — `load-bearing sauce`, `freshly ground truth`,
`coarse-grained salt`, `a genuinely sharp point`, `all green`, `two sides (and it's worth
separating them)`.

Measured against the built corpora, the caricature is **about half accurate**, and the half it
gets right it gets right by a wide margin.

Landed, against pooled general English:

| term | Claude /M | general English /M | ratio |
|---|---:|---:|---:|
| `load-bearing` | 126 | 0.2 | **597x** |
| `all green` | 110 | 0.87 | **126x** |
| `full picture` | 91 | 0.83 | **110x** |
| `deliberately` | 364 | 10.0 | 36x |
| `genuinely` | 483 | 14.3 | 34x |
| `a genuine` | 105 | 6.6 | 16x |
| `intentional` | 56 | 4.9 | 12x |

Invented by the artist — these do not occur in the corpus at all: `coarse-grained` (0),
`substrate` (0), `with teeth` (0), `belt and suspenders` (0), `genuinely sharp` (0). And
`teeth` runs at 9 per million against 50 in general English: **used five times less** than
humans use it.

**The limitation this exposes.** A caricature is a hypothesis, and half of this one was wrong in
a direction no reader could have detected by intuition — the invented terms *feel* exactly as
characteristic as the real ones. That is the project's premise inverted, and it is the argument
for the measurement rather than a footnote to it.

**The open question.** In the 6,079 tokens of the authoring session itself that had reached disk,
`genuinely` ran at **1,152 per million** against 483 corpus-wide, and `honest` at 494 against
219. A single small sample proves nothing, but if the rate of these words varies by session or
drifts over time, the corpus-wide figure is an average over a moving target and every reported
rate is a smear rather than a value. Testing it needs per-session rates plotted against session
date, which the harvested data already supports and nothing currently does.
