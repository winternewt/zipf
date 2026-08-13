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
