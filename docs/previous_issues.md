# Resolved findings

Dogfooding findings resolved **in this repo**, each with its resolution and a code pointer.
Check here before re-investigating a finding that looks fixed.

Ids are never reused. Compute the next id from `dogfooding.md` **and** this file.

---

## F1 — stripping inline code left its possessive behind, minting a fake top-40 word

*found:* first tokenizer run on the real transcript store · *resolved:* `src/zipf/tokenize.py`

Removing an inline code span left the clitic that was attached to it. `` the `ufw`'s default ``
became `` the  's default ``, and the orphaned `'s` was then tokenized as a bare `s`. It reached
**rank 32** in the Claude corpus, above `your` and `was`.

This is worse than a cosmetic defect. The fragment is produced by *code density*, not by prose,
so it concentrates in precisely the corpus that contains the most inline code — the Claude one.
Measured against any human baseline it would have ranked as a massively overused "word" that
nobody writes, and it would have looked exactly like a finding.

**Resolution:** `_ORPHAN_CLITIC` removes an apostrophe-clitic not preceded by a letter, before
word matching. One-letter tokens other than `a` and `i` are dropped as markup artifacts.
Verified: `s`, `t`, `ll`, `d`, `m` all fall to zero occurrences; `a` and `i` survive intact.

## F2 — hyphen splitting minted `re` as a frequent word

*found:* same run, after F1 · *resolved:* `src/zipf/tokenize.py`

With hyphens treated as separators, `re-run` and `re-read` emitted a standalone `re`, which
reached **797 occurrences** — around rank 90. Prefixed compounds are a register habit rather
than a word, so this would have surfaced as a heavily overused token that does not exist as an
independent word in English.

**Resolution:** `_WORD` now joins on `-` and `'` rather than splitting, so `re-run` and
`well-known` are single tokens. `re` falls from 797 to 9 (genuine uses). Type count rises from
10.8k to 15.8k, which is the expected cost of keeping compounds intact.

**What this class of bug teaches, and why both entries are here rather than one:** every
tokenizer artifact so far has been *correlated with code density*, which is the single biggest
difference between this target corpus and every human baseline. That makes tokenizer bugs
systematically indistinguishable from the finding the project is looking for. Any future
tokenizer change gets checked against the top 100 by hand before a result is quoted from it.

## F5 — the dispersion ceiling was a frequency filter wearing a dispersion label

*found:* testing the second reading of F4 · *resolved:* `src/zipf/stats.py::dispersion_excess`

F4 raised two contradictory readings of why `churn` was rejected and noted that only one was
testable: **is DP correlated with target count?** It is, strongly.

Measured over the 2,355 candidate words:

- Spearman's rho between log count and DP: **−0.768**.
- Words occurring 20–30 times: median DP 0.803, **72.1%** above the 0.75 ceiling.
- Words occurring 30–50 times: median DP 0.730, 43.1% above.
- Words occurring 100–300 times: median DP 0.465, 5.1% above.
- Words occurring over 300 times: median DP 0.289, **0%** above.

A rare word cannot spread evenly across 128 sessions however habitual it is, so a flat ceiling
rejects words for being rare and reports the rejection as concentration. It was discarding
exactly the distinctive low-frequency vocabulary the project exists to find.

**Resolution:** `dispersion_excess` compares each word's DP against the median DP of words in
the same log-count band, so the gate asks "more concentrated *than words of its own
frequency*?" rather than "more concentrated than a constant". `compare.py` reports both gates:
`well_dispersed` (flat, primary) and `well_dispersed_conditional` (frequency-neutral).

**Both are published rather than the flat one being replaced**, because the flat gate's bias
was found *after* a ranking already existed. Silently swapping to the gate that admits more
words is indistinguishable from tuning for a nicer result, so the two are reported side by
side and the disagreement is part of the output: 735 words pass the flat gate, 664 the
conditional, 556 both, and they disagree on 287.

**The check that this was not fitted to the hypothesis:** the finding originated in an attempt
to understand why `churn` — a word the user predicted — was rejected. Under the new gate
`churn` is **still rejected** (DP 0.82 against an expected 0.80 for its frequency, excess
+0.019). The repair did not rescue the word that motivated it, which is the evidence that it
corrected a real bias rather than fitting one.

## F6 — the Dirichlet prior was renormalised over the candidate subset, manufacturing z-scores

*found:* adversarial check of an implausible n-gram result · *resolved:* `src/zipf/stats.py`

The 4-gram table ranked **`the rest of the`** first, at z = 75.6. That is ordinary English, and
the claim was checkable against the corpus rather than against intuition: Claude uses it at
75.4 per million and the best reference tier at 62.7 per million. **A 1.2x rate ratio was
scoring higher than `let me read the`, which has a ratio of roughly 22,000x.**

The cause is in the prior. Monroe's `alpha_w = prior_mass * p_background(w)` requires
`p_background` to be the word's rate in the *full* background corpora. The implementation
computed `background / background.sum()`, normalising over whichever words were in the
comparison. That inflates every alpha by `(all tokens / candidate tokens)`, and the factor
explodes as the candidate set narrows:

| stage | candidates | inflation |
|---|---:|---|
| unigrams | 2,355 types covering most tokens | small |
| 4-grams | 347 of 372,588 types | enormous |

For `the rest of the` the pseudo-count reached the hundreds against a real count of 30, so the
prior — not the data — set the target's apparent rate.

**Resolution:** `log_odds_dirichlet` takes `background_total` and uses it as the denominator,
with `alpha_total` becoming the prior mass actually landing on the compared vocabulary rather
than the full `prior_mass`. `compare.py` and `ngrams.py` pass the summed full totals of every
corpus. Regression test: a word at an identical rate in both corpora, with a three-entry
candidate vocabulary, must score |z| < 1; the same call without `background_total` scores more
than ten times higher, so the guard fails if the bug returns.

**Impact, measured rather than assumed.** Unigrams were barely affected — 739 qualifying words
before, 735 after — because 2,355 candidates cover most of the corpus. The n-gram tables were
invalid and were recomputed.

**Why no test caught it:** every unit test used a full vocabulary, where the subset
normalisation is exactly correct. The bug only exists when the vocabulary is a subset, which is
precisely the case the n-gram stage introduced and no test covered. The lesson is the one F1
and F2 already recorded from the other direction: this project's defects are systematically
shaped like its findings, so an implausible *result* is the detector of last resort, and
checking one against the corpus is worth more than re-reading the code.

## F7 — a junk fragment became a lemma and swallowed the word it prefixes

*found:* first folded run · *resolved:* `src/zipf/morphology.py`

The first inflection-folded ranking contained `noth`, `statu`, `sible`, `untouch` and
`unchange` as lemmas. None is a word.

The fold map was built from the **union vocabulary** of every corpus, on the reasoning that a
merge should only happen where the data attests both forms. But attestation in 240 million words
of scraped text is nearly free: the tail contains typos, OCR damage and fragments, so any
three-letter scrap that happens to prefix a real word gets generated into it. `noth` occurs 26
times across the whole union; `nothing` occurs some 200,000 times; the rule folded the second
into the first.

**Resolution:** `build_fold_map` now takes pooled *counts* rather than a bare vocabulary, and
applies two guards — a base must be attested at least 50 times, and a form may be at most 25x
more frequent than its proposed base. Real lemmas are not thousands of times rarer than their
own inflections. Irregular pairs are exempt, because `data` genuinely is far more frequent than
`datum`. Verified: `nothing` and `status` are intact and the five junk lemmas are gone.

The ratio is deliberately loose. `running` genuinely outnumbers `run` in some corpora, and a
guard tight enough to catch every fragment would break correct merges — which is the worse
error, because a missed merge is visible and a wrong one is not.

## F8 — the possessive rule ate a contraction

*found:* checking the fold groups by hand · *resolved:* `src/zipf/morphology.py`

`let's` was folded into `let`. It is not the possessive of `let`; it is `let us`, and it is a
different construction from the one the study is measuring. The `X's` rule cannot tell a
possessive from a contraction of `is`/`has`/`us`, and the corpus's single strongest finding is
built on `let`, so corrupting its count would have hit the headline result.

**Resolution:** common `'s` contractions are added to the protected set. Genuine possessives
like `agent's` still fold. Impact was small here (`let's` occurs twice against `let` at 4,946)
but the defect was in the most load-bearing word in the study.

## F9 — the role label was decided by rule order rather than by evidence

*found:* a test written from the user's own bracket notation · *resolved:* `src/zipf/morphology.py`

`gaps` was labelled `third_person`. It is equally the plural noun. The label was not a judgement
— the verbal rules simply ran after the nominal ones and overwrote the entry in a dict.

The merge itself was never wrong (both roles share the base), so this would never have shown up
as a bad number. It would have shown up as a *confident* number: a report saying `gaps` is a
verb form, with nothing behind that but dict ordering.

**Resolution:** roles are collected as a set, and a form generated by both paradigms is labelled
`plural_or_third_person`. Without a part-of-speech tagger there is no evidence to choose, and
the project's rule everywhere else is that unknown is a value, not a thing to guess. `axis` was
added to the irregulars in the same change, which correctly makes `axes` ambiguous between `axe`
and `axis` and therefore unfolded.
