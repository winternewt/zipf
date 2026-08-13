# Roadmap

Active-only, forward-only. One `## RMn — name` per open item. Shipped items move to
`ROADMAP_HISTORY.md` with their rationale; nothing is deleted, only relocated.

---

## RM2 — Corpus size asymmetry is uncontrolled

*severity:* medium · *status:* open · *owner:* unassigned

The Claude corpus is 428,453 words; each reference tier is 60 million. The Dirichlet prior
handles this for the *variance* of the estimate, but the effective vocabulary sizes differ
enough that rank-based comparisons are not directly interpretable. Decide whether to subsample
reference tiers to a matched size (loses power, gains interpretability) or to report rank
statistics only within matched frequency bands.

## RM4 — No human-written control from the same author

*severity:* medium · *status:* open · *owner:* unassigned

The strongest possible baseline is the *user's own* prose on the same topics in the same repos
— commit messages, issues, README text in the harvested projects. It controls for topic,
domain, era and register simultaneously, which no public corpus does. Cheap to build and the
most informative tier available; it was not in the agreed four-tier panel, so it goes here
rather than into the build.

## RM5 — N-gram z-scores are optimistic by an unquantified amount

*severity:* medium · *status:* open · *owner:* unassigned

Overlapping n-grams are not independent observations: `a b c` contributes to two bigrams that
share a token. The reported z-scores therefore overstate significance. The report and the
artifact both say so, which discharges the honesty requirement but not the measurement.

A session-level bootstrap over n-grams would give an empirical answer, since it resamples the
correlated unit directly. That machinery already exists for unigrams (`stats.bootstrap_z`) and
was not wired into the n-gram path, which currently reports no bootstrap column at all.

## RM6 — Two dispersion gates are published and neither is chosen

*severity:* low · *status:* open · *owner:* unassigned

`compare.py` reports `well_dispersed` (flat DP ceiling) and `well_dispersed_conditional`
(frequency-neutral, per F5). They disagree on 287 words: 735 pass the flat gate, 664 the
conditional, 556 both. Publishing both was the right call *at the time*, because the flat
gate's bias was discovered after a ranking already existed and silently switching to the more
permissive gate is indistinguishable from tuning for a nicer result.

That justification expires. The conditional gate is the better statistic on its merits and
should become primary once the choice can be made on evidence rather than on sequence — most
cleanly by checking which gate gives the lower null-test false-positive floor, which is a
criterion fixed in advance and independent of which words either admits.

## RM7 — Derivational folding is not implemented

*severity:* medium · *status:* open · *owner:* unassigned

Folding stops at inflection. `verify`, `verified` and `verifying` are one entry;
`verification` is a separate one, and both appear in the results. The user's own ladder put this
in the outer bracket, and it is a real level: `verify` at 2,684 per million plus `verification`
at 371 is one habit reported twice.

It is left open rather than built because derivation changes part of speech and often meaning,
so it is a research decision rather than a normalisation. `agape` marks where the ladder must
stop entirely; `verification` sits in between and needs an explicit rule about which suffixes
preserve enough meaning to merge — `-ion`, `-ment`, `-ness` probably; `-er`, `-able`, `-ly`
probably not.

## RM8 — No part-of-speech information anywhere

*severity:* low · *status:* open · *owner:* unassigned

`-s` is both a plural and a third-person marker, and the folder records the role as ambiguous
because there is no evidence to choose. That is the honest handling, but it means noun and verb
senses of one lemma are merged: `meeting` the event folds into `meet` the verb.

A tagger would resolve it, at the cost of a dependency and a per-token pass over 273 million
words. Worth doing only if a result turns out to hinge on a noun/verb distinction; nothing in
the current ranking does.

## RM9 — No issue or pull-request discussion corpus

*severity:* medium · *status:* open · *owner:* unassigned

The `commits` tier covers the terse imperative register of commit messages. It does not cover
the *discussion* register — issue threads, pull-request review, design argument — which is much
closer in shape to what the target corpus actually is: extended prose explaining engineering
decisions to a human.

The obvious source, `bigcode/the-stack-github-issues`, is a **gated** Hugging Face dataset and
returns 401 without accepting its terms and supplying a token, so it could not be built here.

The alternatives found were rejected rather than substituted, and the reason matters:
`Dahoas/code-review-instruct-critique-revision` and its siblings contain **model-generated**
critiques. Using them as a human baseline would import generated text into the reference side,
which is the one contamination this design cannot tolerate — it does not merely bias toward the
null, it makes the baseline partly the thing being measured.

Filling this needs either the gated dataset with credentials, or a fresh scrape of permissively
licensed issue threads with a pre-2022 cutoff.
