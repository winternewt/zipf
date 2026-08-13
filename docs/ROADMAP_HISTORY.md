# Roadmap history

Shipped items relocated from `ROADMAP.md`, each with the rationale it carried. Nothing is
deleted here.

---

## RM1 — Extend to 2/3/4-gram chains

*shipped* · `src/zipf/ngrams.py`, `uv run zipf chains`

**The gate that had to open first.** The user's instruction was *"start with 1-word, plan
1-2-3-4 chains extension once you see stat difference in 1-word."* The gate was a unigram run
showing a real, dispersion-robust separation against at least three of four tiers. It opened
on a run finding 31.4% of candidate words over-used against a 2.71% null-test floor — roughly
11x — so the phrase stage was built.

**What it produced.** 688 two-word chains, 140 three-word and 13 four-word chains clear every
gate. The result is more concentrated than the unigram one: every surviving four-gram is a
variant of the same construction (`let me read the`, `let me check the`, `let me verify the`,
`let me look at the`), which reframes the headline finding. The strongest signal is not a
favourite word but a sentence opening.

**The three design questions logged when the item was written, and how each resolved:**

- *N-gram tables are one to two orders of magnitude larger than unigram tables; streaming
  aggregation is likely required.* It was not. Counting the target exhaustively and reducing
  each reference tier to a **candidate set** of the target's frequent n-grams bounded memory
  without touching any rate, because the denominator is still counted over the full stream.
- *The informative prior needs a background n-gram distribution, which is expensive.* It is
  cheap under the candidate-set approach — but this is exactly where the stage's one serious
  defect lived. See F6 in `previous_issues.md`: normalising the prior over the candidate subset
  rather than the corpus inflated every pseudo-count, and ranked the ordinary English phrase
  *the rest of the* above *let me read the*. The fix was to pass the true background total.
  The unigram stage had the same bug and was barely affected, because its candidate set covers
  most of the corpus; the narrower the candidate set, the worse the inflation.
- *Overlapping n-grams are not independent observations, so z-scores will be optimistic and the
  report must say by roughly how much.* Partly discharged. The report and the artifact both
  state that these z-scores are an ordering rather than calibrated significance. Quantifying
  the inflation was not done and is now RM5.

## RM3 — Per-project topic confound

*shipped* · `src/zipf/domain.py`, `src/zipf/corpora.py` (biomedical tier), `uv run zipf domain`

The item was promoted to high severity by evidence: the null test's own false positives were
almost entirely project-domain nouns, so the 2.71% floor was topic leakage rather than
statistical noise, and the same confound sat in the real ranking.

**What shipped, and why in this order.** The item proposed leave-one-project-out as the control.
That was not the cheapest correct answer. Two others came first:

- **A baseline that shares the confound.** PubMed abstracts as a fifth tier. Because the ranking
  statistic is the minimum z across tiers, a word common in biomedical writing now fails
  structurally — no list of jargon is written by anyone, which matters because hand-curating one
  is the measurement being tuned until it agrees with its author. Effect alone: 735 -> 666.
- **Specialisation**, `log2(specialist rate / general rate)`, computed **from the reference
  corpora only**. An external yardstick can interpret the ranking without contaminating it.

**The second half of the item turned out to be the more important half.** Domain vocabulary does
not only occupy the ranking, it widens the spread of log-odds, so a fixed threshold means
something different here than in a topic-matched comparison. Reading the threshold off the null
distribution gives z >= 4.69 at a 1% false-positive rate against the conventional 3.00 — the
constant was 56% too lenient, and no amount of removing domain words would have fixed that.

**Measured outcome.** After the biomedical tier, folding and recalibration, 352 of the original
735 words survive as style vocabulary. Domain words now have a slightly *lower* median z than
style words (1.32 against 1.40), which is the sign that the skew is absorbed rather than merely
labelled.

**Left open:** leave-one-project-out stability, which remains the only instrument that would
catch a word carried by one project *and* attested in the reference corpora. Project dispersion
is a cheaper approximation of it and is now reported per word.
