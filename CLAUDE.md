# Agent Guidelines — zipf

`zipf` is a research pipeline that measures which words Claude Code overuses. It harvests the
prose half of local Claude Code transcripts, tokenizes it identically to four human-written
reference corpora, and ranks the vocabulary by how far its rate departs from human baselines.
It is an **end-user research application**, not a published library: nothing imports it, so it
carries no API contract, no curated `__all__`, and no consumer inbox.

`AGENTS.md` is a symlink to this file. If the two ever differ, that is a bug —
`ln -sf CLAUDE.md AGENTS.md`.

---

## Read these first, in this order

Obligatory. Read them yourself. **Do not delegate a document you are about to judge a design
against** — a subagent returns a summary, and a summary of a rule drops the qualifier the
decision turned on. Delegation is for finding things, never for deciding them.

1. **docs/ROADMAP.md** — active-only, forward-only. One `## RMn — name` section per *open*
   item with a severity/status/owner line.
2. **docs/CHANGELOG.md** — what actually shipped, newest first.
3. **docs/dogfooding.md** — open findings from using the shipped surface for real work. Read
   it before touching the CLI or the stats surface.
4. **docs/METHODOLOGY.md** — what each statistic means, what it is blind to, and which
   confounds the design controls for. **Read it before changing any number that ends up in a
   result.** A metric changed without reading it is a metric whose claim silently changed.
5. **docs/CORPORA.md** — provenance, licence, date cutoff and known contamination of every
   reference corpus.

Everything below is self-contained: no rule here requires following a link to know what you
must not do. Links carry positive detail only.

This repo ships no `.claude/` assets — no skills, agents or workflows. If any are added, name
them here or they get discovered by accident.

---

## 1. Adopting these guidelines: ask, never infer

**When two rules conflict — this file against a sibling repo's, this file against the user's
global preferences, a rule against what the code actually does — stop and run a
questionnaire. Do not pick the one that looks better, do not synthesize a compromise, and do
not silently follow the more specific file.**

How to run one:

1. **Survey first, ask second.** Read the conflicting rules in full and find out *why* each
   side adopted its version. A question that does not carry the reason is unanswerable.
2. **One question per contradiction, batched** — do not drip-feed. Two to four concrete
   options each, never an open prompt.
3. **Each option states its cost**: what breaks, what it forces elsewhere, which existing rule
   it contradicts.
4. **Recommend one and say so.** A questionnaire with no recommendation offloads work the
   survey already did.
5. **Record the answer where it will be read again** — the rule into the relevant section
   below, the reasoning into §10 in the user's own words. An answered contradiction that is
   not written down gets re-asked, which is worse than having guessed.

---

## 2. Non-negotiables

Read the whole list before the first edit. The reason follows each one, because a rule
without its reason gets rationalised away at 2 a.m.

### Environment and tooling

- **Never `uv pip install`.** Use `uv sync` / `uv add` / `uv add --dev`. `uv pip install`
  writes into the venv without touching `pyproject.toml` or the lockfile, so the next clean
  checkout silently lacks the dependency.
- **Never call bare `python` / `python3`.** Always `uv run python …`, `uv run pytest …`.
- **Never hardcode a version string.** It comes from `pyproject.toml` via
  `importlib.metadata.version("zipf")`. Two sources of truth drift, and the one you read is
  the wrong one.
- **Never use a placeholder path or a fabricated example value** in committed code. In this
  repo that has a sharp edge: **never write an invented word-frequency number, z-score or
  example word into docs or code.** Every number in `docs/` is copied from a real run. A
  fabricated "example" result is indistinguishable from a finding and outlives the session
  that invented it.
- **Never commit large data.** No parquet/zst/7z corpus files. Corpora live under `data/`,
  which is git-ignored. Anything over ~5 MB that genuinely must travel goes through Git LFS.
  Note the history gotcha: a blob committed *before* `git lfs track` stays in every past
  commit even after the pointer replaces it at HEAD, so the pack still ships it. Surface it;
  the remediation is the user's to run.
- **Never run tree operations.** No `git push`, tags, releases, branch management or history
  rewriting — the user's domain. **Never `git stash drop` / `git stash clear`**, even on
  explicit request. **Never `git add -A` or `git add .`** — stage explicit paths. Commit only
  when asked.

### Correctness of the measurement

- **Never compare corpora tokenized by different code paths.** Every corpus — Claude Code
  replies and all four references — goes through the single tokenizer in
  `src/zipf/tokenize.py`. A baseline that keeps code blocks while the target strips them does
  not measure style, it measures the tokenizer, and the difference lands in the results
  looking exactly like a finding.
- **Never report a raw frequency ratio as evidence of overuse.** A ratio has no variance
  attached, so a word seen three times outranks a word seen three thousand times. Use the
  log-odds-with-informative-Dirichlet-prior z-score; report the ratio only as colour beside it.
- **Never report a word whose count is concentrated in one session or one project.** A term
  from a single long task looks exactly like a stylistic tic in aggregate. Every reported word
  carries its dispersion (Gries DP), its session count, and its largest single-session share.
- **Never fill a value from the same source that checks it.** The reference corpora exist *to
  be an independent yardstick*. Deriving a stop-list, a vocabulary filter, or a smoothing
  prior from the Claude corpus and then measuring the Claude corpus against it makes the check
  compare a convention against itself, and it agrees perfectly. Priors come from the pooled
  background, filters from external lists.
- **Never treat reproducibility as correctness.** A pinned seed, a stable hash or a green
  `--strict` run means the pipeline is *reproducible*, not that the number means what you
  think. They have no opinion on whether the value names what you meant.
- **Never present a result without its contamination caveat.** Every human corpus after
  roughly 2022 contains LLM-generated text. That biases every finding toward the null
  (baselines drift toward Claude), so a surviving finding is conservative — but the direction
  must be stated, not assumed known.
- **Never silently fall back when a corpus is missing.** If a reference tier failed to build,
  refuse with an explicit error or label the output prominently with which tiers are missing.
  A three-tier result presented as four-tier is a false claim, and the reader cannot see it.

### General engineering

- **Never nest a `try`/`except` inside another.** It hides the real error. Let typed
  exceptions propagate; wrap only where there is a genuine recovery path.
- **Never collapse "unknown" into a boolean.** See §5.
- **Never mock the transformation under test.** See §6.
- **Never claim a test "would have caught" a bug without first running it against the buggy
  code and watching it fail.**
- **Never route around a capability the pipeline lacks while dogfooding.** See §7.
- **Never resolve a contradiction between two rules by inference.** Run §1.

---

## 3. Repository layout, data and assets

```
src/zipf/             source (src layout, uv_build)
tests/                pytest suite
docs/                 all markdown except this file and README.md
assets/               small real fixtures that MUST travel — committed
data/input/           downloaded raw corpora        ─┐
data/interim/         tokenized token streams        ├─ git-ignored, never travels
data/output/          frequency tables and results  ─┘
scripts/              one-off operational scripts, not importable code
```

- `data/` is git-ignored by ignore-all + allowlist. To commit another subtree, add explicit
  `!<dir>/` and `!<dir>/**` lines.
- Committed test fixtures live in `assets/`, not `data/`. If a test writes, it writes to a
  `tmp_path` or a `platformdirs`-resolved cache dir, **never into the project tree**.
- Never hardcode a platform-specific cache path; resolve it at runtime.
- The Claude Code transcript store (`~/.claude/projects`) is **read-only input**. Never write
  to it, never move or rewrite a transcript. It is the user's session history and the only
  copy.

---

## 4. Build, run, test

```bash
uv sync                            # install
uv run pytest -vvv                 # the suite; -vvv when diagnosing
uv run zipf --help                 # every subcommand
```

Pipeline order — each step reads the previous step's output from `data/`:

```bash
uv run zipf harvest                # ~/.claude/projects  -> data/interim/claude_*.parquet
uv run zipf fetch --tier all       # reference corpora   -> data/input/
uv run zipf tokenize --corpus all  # raw text            -> data/interim/*.parquet
uv run zipf count --corpus all     # token streams       -> data/output/counts_*.parquet
uv run zipf compare                # counts              -> data/output/overuse_*.parquet
uv run zipf report                 # results             -> data/output/report.md
```

**Every CLI loads `.env` via `python-dotenv` at startup** before reading configuration. New
configurable values are read from env vars with sensible defaults, documented in
`.env.template`, and mentioned here. `HF_TOKEN` is optional and only raises rate limits.

**Timestamps: store ISO-8601 UTC, display local.** Never write a naive
`YYYY-MM-DD HH:MM:SS` — it is misparsed as local time and breaks string comparison against
ISO values. JSON output, parquet columns and operator logs stay ISO.

**Long downloads print their target URL and byte total before starting**, so an interrupted
run can be diagnosed without re-running it.

---

## 5. Coding standards

- **Type hints mandatory** (3.12 syntax). **`pathlib.Path`** for every path.
- **Absolute imports only**, and **no inline imports** — every import at module top level.
  The sole exception is a guarded module-level `try/except ImportError` for an *optional*
  dependency, raising a clear message with install instructions.
- **Dependency tier.** This is an application, so the tier is loose: anything needed may be a
  hard dependency. The one rule that survives is that a dependency must be *used*, not
  aspirational.
- **Typer for every CLI. Pydantic 2** at every boundary — CLI options, corpus specs, on-disk
  manifests, anything crossing into or out of a file. Internal hot loops may use plain dicts
  and numpy arrays where Pydantic per-row overhead would dominate; that is a performance
  accommodation, stated locally, not a general preference.
- **Standard-library `logging` for diagnostics — never `print`.** `print` is only for CLI
  output the user asked to see. Typer's `echo` counts as CLI output.
- **Polars over Pandas.** Prefer expressions over Python loops; `scan_*` lazyframes and
  `sink_*` streaming on the corpus-sized paths; pre-filter before joining so you never
  materialize more than needed. The reference corpora do not fit in RAM uncollected — a bare
  `collect()` on a full tier is a bug, not a style choice.
- **No `__all__`.** This is an application: import from where the symbol actually lives, and
  do not add re-export `__init__.py` files.
- **Constrained vocabularies are `frozenset[str]` + a validator** where they must grow
  additively (corpus tier names, statistic names), and a `str`-subclassing enum where bound to
  a fixed on-disk schema. Never a bare `Literal` in a persisted model — it makes every
  addition a breaking change to files already written.
- **Answers are three-valued: true / false / unknown, and `None` is never `False`.** This
  repo's specific instance: a word absent from a reference corpus is **not** a word with
  frequency zero. "Not attested in this corpus" and "attested at rate 0" differ, and the
  smoothing prior must not erase the distinction. A tier that failed to build yields
  `unknown`, never a silently-omitted column. **A comparison that could not run is not a
  comparison that found nothing.** Combine with **Kleene** semantics, not withhold-on-any-
  unknown: a word overused against three tiers with the fourth unknown is still a reportable
  result, labelled 3-of-4.
- **Deterministic ordering is load-bearing.** Frequency tables are hashed and diffed between
  runs. Never derive emitted rows from `set`/`dict` iteration, or from polars
  `mode()`/`unique()` without an explicit stable sort or tie-break — neither gives any order
  guarantee, and `mode()` is unstable call-to-call. Ties in a ranked result sort by
  `(-score, token)`, never by insertion order. Every new ordering gets a test.
- **Aggregate repeated warnings.** A warning inside a per-document loop over a million-row
  corpus needs collapsing before it ships: group by *reason*, not by row, and say which case
  it is once, with a count. Distinguish "the document had no text field" (an absence) from
  "the document had text we could not decode" (a real value we cannot hold).
- **Heed terminal warnings, deprecations especially.** Treat a deprecation in code you touched
  as a **blocker**: find the current upstream API, fix it, and update the rule here.
- **Refactor internals aggressively** — no dead code, no old API kept for nostalgia. The
  exception is the *on-disk artifact shape*: a parquet schema change orphans every result
  already computed, so it is allowed but deliberate, and it bumps the manifest version.

---

## 6. Testing — layer 1

- **Real data + ground truth.** Exercise the actual code path against real fixtures in
  `assets/` and **compute expected values at runtime** from the fixture. Do not mock a
  transformation. The statistics have closed forms — test them against hand-computed values,
  not against a second copy of the implementation.
- **Meaningful assertions.** Prefer relationships, aggregates and set equality over existence
  or count checks. `assert set(tokens) == set(expected)`, not `assert len(tokens) > 0`.
- **What to validate:** token counts and aggregates; the tokenizer's code-stripping on real
  markdown containing real fenced blocks; the log-odds sign and monotonicity; dispersion
  against a constructed concentrated-vs-spread case; parquet round-trip survival; ordering
  determinism across two runs.
- **Hardcoding domain constants is fine** (a stop-list member, a documented threshold).
  **Hardcoding a token count read off a corpus dump is not** — it drifts the moment the corpus
  is rebuilt. Derive it from the fixture at runtime.
- **Deterministic coverage.** Fixed seeds and explicit filters; representative *and* edge
  cases — empty documents, a document that is entirely a code block, a token appearing exactly
  once.
- **Be resilient to changing data.** Tests that need a downloaded corpus `pytest.skip()` when
  it is absent rather than asserting it exists, and skip gracefully when offline rather than
  mocking the download.
- **Avoid the AI test anti-patterns**: happy-path-only; expectations derived by inspecting the
  data; mocking the transformation under test; redundant checks; ignoring nulls, empties,
  boundaries and malformed input; and asserting a test "would have caught" a bug without
  demonstrating the failure first.

---

## 7. Dogfooding — layer 2

Tests prove the code does what it was told. Dogfooding asks a different question: *is this
usable, and what is missing?* Both are required; neither substitutes for the other.

**Do not "verify the pipeline's answers" with a second independent implementation while
dogfooding — that is a test, and it belongs in the suite.** Use the tool, notice the friction,
write down what was not there.

- **A capability the tool LACKS is the result, not an obstacle to route around.** This repo's
  live temptation: when the CLI cannot answer a question about the corpus, the reflex is a
  throwaway pandas one-liner in a shell. That proves the task is possible with *general*
  tooling, which was never in question, and teaches nothing about the product. Record the gap;
  if it blocks the work, build it into the CLI and carry on **with the CLI**.
- **Run the adversarial round.** Be a beta-tester trying to show the thing fails at something
  it advertises, then switch back and fix. **Attack claims, not gaps** — a documented deferral
  is a decision; what counts is where a docstring or doc *promises* something the code does
  not do. **Use real data** — no invented word, no invented z-score.
- **Pick the probe where the design generalized from one case.** If a statistic was validated
  on one corpus pair, run it on a pair with a very different size ratio. If a convention states
  a boundary (a minimum count, a frequency floor), take a real token at its edge.
- **Turn the tool on the work you just did.** The obvious probe here: run the pipeline against
  a corpus it should find *nothing* in — Claude replies split in half and compared to
  themselves. A stylometric method that finds "significant overuse" between two halves of the
  same corpus is miscalibrated, and no unit test would have said so.
- **Dogfood a finding before you report it.** Build a real example against the actual code
  path and show it fails. A loss that is mechanically possible but has no realistic
  instantiation is noise.
- **Finish each probe as a committed reference example whose README names what it broke.** The
  example is the regression test; the README is the evidence. A finding recorded only in a
  commit message is not reproducible. Keep the failure in the suite by demonstrating it on the
  *old* behaviour, not by asserting that it used to fail.
- **Separate "fix it" from "surface it" before writing any code, and be strict about the
  line.** Fix a false claim, a misdiagnosis, a wall of un-aggregated warnings. Surface anything
  where the obvious repair is itself a design decision — and say *why each candidate repair is
  wrong*, because that is what makes the item actionable later.

### The finding log

Findings carry stable `F#` IDs and **move** between files — never duplicated.

| File | Holds |
|---|---|
| `docs/dogfooding.md` | open quirks, bugs and UX gaps found by using the shipped surface |
| `docs/previous_issues.md` | findings resolved **here**, each with its resolution and a code pointer |

There is no `<upstream>-pending-fixes.md`: this repo consumes no locally-cloned repo it does
not own. Add one if that changes.

---

## 8. Docs and their lifecycle

- **All new markdown goes in `docs/`** — the only exceptions are this file and `README.md`.
- **`docs/` is the single ground truth.** This file duplicates from it only where a fact is
  needed to *orient* — and every prohibition lives here in full, because a `don't` behind a
  link is a `don't` that does not get read.
- **`docs/ROADMAP.md` is active-only.** Shipped items move to `docs/ROADMAP_HISTORY.md` with
  their rationale. Nothing is deleted; it is relocated.
- **`docs/CHANGELOG.md` records what shipped**, newest first.
- **`docs/METHODOLOGY.md` and `docs/CORPORA.md` are updated in the same change as the code
  they describe**, not after. A metric whose definition moved without its doc is a result
  nobody can interpret.
- **Update this file and the affected `docs/` in the same change as the refactor.** Policy is
  written first; code complies.
- **Run the commands yourself** rather than telling the user to run them — except where a
  command genuinely needs an interactive terminal, which is when you hand over a verbatim line.

There is no consumer inbox. `zipf` is an end-user application with no downstream consumers, and
standing up a `docs/FEEDBACK.md` here would produce a file that is always empty. If it ever
acquires a consumer, add the inbox then.

### Prose style

Natural, human prose. Avoid AI tells — em-dash pile-ups, filler transitions, marketing voice.
This repo has an obvious and deserved irony: **it is a detector of overused LLM vocabulary, so
its own docs are held to the list it produces.** When a run promotes a word into the overuse
table, stop using that word in this repo's prose.

Never hallucinate documentation or overpromise an unimplemented feature. Specifically: **this
project must never be described as detecting whether a given text was written by an LLM.** It
measures aggregate rate differences between corpora. It is not a classifier, it has no
per-document verdict, and any wording implying one is a false claim about what shipped.

---

## 9. Self-correction

When outdated API knowledge causes a real crash or logic failure, fix the code **and** update
this file (and the affected `docs/`) with the correct pattern, so the next agent does not
repeat it. The same applies when the user corrects a preference: it goes below, in their words,
with the reason.

## 10. Learned user preferences

<Append-only. One line each, in the user's terms, with the why where it is not obvious.>

- **"start with 1-word, plan 1-2-3-4 chains extension once you see stat difference in 1-word"**
  — n-gram work is *gated on unigram evidence*, not built speculatively alongside it. Establish
  the unigram signal first, then extend to 2/3/4-grams.
- **Expected findings are named up front** ("instinct, churn, gap", "you're not imagining it").
  Treat these as *predictions to be tested*, never as targets to be reached. A pipeline tuned
  until it surfaces the expected words has been fitted to its own hypothesis; report where the
  predictions failed as prominently as where they held.
- **"auto-commit in this repo"** — commit without being asked, as work reaches a coherent state.
  This **overrides the global preference** ("only `git commit` when the user explicitly asks"),
  and the override is scoped to this repository only; assume the global rule still holds
  everywhere else. Everything else in the global rule stands unchanged and is **not** implied by
  this one: no `git push`, no tags, no releases, no branch management, no history rewriting,
  never `git stash drop`/`clear`, and **never `git add -A` or `git add .`** — stage explicit
  paths and read `git status` first. Auto-commit means the commit step stops needing permission,
  not that the tree stops being the user's.
- **"start with 1-word, plan 1-2-3-4 chains extension once you see stat difference"** — sequence
  expensive stages behind evidence from cheap ones, and say what the gate was before opening it.
- **"build cheapest first, within same form"** — on morphology, and it generalises: ship the
  layer whose correctness is easiest to argue, measure what it changed, then decide whether the
  next layer is worth its ambiguity. The expensive layers are where the semantic errors live.

## 11. Learned workspace facts

<Append-only. Environment, ports, credential layout, host quirks, sibling-repo paths.>

- Claude Code transcripts live at `~/.claude/projects/<slugified-cwd>/*.jsonl`, one file per
  session. Sidechain (subagent) sessions nest one level deeper in a per-session subdirectory.
- **Assistant `thinking` blocks are signature-only in this store** — the plaintext is not
  retained, so reasoning text cannot be analysed. Only `text` blocks carry prose.
- `~/.claude/projects/*/memory/` holds memory files, not transcripts. Exclude it from harvest.
- Corpus scratch space belongs on `/data` (multi-TB); `/` has under 15 GB free and will fill.
- `files.pushshift.io` is dead. Reddit data comes from Hugging Face mirrors instead.
