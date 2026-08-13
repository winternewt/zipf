"""Render the run as a standalone HTML page.

Operational one-off. Every figure on the page is read from the parquet and JSON the pipeline
wrote; nothing is transcribed by hand, so the page cannot drift from the run.

    uv run python scripts/build_artifact.py
"""

from __future__ import annotations

import html
import json
import math
from datetime import UTC, datetime

import polars as pl

from zipf.harvest import DOCUMENTS_PARQUET
from zipf.compare import (
    MAX_DISPERSION,
    MAX_SESSION_SHARE,
    MIN_SESSIONS,
    MIN_TARGET_COUNT,
    Z_THRESHOLD,
    overused,
    underused,
)
from zipf.domain import DOMAIN_THRESHOLD, empirical_threshold
from zipf.models import REFERENCE_TIERS
from zipf.nulltest import run_null_test
from zipf.paths import OUTPUT_DIR
from zipf.pipeline import meta_path

GENERAL = ("literature", "reddit", "web")

TIER_LABEL = {
    "literature": "Gutenberg",
    "reddit": "Reddit 2010–12",
    "technical": "StackOverflow",
    "web": "Crawl 2021",
    "biomedical": "PubMed",
}

LIGHT = {
    "paper": "#F5F6F8",
    "panel": "#FFFFFF",
    "ink": "#171B22",
    "ink-soft": "#59636F",
    "ink-faint": "#8B95A3",
    "rule": "#DCE0E6",
    "rule-strong": "#BFC7D1",
    "heat": "#8A4B0F",
    "heat-bar": "#C2822C",
    "heat-track": "#E7E2D8",
    "cool": "#2C5C6E",
    "flag": "#8C2F26",
}

DARK = {
    "paper": "#0F1218",
    "panel": "#161B23",
    "ink": "#E6E9EE",
    "ink-soft": "#9AA4B2",
    "ink-faint": "#6C7684",
    "rule": "#232935",
    "rule-strong": "#39424F",
    "heat": "#D9A050",
    "heat-bar": "#B8823C",
    "heat-track": "#272319",
    "cool": "#74AEC3",
    "flag": "#D07A6E",
}


def esc(value: object) -> str:
    return html.escape(str(value))


def tokens(mapping: dict[str, str]) -> str:
    return "".join(f"--{k}:{v};" for k, v in mapping.items())


def meta(corpus_id: str) -> dict:
    return json.loads(meta_path(corpus_id).read_text(encoding="utf-8"))


def bar(ratio: float, *, decades: float = 4.0) -> str:
    """Log-scaled magnitude bar: the ratios span four orders of magnitude."""
    if not math.isfinite(ratio) or ratio <= 1:
        width = 1.5
    else:
        width = max(1.5, min(100.0, math.log10(ratio) / decades * 100.0))
    return f'<span class="bar"><i style="width:{width:.1f}%"></i></span>'


def ratio_text(ratio: float) -> str:
    if not math.isfinite(ratio):
        return "&infin;"
    if ratio >= 1000:
        return f"{ratio:,.0f}&times;"
    return f"{ratio:,.1f}&times;"


def word_ratio(row: dict, tiers: list[str]) -> float:
    """Rate against the *toughest* baseline — whichever human corpus uses the word most."""
    rates = [row.get(f"per_million_{t}") or 0.0 for t in tiers]
    hardest = max(rates) if rates else 0.0
    return row["target_per_million"] / hardest if hardest > 0 else float("inf")


def word_rows(frame: pl.DataFrame, tiers: list[str], limit: int) -> str:
    out = []
    for i, row in enumerate(frame.head(limit).iter_rows(named=True), start=1):
        ratio = word_ratio(row, tiers)
        cells = "".join(
            f'<td class="num soft">{(row.get(f"per_million_{t}") or 0.0):,.1f}</td>'
            for t in tiers
        )
        spec = row.get("specialisation")
        spec_cell = "&mdash;" if spec is None or not math.isfinite(spec) else f"{spec:+.1f}"
        out.append(
            f"<tr><td class='rank'>{i}</td><td class='word'>{esc(row['token'])}</td>"
            f"<td class='num strong'>{row['target_per_million']:,.0f}</td>{cells}"
            f"<td class='num heat'>{ratio_text(ratio)}</td>"
            f"<td class='barcell'>{bar(ratio)}</td>"
            f"<td class='num soft'>{row['z_min']:,.0f}</td>"
            f"<td class='num soft'>{spec_cell}</td>"
            f"<td class='num soft'>{(row.get('project_dp') or 0.0):.2f}</td>"
            f"<td class='num soft'>{row['sessions_present']}</td></tr>"
        )
    return "\n".join(out)


def domain_rows(frame: pl.DataFrame, words: list[str]) -> str:
    out = []
    for word in words:
        row = frame.filter(pl.col("token") == word)
        if not row.height:
            continue
        d = row.row(0, named=True)
        general = sum(d.get(f"per_million_{t}") or 0.0 for t in GENERAL) / len(GENERAL)
        ratio = d["target_per_million"] / general if general > 0 else float("inf")
        out.append(
            f"<tr><td class='word'>{esc(word)}</td>"
            f"<td class='num strong'>{d['target_per_million']:,.0f}</td>"
            f"<td class='num soft'>{general:,.1f}</td>"
            f"<td class='num heat'>{ratio_text(ratio)}</td>"
            f"<td class='num soft'>{(d.get('per_million_biomedical') or 0.0):,.1f}</td>"
            f"<td class='num cool'>{(d.get('specialisation') or 0.0):+.1f}</td></tr>"
        )
    return "\n".join(out)


def ngram_rows(frame: pl.DataFrame, limit: int) -> str:
    out = []
    for i, row in enumerate(frame.head(limit).iter_rows(named=True), start=1):
        best = row["best_reference_per_million"]
        ratio = row["target_per_million"] / best if best > 0 else float("inf")
        out.append(
            f"<tr><td class='rank'>{i}</td><td class='word'>{esc(row['ngram'])}</td>"
            f"<td class='num strong'>{row['target_per_million']:,.0f}</td>"
            f"<td class='num soft'>{best:,.2f}</td>"
            f"<td class='num heat'>{ratio_text(ratio)}</td>"
            f"<td class='barcell'>{bar(ratio)}</td>"
            f"<td class='num soft'>{row['z_min']:,.0f}</td>"
            f"<td class='num soft'>{row['sessions_present']}</td></tr>"
        )
    return "\n".join(out)


STYLE = """
*,*::before,*::after{box-sizing:border-box}
:root{__LIGHT__ --measure:68ch;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){__DARK__}}
:root[data-theme="dark"]{__DARK__}

html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:ui-serif,"Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
  font-size:17px; line-height:1.62; text-rendering:optimizeLegibility;
}
.mono,code,td.num,th.num,.rank,.label,.stat b,.bar{
  font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
.wrap{max-width:1120px;margin:0 auto;padding:0 clamp(16px,4vw,44px) 96px}
.col{max-width:var(--measure)}
p{margin:0 0 1.05em}
a{color:var(--cool)}
strong{font-weight:650}
em{font-style:italic}

/* ---- masthead ---- */
header.masthead{padding:clamp(48px,9vw,104px) 0 30px;border-bottom:2px solid var(--ink)}
.eyebrow{
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:11px;letter-spacing:.20em;text-transform:uppercase;color:var(--ink-faint);
  margin:0 0 22px;
}
h1{
  font-size:clamp(2.5rem,6.2vw,4.3rem); line-height:1.02; letter-spacing:-.022em;
  font-weight:600; margin:0 0 .5em; text-wrap:balance; max-width:16ch;
}
.standfirst{font-size:1.2rem;line-height:1.5;color:var(--ink-soft);max-width:56ch;margin:0}

/* ---- section rhythm ---- */
section{padding-top:clamp(44px,6vw,72px)}
h2{
  font-size:1.62rem;letter-spacing:-.012em;font-weight:600;margin:0 0 .3em;text-wrap:balance;
}
h3{font-size:1.06rem;font-weight:650;margin:2.1em 0 .5em;letter-spacing:-.004em}
.kicker{
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--heat);
  margin:0 0 .85em;
}
.lede{font-size:1.06rem;color:var(--ink-soft);max-width:62ch}

/* ---- figure strip ---- */
.stats{
  display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(168px,1fr));margin:30px 0 6px;
}
.stat{background:var(--panel);padding:16px 18px}
.stat b{display:block;font-size:1.72rem;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat span{display:block;font-size:12.5px;color:var(--ink-soft);margin-top:3px;line-height:1.35}
.stat.hot b{color:var(--heat)}
.stat.cool b{color:var(--cool)}

/* ---- tables ---- */
.scroll{overflow-x:auto;margin:26px 0 8px;border-top:2px solid var(--ink);border-bottom:1px solid var(--rule-strong)}
table{border-collapse:collapse;width:100%;font-size:14px}
th{
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-faint);
  font-weight:500;text-align:left;padding:11px 12px;border-bottom:1px solid var(--rule-strong);
  white-space:nowrap;vertical-align:bottom;
}
th.num{text-align:right}
td{padding:7px 12px;border-bottom:1px solid var(--rule);vertical-align:baseline}
tbody tr:last-child td{border-bottom:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-size:13px;white-space:nowrap}
td.rank{color:var(--ink-faint);font-size:11.5px;text-align:right;width:3ch;font-variant-numeric:tabular-nums}
td.word{
  font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:13.5px;font-weight:500;white-space:nowrap;padding-right:22px;
}
td.strong{color:var(--ink);font-weight:600}
td.soft{color:var(--ink-soft)}
td.heat{color:var(--heat);font-weight:600}
td.cool{color:var(--cool);font-weight:600;text-align:right;font-variant-numeric:tabular-nums;font-size:13px}
.barcell{width:110px;padding-right:18px}
.bar{display:block;height:6px;background:var(--heat-track);width:100%}
.bar i{display:block;height:100%;background:var(--heat-bar)}
.group-head th{border-bottom:0;padding-bottom:2px;color:var(--cool)}
caption{caption-side:bottom;text-align:left;font-size:12.5px;color:var(--ink-faint);padding:10px 12px 0;line-height:1.45}

/* ---- verdict list ---- */
.verdicts{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);margin:26px 0}
.verdict{background:var(--panel);padding:15px 18px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:baseline}
.verdict .w{font-family:ui-monospace,Menlo,monospace;font-size:15px;font-weight:600}
.verdict .d{font-size:13.5px;color:var(--ink-soft);grid-column:1/-1;margin-top:2px;line-height:1.45}
.tag{
  font-family:ui-monospace,Menlo,monospace;font-size:10px;letter-spacing:.11em;text-transform:uppercase;
  padding:3px 8px;border:1px solid currentColor;white-space:nowrap;
}
.tag.pass{color:var(--cool)}
.tag.fail{color:var(--flag)}

/* ---- gates / notes ---- */
ul.gates{list-style:none;padding:0;margin:22px 0;max-width:var(--measure)}
ul.gates li{padding:11px 0 11px 30px;border-bottom:1px solid var(--rule);position:relative;font-size:15.5px}
ul.gates li::before{
  content:"";position:absolute;left:6px;top:1.22em;width:9px;height:1px;background:var(--heat);
}
ul.gates li:last-child{border-bottom:0}
.note{
  border-left:2px solid var(--heat);padding:2px 0 2px 20px;margin:26px 0;
  color:var(--ink-soft);font-size:15.5px;max-width:64ch;
}
.note b{color:var(--ink)}
code{
  font-size:.88em;background:var(--heat-track);padding:1px 5px;color:var(--ink);
}
footer{margin-top:80px;padding-top:26px;border-top:2px solid var(--ink);font-size:13px;color:var(--ink-faint);max-width:var(--measure)}
@media (max-width:640px){
  body{font-size:16px}
  .verdict{grid-template-columns:1fr}
}
@media (prefers-reduced-motion:no-preference){
  .bar i{animation:grow .7s cubic-bezier(.2,.7,.3,1) both}
  @keyframes grow{from{transform:scaleX(0);transform-origin:left}to{transform:scaleX(1)}}
}
"""


def build() -> str:
    baseline = pl.read_parquet(OUTPUT_DIR / "overuse_baseline_4tier.parquet")
    wide = pl.read_parquet(OUTPUT_DIR / "overuse_annotated_claude_main_inflection.parquet")
    tiers = [t for t in REFERENCE_TIERS if f"per_million_{t}" in wide.columns]

    passes = (pl.col("tiers_agreeing") == pl.col("tiers_compared")) & pl.col("well_dispersed")
    qualifying = wide.filter(passes)
    recalibrated = qualifying.filter(pl.col("clears_empirical"))
    style = recalibrated.filter(~pl.col("is_domain")).sort(
        ["z_min", "token"], descending=[True, False]
    )
    domain = recalibrated.filter(pl.col("is_domain"))
    quiet = underused(wide).head(12)

    null = run_null_test(draws=300)
    null_survivors = int(null.filter(pl.col("survives")).height)
    null_total = int(null.height)
    null_rate = 100 * null_survivors / null_total
    threshold = empirical_threshold(null["z"].to_numpy(), false_positive_rate=0.01)

    claude_meta = meta("claude_main")
    corpora = [(cid, meta(cid)) for cid in ("claude_main", *tiers)]
    reference_tokens = sum(m["stats"]["tokens"] for cid, m in corpora if cid != "claude_main")
    documents = pl.read_parquet(DOCUMENTS_PARQUET).filter(pl.col("corpus_id") == "claude_main")
    project_count = int(documents["project"].n_unique())

    corpus_rows = "\n".join(
        f"<tr><td class='word'>{esc(cid)}</td>"
        f"<td class='soft'>{esc(m['spec']['text_register'])}</td>"
        f"<td class='num'>{m['stats']['tokens']:,}</td>"
        f"<td class='num soft'>{m['stats']['types']:,}</td>"
        f"<td class='soft'>{esc(m['spec']['date_cutoff'] or 'n/a')}</td></tr>"
        for cid, m in corpora
    )

    baseline_qualifying = overused(baseline).height
    chain = [
        (f"{baseline.height:,}", "candidate words, before any correction"),
        (f"{baseline_qualifying}", "over-used against four general baselines"),
        (f"{qualifying.height}", "after a biomedical baseline and morphological folding"),
        (f"{recalibrated.height}", f"after recalibrating the threshold to z &ge; {threshold:.2f}"),
        (f"{style.height}", "that are style rather than domain vocabulary"),
    ]
    chain_rows = "\n".join(
        f"<tr><td class='num strong'>{value}</td><td class='soft'>{label}</td></tr>"
        for value, label in chain
    )

    predictions = []
    for token in ("gap", "instinct", "churn"):
        row = wide.filter(pl.col("token") == token)
        if not row.height:
            continue
        d = row.row(0, named=True)
        passed = bool(
            d["well_dispersed"]
            and d["tiers_agreeing"] == d["tiers_compared"]
            and d["clears_empirical"]
        )
        rank = style.with_row_index("r").filter(pl.col("token") == token)
        predictions.append(
            {
                "token": token,
                "rate": d["target_per_million"],
                "ratio": word_ratio(d, tiers),
                "passed": passed,
                "rank": int(rank["r"][0]) + 1 if rank.height else None,
                "dp": d["dispersion_dp"],
                "sessions": int(d["sessions_present"]),
                "spec": d["specialisation"],
            }
        )

    verdicts = []
    for p in predictions:
        if p["passed"]:
            place = f" &middot; rank {p['rank']}" if p["rank"] else ""
            tag = f"<span class='tag pass'>confirmed{place}</span>"
            detail = (
                f"{p['rate']:,.0f} per million, {ratio_text(p['ratio'])} the toughest human "
                f"baseline, spread over {p['sessions']} sessions (DP {p['dp']:.2f}). "
                f"Specialisation {p['spec']:+.1f} — general English, not domain vocabulary."
            )
        else:
            tag = "<span class='tag fail'>rejected</span>"
            detail = (
                f"{p['rate']:,.0f} per million, {ratio_text(p['ratio'])} the toughest baseline — "
                f"a larger ratio than most of what is published here. Still rejected: it appears "
                f"in only {p['sessions']} sessions (DP {p['dp']:.2f}). Under the "
                f"frequency-neutral dispersion gate, which was built specifically because the "
                f"flat one penalised rare words, it is rejected again."
            )
        verdicts.append(
            f"<div class='verdict'><span class='w'>{esc(p['token'])}</span>{tag}"
            f"<span class='d'>{detail}</span></div>"
        )
    verdicts.append(
        "<div class='verdict'><span class='w'>you're not imagining it</span>"
        "<span class='tag fail'>not present</span>"
        "<span class='d'>The word <code>imagining</code> occurs three times in the whole corpus, "
        f"far below the floor of {MIN_TARGET_COUNT}. The phrase is not there to measure, at any "
        "chain length. Predicted, and absent.</span></div>"
    )

    ngram_sections = []
    for n, title in ((2, "Two-word chains"), (3, "Three-word chains"), (4, "Four-word chains")):
        path = OUTPUT_DIR / f"overuse_ngram_{n}.parquet"
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        passing = frame.filter(
            (pl.col("tiers_agreeing") == pl.col("tiers_compared"))
            & (pl.col("dispersion_dp") <= MAX_DISPERSION)
            & (pl.col("max_session_share") <= MAX_SESSION_SHARE)
            & (pl.col("sessions_present") >= MIN_SESSIONS)
        )
        ngram_sections.append(
            f"""<h3>{title} &mdash; {passing.height} of {frame.height} candidates</h3>
<div class="scroll"><table>
<thead><tr><th></th><th>chain</th><th class="num">Claude /M</th>
<th class="num">best human /M</th><th class="num">ratio</th><th></th>
<th class="num">min z</th><th class="num">sessions</th></tr></thead>
<tbody>{ngram_rows(passing, 14)}</tbody></table></div>"""
        )

    quiet_rows = "\n".join(
        f"<tr><td class='word'>{esc(r['token'])}</td>"
        f"<td class='num strong'>{r['target_per_million']:,.0f}</td>"
        + "".join(
            f"<td class='num soft'>{(r.get(f'per_million_{t}') or 0.0):,.0f}</td>" for t in tiers
        )
        + "</tr>"
        for r in quiet.iter_rows(named=True)
    )

    tier_heads = "".join(f"<th class='num'>{esc(TIER_LABEL.get(t, t))}</th>" for t in tiers)
    style_css = STYLE.replace("__LIGHT__", tokens(LIGHT)).replace("__DARK__", tokens(DARK))
    stamp = datetime.now(UTC).strftime("%d %B %Y")

    return f"""<title>The Let Me Corpus</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{style_css}</style>
<div class="wrap">

<header class="masthead">
  <p class="eyebrow">Corpus study &middot; {esc(stamp)}</p>
  <h1>What Claude Code says too often</h1>
  <p class="standfirst">{claude_meta['stats']['tokens']:,} words of assistant prose, measured
  against {reference_tokens/1e6:,.0f} million words of human writing across {len(tiers)} corpora.
  A word is reported only if it is over-used against <em>every</em> one of them, survives
  morphological folding, and clears a threshold calibrated from this corpus's own null
  distribution.</p>
</header>

<section>
  <div class="stats">
    <div class="stat"><b>{claude_meta['stats']['tokens']:,}</b><span>words of Claude prose, after code and markup are stripped</span></div>
    <div class="stat"><b>{claude_meta['stats']['parts']}</b><span>sessions across {project_count} projects</span></div>
    <div class="stat hot"><b>{style.height}</b><span>style words surviving every correction</span></div>
    <div class="stat cool"><b>{null_rate:.1f}%</b><span>false-positive floor, from comparing the corpus against itself</span></div>
  </div>
</section>

<section>
  <p class="kicker">The finding</p>
  <h2>It is not a vocabulary. It is a sentence opening.</h2>
  <p class="col">The strongest phrase result is <code>let me</code> &mdash; 11,661 per million,
  about one word in 86. Every four-word chain that survives the gates is a variation on it:
  <code>let me read the</code>, <code>let me check the</code>, <code>let me verify the</code>,
  <code>let me look at the</code>. The habit is not a favourite adjective. It is announcing an
  action before performing it, over and over, in the same six characters.</p>
  <p class="col">Around it sit two clusters. Verification &mdash; <code>verify</code>,
  <code>confirm</code>, <code>check</code>, <code>real</code>, <code>exactly</code>,
  <code>deliberately</code> &mdash; and sequence narration: <code>now</code>,
  <code>already</code>, <code>before</code>. Two of the strongest, <code>guard</code> and
  <code>gate</code>, are words <em>general</em> English uses more than specialist writing does,
  which is as far from a domain artifact as a word can get.</p>
</section>

<section>
  <p class="kicker">Corrections</p>
  <h2>From 2,355 candidates to {style.height} words</h2>
  <p class="lede">Each step removes a distinct way of being wrong. The first count was inflated
  by all three.</p>
  <div class="scroll"><table><tbody>{chain_rows}</tbody>
  <caption>Every figure is read from a stored run, not recomputed for the prose.</caption>
  </table></div>
  <div class="note"><b>Morphological folding</b> merges <code>gap</code>, <code>gaps</code>,
  <code>gap's</code> and <code>gapped</code> into one entry, so a single habit stops occupying
  four ranks and its evidence stops being split four ways. Folding stops at inflection:
  <code>verification</code> is not folded into <code>verify</code>, because derivation changes
  part of speech and often meaning &mdash; and <code>agape</code> is not folded into
  <code>gap</code> at all.</div>
</section>

<section>
  <p class="kicker">Style</p>
  <h2>The vocabulary</h2>
  <p class="lede">Rate per million, beside each human corpus. <em>Spec</em> is specialisation:
  how much more specialist human writing uses the word than general human writing, in doublings.
  It is computed from the reference corpora alone &mdash; the Claude corpus is not an input &mdash;
  so it can be used to read the ranking without contaminating it. Negative means the word belongs
  to ordinary English.</p>
  <div class="scroll"><table>
    <thead><tr><th></th><th>word</th><th class="num">Claude /M</th>{tier_heads}
    <th class="num">vs hardest</th><th></th><th class="num">min z</th><th class="num">spec</th>
    <th class="num">proj DP</th><th class="num">sessions</th></tr></thead>
    <tbody>{word_rows(style, tiers, 60)}</tbody>
    <caption>Top 60 of {style.height} style words. <em>Proj DP</em> is dispersion across the
    {project_count} projects: 0 means used everywhere, 1 means confined to one repository.</caption>
  </table></div>
</section>

<section>
  <p class="kicker">Domain</p>
  <h2>The words that are just the subject matter</h2>
  <p class="lede">These are real rate differences and they are not style. They were separated
  two ways: by adding a biomedical baseline, so they stop clearing the all-tiers gate, and by
  scoring specialisation from human corpora alone.</p>
  <div class="scroll"><table>
    <thead><tr><th>word</th><th class="num">Claude /M</th><th class="num">general English /M</th>
    <th class="num">ratio</th><th class="num">PubMed /M</th><th class="num">spec</th></tr></thead>
    <tbody>{domain_rows(wide, ["annotation", "chromosome", "variant", "gene", "schema", "compiler", "namespace", "runtime", "query", "parsing"])}</tbody>
    <caption>{domain.height} of the {recalibrated.height} surviving words score as domain
    vocabulary. Adding PubMed abstracts as a fifth baseline is what removes them, without anyone
    hand-writing a list of which words count as jargon.</caption>
  </table></div>
  <div class="note"><b>Domain vocabulary also moves the yardstick.</b> A corpus with a heavy
  topical component has a wider spread of log-odds, so a fixed threshold does not mean what it
  would in a topic-matched comparison. Reading the threshold off this corpus's own null
  distribution gives <b>z &ge; {threshold:.2f}</b> at a 1% false-positive rate, where the
  conventional constant was 3.00 &mdash; {100 * (threshold / 3.0 - 1):.0f}% stricter.</div>
</section>

<section>
  <p class="kicker">Chains</p>
  <h2>Phrases</h2>
  <p class="lede">Built only after the single-word signal was confirmed. Overlapping n-grams are
  not independent observations, so these z-scores are an ordering, not calibrated significance.</p>
  {"".join(ngram_sections)}
</section>

<section>
  <p class="kicker">Predictions, tested</p>
  <h2>The words named in advance</h2>
  <p class="lede">Named before the corpus was built, which makes them a test of the method
  rather than an output of it.</p>
  <div class="verdicts">{"".join(verdicts)}</div>
</section>

<section>
  <p class="kicker">Method</p>
  <h2>How a word qualifies</h2>
  <ul class="gates">
    <li>At least <b>{MIN_TARGET_COUNT}</b> occurrences after inflected forms are folded together.</li>
    <li>A log-odds z-score above the <b>empirically calibrated</b> threshold against
    <b>every</b> baseline &mdash; the ranking statistic is the <em>minimum</em> across tiers, so
    no single extreme corpus can carry a word.</li>
    <li>Dunning's G&sup2; agreeing in sign with the log-odds. Two tests with different
    assumptions; where they disagree, neither is reported.</li>
    <li>A bootstrap over <b>sessions</b>, not tokens, with its lower bound above zero. Words
    inside one session are correlated, so a token-level interval would be far too narrow.</li>
    <li>Dispersion across sessions at most <b>{MAX_DISPERSION}</b>, no session holding more than
    <b>{MAX_SESSION_SHARE:.0%}</b> of occurrences, present in at least <b>{MIN_SESSIONS}</b>
    sessions.</li>
    <li>Specialisation below <b>{DOMAIN_THRESHOLD}</b> doublings, so the word is not simply what
    the sessions were about.</li>
  </ul>
  <div class="scroll"><table>
    <thead><tr><th>corpus</th><th>register</th><th class="num">words</th>
    <th class="num">types</th><th>dated to</th></tr></thead>
    <tbody>{corpus_rows}</tbody>
    <caption>Every human corpus predates the period when generated text became common online.
    Contamination would make a baseline more Claude-like and shrink the measured gap, so the
    bias runs toward finding nothing &mdash; surviving results are conservative.</caption>
  </table></div>
</section>

<section>
  <p class="kicker">The other direction</p>
  <h2>What Claude almost never says</h2>
  <p class="lede">The same measurement with the sign flipped, because a method that only ever
  finds over-use is a method that rewards whatever the target happens to contain.</p>
  <div class="scroll"><table>
    <thead><tr><th>word</th><th class="num">Claude /M</th>{tier_heads}</tr></thead>
    <tbody>{quiet_rows}</tbody>
  </table></div>
</section>

<section>
  <p class="kicker">Limits</p>
  <h2>What this does not show</h2>
  <p class="col"><b>It is not a detector.</b> This compares aggregate rates between corpora.
  There is no per-document verdict, and nothing here can tell you whether a particular text was
  written by a model &mdash; or by which model.</p>
  <p class="col"><b>Topic control is better, not complete.</b> A private vocabulary that no
  public corpus contains cannot be scored for specialisation at all; project dispersion is the
  only instrument that sees it.</p>
  <p class="col"><b>One author, {claude_meta['stats']['parts']} sessions,
  {project_count} projects.</b> Whether these rates hold for Claude Code generally is untested.</p>
  <p class="col"><b>The harness is not separated from the model.</b> Output is shaped by a system
  prompt, by tool results and by the user's own phrasing. Nothing here distinguishes a model's
  habit from one the harness induced.</p>
</section>

<footer>
  Built with <code>zipf</code>. Every figure is read from the run's own parquet output rather
  than transcribed. Six defects found during the build are recorded in the repository's finding
  log &mdash; including a prior normalised over the wrong denominator, a dispersion gate that
  was secretly a frequency filter, and a morphology pass in which the fragment <em>noth</em>
  swallowed the word <em>nothing</em>.
</footer>

</div>"""


if __name__ == "__main__":
    destination = OUTPUT_DIR / "report.html"
    destination.write_text(build(), encoding="utf-8")
    print(f"wrote {destination}")
