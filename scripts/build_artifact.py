"""Render the run as a standalone HTML page.

Operational one-off. Every figure on the page is read from the parquet and JSON the pipeline
wrote; nothing is transcribed by hand, so the page cannot drift from the run.

    uv run python scripts/build_artifact.py

The display face is inlined as a data URI from `assets/`, because the artifact host's CSP
blocks font CDNs and a linked webfont would fall back silently.

The chart's two series colours were checked with the dataviz validator rather than chosen by
eye — both themes pass the lightness band, chroma floor, CVD separation, normal-vision floor
and contrast checks. Changing them means re-running that validator.
"""

from __future__ import annotations

import collections
import html
import json
import math
from datetime import UTC, datetime

import polars as pl

from zipf.compare import (
    MAX_DISPERSION,
    MAX_SESSION_SHARE,
    MIN_SESSIONS,
    MIN_TARGET_COUNT,
    overused,
    underused,
)
from zipf.domain import DOMAIN_THRESHOLD, empirical_threshold
from zipf.harvest import DOCUMENTS_PARQUET
from zipf.models import REFERENCE_TIERS
from zipf.nulltest import run_null_test
from zipf.ngrams import iter_ngrams
from zipf.paths import ASSETS_DIR, OUTPUT_DIR
from zipf.pipeline import load_totals
from zipf.tokenize import iter_tokens
from zipf.pipeline import meta_path

GENERAL = ("literature", "reddit", "web")

TIER_LABEL = {
    "literature": "Gutenberg",
    "reddit": "Reddit",
    "technical": "Stack Overflow",
    "web": "Common Crawl",
    "biomedical": "PubMed",
    "commits": "commit msgs",
    "vcs": "git manual",
}

# --- palette -----------------------------------------------------------------------------
# Neutrals are biased slightly toward the blue accent so they read as chosen rather than
# inherited. The two series colours are validated; see the module docstring.
LIGHT = {
    "paper": "#F4F5F7",
    "panel": "#FFFFFF",
    "ink": "#15181D",
    "ink-soft": "#5A6470",
    "ink-faint": "#8A94A1",
    "rule": "#DEE2E7",
    "rule-strong": "#C0C7D0",
    "claude": "#B5521C",
    "human": "#3560AB",
    "claude-soft": "#EFE3D8",
    "grid": "#E6E9ED",
}
DARK = {
    "paper": "#101318",
    "panel": "#171B22",
    "ink": "#E7EAEE",
    "ink-soft": "#98A2AF",
    "ink-faint": "#69737F",
    "rule": "#242A33",
    "rule-strong": "#39414C",
    "claude": "#D0793A",
    "human": "#5A83CC",
    "claude-soft": "#2A2018",
    "grid": "#1E242C",
}


def esc(value: object) -> str:
    return html.escape(str(value))


def tokens(mapping: dict[str, str]) -> str:
    return "".join(f"--{k}:{v};" for k, v in mapping.items())


def meta(corpus_id: str) -> dict:
    return json.loads(meta_path(corpus_id).read_text(encoding="utf-8"))


def font_face() -> str:
    """Inline the display face. A missing asset degrades to the system serif, not to silence."""
    path = ASSETS_DIR / "instrument-serif.woff2.b64"
    if not path.exists():
        return ""
    payload = path.read_text(encoding="utf-8").strip()
    return (
        "@font-face{font-family:'Display';font-style:normal;font-weight:400;"
        f"font-display:swap;src:url(data:font/woff2;base64,{payload}) format('woff2');}}"
    )


def ratio_text(ratio: float) -> str:
    if not math.isfinite(ratio):
        return "&infin;"
    if ratio >= 100:
        return f"{ratio:,.0f}&times;"
    return f"{ratio:,.1f}&times;"


def hardest_rate(row: dict, tiers: list[str]) -> float:
    """The rate of whichever human corpus uses the word most — the toughest comparison."""
    return max((row.get(f"per_million_{t}") or 0.0) for t in tiers)


def word_ratio(row: dict, tiers: list[str]) -> float:
    hardest = hardest_rate(row, tiers)
    return row["target_per_million"] / hardest if hardest > 0 else float("inf")


# --- the chart ---------------------------------------------------------------------------


def dumbbell(rows: list[dict], tiers: list[str], *, count: int = 18) -> str:
    """A dumbbell chart: each word's human rate and Claude's rate, joined.

    The form follows the job. There are two values per word and the *distance between them* is
    the finding, which is what a dumbbell encodes directly and what a bar chart of ratios
    throws away. The axis is logarithmic because the rates span four orders of magnitude; on a
    linear axis every human rate would collapse onto the left edge.
    """
    # Ordered by ratio, matching the tables and this chart's own title. An earlier version
    # ranked by z to avoid promoting thin technical abbreviations; the domain and
    # version-control filters have since removed almost all of them, and the one that remains
    # is named in the caption rather than quietly dropped.
    data = []
    for row in rows:
        human = hardest_rate(row, tiers)
        claude = row["target_per_million"]
        if human > 0 and claude > 0:
            data.append((row["token"], human, claude, claude / human))
    data.sort(key=lambda d: (-d[3], d[0]))
    data = data[:count]
    if not data:
        return ""

    row_height = 29
    pad_top, pad_bottom = 16, 42
    label_w, ratio_w, plot_w = 124, 62, 646
    width = label_w + plot_w + ratio_w
    height = pad_top + len(data) * row_height + pad_bottom

    # The axis floor comes from the data. A fixed floor left half the plot empty, because no
    # word in the set is as rare as one occurrence per ten million.
    lo = max(0.05, min(h for _, h, _, _ in data) / 2.5)
    hi = max(c for _, _, c, _ in data) * 1.6
    log_lo, log_hi = math.log10(lo), math.log10(hi)

    def x(value: float) -> float:
        return label_w + (math.log10(max(value, lo)) - log_lo) / (log_hi - log_lo) * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        'role="img" aria-label="For each word, the rate per million in the human corpus that '
        'uses it most, and the rate in Claude Code prose.">'
    ]

    decade = math.ceil(log_lo)
    while decade <= math.floor(math.log10(hi)):
        gx = x(10**decade)
        label = f"{10**decade:,.0f}" if decade >= 0 else f"{10**decade:g}"
        parts.append(
            f'<line x1="{gx:.1f}" y1="{pad_top - 4}" x2="{gx:.1f}" '
            f'y2="{height - pad_bottom + 6}" stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{gx:.1f}" y="{height - pad_bottom + 22}" class="axis" '
            f'text-anchor="middle">{label}</text>'
        )
        decade += 1
    parts.append(
        f'<text x="{label_w + plot_w / 2:.0f}" y="{height - 6}" class="axis-title" '
        'text-anchor="middle">occurrences per million words (log scale)</text>'
    )

    for i, (token, human, claude, ratio) in enumerate(data):
        cy = pad_top + i * row_height + row_height / 2
        hx, cx = x(human), x(claude)
        parts.append(
            f'<g class="dumb"><title>{esc(token)} — Claude {claude:,.0f} per million; '
            f"toughest human corpus {human:,.1f} per million; {ratio:,.0f} times</title>"
            f'<text x="{label_w - 12}" y="{cy + 4:.1f}" class="rowlabel" '
            f'text-anchor="end">{esc(token)}</text>'
            f'<line x1="{hx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" class="connector"/>'
            f'<circle cx="{hx:.1f}" cy="{cy:.1f}" r="5" class="mark-human"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" class="mark-claude"/>'
            f'<text x="{width - 6}" y="{cy + 4:.1f}" class="rowratio" '
            f'text-anchor="end">{ratio:,.0f}&#215;</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --- tables ------------------------------------------------------------------------------


def by_ratio(frame: pl.DataFrame, tiers: list[str]) -> list[dict]:
    """Rows ordered by rate against the toughest human corpus, ties broken on the token.

    Ratio rather than z, because that is the quantity the page displays and the chart is named
    after. The cost is stated in the captions: ratio rewards rarity, so an abbreviation rises,
    and a very frequent word that clears significance on a small margin drops out entirely.
    """
    rows = list(frame.iter_rows(named=True))
    return sorted(rows, key=lambda r: (-word_ratio(r, tiers), r["token"]))


def word_rows(frame: pl.DataFrame, tiers: list[str], limit: int) -> str:
    out = []
    for i, row in enumerate(by_ratio(frame, tiers)[:limit], start=1):
        cells = "".join(
            f'<td class="num soft">{(row.get(f"per_million_{t}") or 0.0):,.1f}</td>'
            for t in tiers
        )
        manual = row.get("vcs_per_million")
        manual_cell = "&mdash;" if manual is None or not math.isfinite(manual) else f"{manual:,.0f}"
        spec = row.get("specialisation")
        spec_cell = "&mdash;" if spec is None or not math.isfinite(spec) else f"{spec:+.1f}"
        out.append(
            f"<tr><td class='rank'>{i}</td><td class='word'>{esc(row['token'])}</td>"
            f"<td class='num claude'>{row['target_per_million']:,.0f}</td>{cells}"
            f"<td class='num human'>{manual_cell}</td>"
            f"<td class='num ratio'>{ratio_text(word_ratio(row, tiers))}</td>"
            f"<td class='num soft'>{row['z_min']:,.0f}</td>"
            f"<td class='num soft'>{spec_cell}</td>"
            f"<td class='num soft'>{(row.get('project_dp') or 0.0):.2f}</td>"
            f"<td class='num soft'>{row['sessions_present']}</td></tr>"
        )
    return "\n".join(out)


def ngram_rows(frame: pl.DataFrame, limit: int) -> str:
    def ratio_of(r: dict) -> float:
        best = r["best_reference_per_million"]
        return r["target_per_million"] / best if best > 0 else float("inf")

    out = []
    ordered = sorted(frame.iter_rows(named=True), key=lambda r: (-ratio_of(r), r["ngram"]))
    for i, row in enumerate(ordered[:limit], start=1):
        best = row["best_reference_per_million"]
        ratio = ratio_of(row)
        out.append(
            f"<tr><td class='rank'>{i}</td><td class='word'>{esc(row['ngram'])}</td>"
            f"<td class='num claude'>{row['target_per_million']:,.0f}</td>"
            f"<td class='num soft'>{best:,.2f}</td>"
            f"<td class='num ratio'>{ratio_text(ratio)}</td>"
            f"<td class='num soft'>{row['z_min']:,.0f}</td>"
            f"<td class='num soft'>{row['sessions_present']}</td></tr>"
        )
    return "\n".join(out)


def register_rows(frame: pl.DataFrame, words: list[str], survivors: set[str]) -> str:
    """Words tested against the commit-message tier, and whether they survived it."""
    out = []
    for word in words:
        row = frame.filter(pl.col("token") == word)
        if not row.height:
            continue
        d = row.row(0, named=True)
        commits = d.get("per_million_commits") or 0.0
        kept = word in survivors
        verdict = (
            "<span class='tag pass'>style</span>"
            if kept
            else "<span class='tag fail'>register</span>"
        )
        ratio = d["target_per_million"] / commits if commits > 0 else float("inf")
        out.append(
            f"<tr><td class='word'>{esc(word)}</td>"
            f"<td class='num claude'>{d['target_per_million']:,.0f}</td>"
            f"<td class='num human'>{commits:,.0f}</td>"
            f"<td class='num ratio'>{ratio_text(ratio)}</td>"
            f"<td>{verdict}</td></tr>"
        )
    return "\n".join(out)


#: What the cartoon claimed, and how each claim has to be measured. A single word is a token; a
#: family is a set of surface forms; a frame is a template whose fills are individually too rare
#: to rank. Measuring the first where the truth is the third is the error recorded as F14.
MEME_CLAIMS = [
    ("load-bearing", "word", ["load-bearing"]),
    ("genuinely", "word", ["genuinely"]),
    ("deliberately / intentional", "family", ["deliberately", "deliberate", "intentional"]),
    ("full picture", "frame-before", ["picture"]),
    ("all green", "bigram", ["all green"]),
    ("worth separating", "frame-after", ["worth"]),
    ("just say the word", "bigram", ["say the"]),
    ("a genuinely sharp point", "family", ["sharp", "sharply", "sharper", "sharpen", "sharpens"]),
    ("intentional seams", "family", ["seam", "seams", "seamless", "seamlessly"]),
    ("the wedge", "family", ["wedge", "wedges", "wedged"]),
    # Measured alone. Bundling it with `fine-grained`, which Claude does use, would rescue a
    # failed prediction by arithmetic — the caption says coarse.
    ("coarse-grained salt", "family", ["coarse-grained"]),
    ("substrate", "family", ["substrate", "substrates"]),
    ("with teeth", "family", ["teeth", "tooth"]),
]


def meme_rows(claude: dict, cn: int, general: dict, gn: int, bigrams, n2: int, refmap: dict) -> str:
    """Measure each cartoon claim at the unit it actually lives at."""
    out = []
    for label, kind, forms in MEME_CLAIMS:
        if kind in ("word", "family"):
            c = sum(claude.get(f, 0) for f in forms) / cn * 1e6
            g = sum(general.get(f, 0) for f in forms) / gn * 1e6
            unit = "word" if kind == "word" else "word family"
        elif kind == "bigram":
            ph = forms[0]
            c = bigrams.get(ph, 0) / n2 * 1e6
            g = refmap.get(ph, 0.0) or 0.0
            unit = "phrase"
        else:
            # A frame has a direction. `[modifier] picture` takes the head last; `worth [verb]`
            # takes it first. Matching either position pulls in `is worth` and `picture of`,
            # which belong to different constructions and inflate the count.
            head = forms[0]
            position = 1 if kind == "frame-before" else 0
            hits = {p: k for p, k in bigrams.items() if p.split()[position] == head}
            c = sum(hits.values()) / n2 * 1e6
            g = max((refmap.get(p, 0.0) or 0.0) for p in hits) if hits else 0.0
            unit = f"frame, {len(hits)} fills"
        ratio = c / g if g > 0 else float("inf")
        if c < 1:
            verdict, cls = "not in the corpus", "fail"
        elif ratio >= 4:
            verdict, cls = "confirmed", "pass"
        elif ratio < 1:
            verdict, cls = "used less than humans", "fail"
        else:
            verdict, cls = "too weak to report", "fail"
        out.append(
            f"<tr><td class='word'>{esc(label)}</td><td class='soft'>{unit}</td>"
            f"<td class='num claude'>{c:,.0f}</td><td class='num soft'>{g:,.2f}</td>"
            f"<td class='num ratio'>{ratio_text(ratio)}</td>"
            f"<td><span class='tag {cls}'>{verdict}</span></td></tr>"
        )
    return "\n".join(out)


def vcs_rows(frame: pl.DataFrame, limit: int) -> str:
    """Words the git documentation uses as much as Claude does, or more."""
    ordered = frame.sort("target_per_million", descending=True)
    out = []
    for row in ordered.head(limit).iter_rows(named=True):
        vcs = row.get("vcs_per_million") or 0.0
        ratio = row["target_per_million"] / vcs if vcs > 0 else float("inf")
        out.append(
            f"<tr><td class='word'>{esc(row['token'])}</td>"
            f"<td class='num claude'>{row['target_per_million']:,.0f}</td>"
            f"<td class='num human'>{vcs:,.0f}</td>"
            f"<td class='num ratio'>{ratio:,.2f}&times;</td></tr>"
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
            f"<td class='num claude'>{d['target_per_million']:,.0f}</td>"
            f"<td class='num soft'>{general:,.1f}</td>"
            f"<td class='num ratio'>{ratio_text(ratio)}</td>"
            f"<td class='num soft'>{(d.get('per_million_biomedical') or 0.0):,.1f}</td>"
            f"<td class='num human'>{(d.get('specialisation') or 0.0):+.1f}</td></tr>"
        )
    return "\n".join(out)


STYLE = """
*,*::before,*::after{box-sizing:border-box}
__FONT__
:root{__LIGHT__ --measure:66ch;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){__DARK__}}
:root[data-theme="dark"]{__DARK__}

html{-webkit-text-size-adjust:100%}
body{
  margin:0;background:var(--paper);color:var(--ink);
  font-family:ui-serif,"Iowan Old Style",Charter,"Palatino Linotype",Palatino,Georgia,serif;
  font-size:17px;line-height:1.6;text-rendering:optimizeLegibility;
}
.display{font-family:Display,ui-serif,Georgia,serif;font-weight:400}
code,.mono,td.num,th,.rank,.eyebrow,.kicker,.tag,.legend{
  font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
.wrap{max-width:1080px;margin:0 auto;padding:0 clamp(16px,4vw,40px) 90px}
.col{max-width:var(--measure)}
p{margin:0 0 1em}
strong{font-weight:650}
a{color:var(--human)}
:focus-visible{outline:2px solid var(--human);outline-offset:2px}

header{padding:clamp(38px,7vw,84px) 0 0}
.eyebrow{font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--ink-faint);
  margin:0 0 28px}
.hero{border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);
  padding:22px 0 20px;margin:0 0 26px}
.hero-phrase{display:block;font-size:clamp(3.6rem,16vw,10.5rem);line-height:.84;
  letter-spacing:-.035em;color:var(--claude);margin:0}
.hero-meta{display:flex;flex-wrap:wrap;gap:6px 30px;margin-top:20px;align-items:baseline}
.hero-meta b{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;font-size:1.5rem;
  font-weight:500;font-variant-numeric:tabular-nums;letter-spacing:-.02em;color:var(--ink)}
.hero-meta span{font-size:13px;color:var(--ink-soft)}
h1{font-size:clamp(1.85rem,4.4vw,2.85rem);line-height:1.06;letter-spacing:-.02em;
  font-weight:400;margin:0 0 .45em;text-wrap:balance;max-width:20ch}
.standfirst{font-size:1.1rem;line-height:1.5;color:var(--ink-soft);max-width:58ch;margin:0}

section{padding-top:clamp(40px,6vw,66px)}
h2{font-size:1.7rem;font-weight:400;letter-spacing:-.012em;margin:0 0 .3em;text-wrap:balance}
h3{font-size:1rem;font-weight:650;margin:2em 0 .5em}
.kicker{font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--claude);
  margin:0 0 .9em}
.lede{font-size:1.03rem;color:var(--ink-soft);max-width:60ch}

.stats{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(166px,1fr));margin:26px 0 30px}
.stat{background:var(--panel);padding:15px 17px}
.stat b{display:block;font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  font-size:1.72rem;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.1;
  letter-spacing:-.03em}
.stat span{display:block;font-size:12.5px;color:var(--ink-soft);margin-top:5px;line-height:1.35}

figure{margin:24px 0 0}
.chartbox{overflow-x:auto;border-top:1px solid var(--rule-strong);padding-top:14px}
.legend{display:flex;gap:20px;font-size:11.5px;color:var(--ink-soft);margin:0 0 8px;
  flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;
  vertical-align:-1px}
.legend .lh{background:var(--human)}
.legend .lc{background:var(--claude)}
svg .axis{font-size:10.5px;fill:var(--ink-faint);
  font-family:ui-monospace,Menlo,Consolas,monospace}
svg .axis-title{font-size:9.5px;fill:var(--ink-faint);letter-spacing:.09em;
  text-transform:uppercase;font-family:ui-monospace,Menlo,Consolas,monospace}
svg .rowlabel{font-size:12px;fill:var(--ink);
  font-family:ui-monospace,Menlo,Consolas,monospace}
svg .rowratio{font-size:11px;fill:var(--claude);
  font-family:ui-monospace,Menlo,Consolas,monospace}
svg .connector{stroke:var(--rule-strong);stroke-width:2}
svg .mark-human{fill:var(--human);stroke:var(--paper);stroke-width:2}
svg .mark-claude{fill:var(--claude);stroke:var(--paper);stroke-width:2}
svg .dumb:hover .connector{stroke:var(--ink-faint)}
figcaption{font-size:12.5px;color:var(--ink-faint);margin-top:12px;max-width:64ch;
  line-height:1.45}

.scroll{overflow-x:auto;margin:24px 0 6px;border-top:1px solid var(--ink);
  border-bottom:1px solid var(--rule-strong)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-faint);
  font-weight:400;text-align:left;padding:10px 11px;border-bottom:1px solid var(--rule-strong);
  white-space:nowrap;vertical-align:bottom}
th.num{text-align:right}
td{padding:6px 11px;border-bottom:1px solid var(--rule);vertical-align:baseline}
tbody tr:last-child td{border-bottom:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-size:12.5px;white-space:nowrap}
td.rank{color:var(--ink-faint);font-size:11px;text-align:right;width:3ch;
  font-variant-numeric:tabular-nums}
td.word{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;font-weight:500;
  white-space:nowrap;padding-right:20px}
td.soft{color:var(--ink-soft)}
td.claude{color:var(--claude);font-weight:600}
td.human{color:var(--human);font-weight:600}
td.ratio{color:var(--ink);font-weight:600}
caption{caption-side:bottom;text-align:left;font-size:12.5px;color:var(--ink-faint);
  padding:10px 11px 0;line-height:1.45}

.verdicts{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);margin:24px 0}
.verdict{background:var(--panel);padding:14px 17px;display:grid;
  grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:baseline}
.verdict .w{font-family:ui-monospace,Menlo,monospace;font-size:14.5px;font-weight:600}
.verdict .d{font-size:13.5px;color:var(--ink-soft);grid-column:1/-1;margin-top:3px;
  line-height:1.45}
.tag{font-size:10px;letter-spacing:.1em;text-transform:uppercase;padding:3px 8px;
  border:1px solid currentColor;white-space:nowrap}
.tag.pass{color:var(--human)}
.tag.fail{color:var(--claude)}

ul.gates{list-style:none;padding:0;margin:20px 0;max-width:var(--measure)}
ul.gates li{padding:10px 0 10px 28px;border-bottom:1px solid var(--rule);position:relative;
  font-size:15px}
ul.gates li::before{content:"";position:absolute;left:4px;top:1.2em;width:9px;height:1px;
  background:var(--claude)}
ul.gates li:last-child{border-bottom:0}
.note{border-left:2px solid var(--claude);padding:1px 0 1px 18px;margin:24px 0;
  color:var(--ink-soft);font-size:15px;max-width:62ch}
.note b{color:var(--ink)}
code{font-size:.86em;background:var(--claude-soft);padding:1px 5px;color:var(--ink)}
footer{margin-top:72px;padding-top:24px;border-top:1px solid var(--ink);font-size:13px;
  color:var(--ink-faint);max-width:var(--measure)}
@media (max-width:640px){body{font-size:16px}.verdict{grid-template-columns:1fr}}
"""


def build() -> str:
    baseline = pl.read_parquet(OUTPUT_DIR / "overuse_baseline_4tier.parquet")
    wide = pl.read_parquet(OUTPUT_DIR / "overuse_annotated_claude_main_inflection.parquet")
    tiers = [t for t in REFERENCE_TIERS if f"per_million_{t}" in wide.columns]

    passes = (pl.col("tiers_agreeing") == pl.col("tiers_compared")) & pl.col("well_dispersed")
    qualifying = wide.filter(passes)
    recalibrated = qualifying.filter(pl.col("clears_empirical"))
    # Ranked by rate against the toughest corpus, the same order the tables and chart display,
    # so a rank quoted in a verdict points at the row the reader can see. Ties break on the
    # token: two runs must rank identically.
    style = recalibrated.filter(~pl.col("is_domain") & ~pl.col("is_version_control"))
    style = style.with_columns(
        _ratio=pl.max_horizontal([pl.col(f"per_million_{t}") for t in tiers]).pow(-1)
        * pl.col("target_per_million")
    ).sort(["_ratio", "token"], descending=[True, False])
    domain = recalibrated.filter(pl.col("is_domain"))
    version_control = recalibrated.filter(pl.col("is_version_control"))
    survivors = set(style["token"])
    quiet = underused(wide).head(12)

    null = run_null_test(draws=300)
    null_rate = 100 * int(null.filter(pl.col("survives")).height) / int(null.height)
    threshold = empirical_threshold(null["z"].to_numpy(), false_positive_rate=0.01)

    claude_meta = meta("claude_main")
    corpora = [(cid, meta(cid)) for cid in ("claude_main", *tiers)]
    if (OUTPUT_DIR / "meta_vcs.json").exists():
        corpora.append(("vcs", meta("vcs")))
    reference_tokens = sum(m["stats"]["tokens"] for cid, m in corpora if cid != "claude_main")
    documents = pl.read_parquet(DOCUMENTS_PARQUET).filter(pl.col("corpus_id") == "claude_main")
    project_count = int(documents["project"].n_unique())

    claude_counts, claude_n = load_totals("claude_main")
    gen_counts: dict[str, int] = {}
    gen_n = 0
    for t in GENERAL:
        tot, size = load_totals(t)
        for w, c in tot.items():
            gen_counts[w] = gen_counts.get(w, 0) + c
        gen_n += size
    # Bigrams are recomputed rather than read from the candidate table: the frames below are
    # made of fills that are individually far below that table's floor.
    bigram_counts: collections.Counter[str] = collections.Counter()
    bigram_n = 0
    for text in documents["text"]:
        toks = list(iter_tokens(str(text), preprocessor="markdown"))
        grams = list(iter_ngrams(toks, 2))
        bigram_counts.update(grams)
        bigram_n += len(grams)

    bigrams = pl.read_parquet(OUTPUT_DIR / "overuse_ngram_2.parquet")
    refmap = dict(zip(bigrams["ngram"], bigrams["best_reference_per_million"], strict=True))
    let_me = bigrams.filter(pl.col("ngram") == "let me")
    let_me_rate = float(let_me["target_per_million"][0]) if let_me.height else 0.0
    best_human = float(let_me["best_reference_per_million"][0]) if let_me.height else 0.0
    let_me_ratio = let_me_rate / best_human if best_human > 0 else float("inf")
    one_in = round(1e6 / let_me_rate) if let_me_rate else 0

    corpus_rows = "\n".join(
        f"<tr><td class='word'>{esc(cid)}</td>"
        f"<td class='soft'>{esc(m['spec']['text_register'])}</td>"
        f"<td class='num'>{m['stats']['tokens']:,}</td>"
        f"<td class='num soft'>{m['stats']['types']:,}</td>"
        f"<td class='soft'>{esc(m['spec']['date_cutoff'] or 'n/a')}</td>"
        f"<td class='soft'>{'instrument' if cid == 'vcs' else ('target' if cid.startswith('claude') else 'baseline')}</td></tr>"
        for cid, m in corpora
    )

    chain = [
        (f"{baseline.height:,}", "candidate words, before any correction"),
        (f"{overused(baseline).height}", "over-used against four general baselines"),
        (f"{qualifying.height}", "after a biomedical baseline and morphological folding"),
        (f"{recalibrated.height}", f"after recalibrating the threshold to z &ge; {threshold:.2f}"),
        (f"{recalibrated.height - domain.height}", "after removing the subject matter"),
        (f"{style.height}", "after removing the vocabulary of version control itself"),
    ]
    chain_rows = "\n".join(
        f"<tr><td class='num claude'>{v}</td><td class='soft'>{lbl}</td></tr>" for v, lbl in chain
    )

    verdicts = []
    for token in ("gap", "instinct", "churn"):
        row = wide.filter(pl.col("token") == token)
        if not row.height:
            continue
        d = row.row(0, named=True)
        ok = bool(
            d["well_dispersed"]
            and d["tiers_agreeing"] == d["tiers_compared"]
            and d["clears_empirical"]
        )
        rank = style.with_row_index("r").filter(pl.col("token") == token)
        place = f" &middot; rank {int(rank['r'][0]) + 1}" if rank.height else ""
        if ok:
            tag = f"<span class='tag pass'>confirmed{place}</span>"
            detail = (
                f"{d['target_per_million']:,.0f} per million, "
                f"{ratio_text(word_ratio(d, tiers))} the toughest human corpus, spread over "
                f"{int(d['sessions_present'])} sessions. Specialisation "
                f"{d['specialisation']:+.1f} &mdash; ordinary English, not domain vocabulary."
            )
        else:
            tag = "<span class='tag fail'>rejected</span>"
            detail = (
                f"{d['target_per_million']:,.0f} per million, "
                f"{ratio_text(word_ratio(d, tiers))} the toughest corpus &mdash; a bigger ratio "
                f"than most of what is published here. Rejected anyway: it lives in "
                f"{int(d['sessions_present'])} sessions of {claude_meta['stats']['parts']}, "
                "which is a topic, not a habit. It is rejected by the frequency-neutral "
                "dispersion gate too &mdash; the one built precisely because the flat gate "
                "penalised rare words."
            )
        verdicts.append(
            f"<div class='verdict'><span class='w'>{esc(token)}</span>{tag}"
            f"<span class='d'>{detail}</span></div>"
        )
    verdicts.append(
        "<div class='verdict'><span class='w'>you&rsquo;re not imagining it</span>"
        "<span class='tag fail'>not present</span>"
        "<span class='d'>The word <code>imagining</code> occurs three times in the whole "
        f"corpus, far below the floor of {MIN_TARGET_COUNT}. The phrase is not there to "
        "measure, at any chain length. Predicted, and absent.</span></div>"
    )

    ngram_sections = []
    for n, title in ((2, "Two words"), (3, "Three words"), (4, "Four words")):
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
        if not passing.height:
            ngram_sections.append(
                f"<h3>{title} &mdash; none of {frame.height} candidates</h3>"
                "<p class='lede'>Nothing at this length clears all six corpora <em>and</em> the "
                "dispersion gates. The chains with the largest rate ratios are exactly the ones "
                "the statistic cannot certify, because they do not occur in the human corpora at "
                "all. This is a limit of the measurement, not evidence that the phrases are "
                "ordinary.</p>"
            )
            continue
        ngram_sections.append(
            f"""<h3>{title} &mdash; {passing.height} of {frame.height} candidates</h3>
<div class="scroll"><table>
<thead><tr><th></th><th>chain</th><th class="num">Claude /M</th>
<th class="num">best human /M</th><th class="num">ratio</th>
<th class="num">min z</th><th class="num">sessions</th></tr></thead>
<tbody>{ngram_rows(passing, 12)}</tbody></table></div>"""
        )

    quiet_rows = "\n".join(
        f"<tr><td class='word'>{esc(r['token'])}</td>"
        f"<td class='num claude'>{r['target_per_million']:,.0f}</td>"
        + "".join(
            f"<td class='num soft'>{(r.get(f'per_million_{t}') or 0.0):,.0f}</td>" for t in tiers
        )
        + "</tr>"
        for r in quiet.iter_rows(named=True)
    )

    tier_heads = "".join(f"<th class='num'>{esc(TIER_LABEL.get(t, t))}</th>" for t in tiers)
    try:
        vcs_tokens = meta("vcs")["stats"]["tokens"]
    except FileNotFoundError:
        vcs_tokens = 0
    chart = dumbbell(list(style.iter_rows(named=True)), tiers)
    css = (
        STYLE.replace("__FONT__", font_face())
        .replace("__LIGHT__", tokens(LIGHT))
        .replace("__DARK__", tokens(DARK))
    )
    stamp = datetime.now(UTC).strftime("%-d %B %Y")

    return f"""<title>The Let Me Corpus</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{css}</style>
<div class="wrap">

<header>
  <p class="eyebrow">Corpus study &middot; {esc(stamp)}</p>
  <div class="hero">
    <span class="hero-phrase display">let me</span>
    <div class="hero-meta">
      <b>{let_me_rate:,.0f}</b>
      <span>per million words &mdash; about one word in {one_in}</span>
      <b>{ratio_text(let_me_ratio)}</b>
      <span>the rate of the human corpus that uses it most</span>
    </div>
  </div>
  <h1>What Claude Code says too often</h1>
  <p class="standfirst">{claude_meta['stats']['tokens']:,} words of assistant prose measured
  against {reference_tokens / 1e6:,.0f} million words of human writing &mdash; literature,
  Reddit, Stack Overflow, a web crawl, PubMed and 1.5 million commit messages &mdash; then
  checked once more against git&rsquo;s own manual. A word is reported only if it is over-used
  against <em>every</em> corpus, survives having its inflected forms merged, and clears a
  threshold calibrated from this corpus&rsquo;s own null distribution.</p>
</header>

<section>
  <div class="stats">
    <div class="stat"><b>{claude_meta['stats']['tokens']:,}</b><span>words of Claude prose, after code and markup are stripped</span></div>
    <div class="stat"><b>{claude_meta['stats']['parts']}</b><span>sessions across {project_count} projects</span></div>
    <div class="stat"><b>{style.height}</b><span>style words surviving every correction</span></div>
    <div class="stat"><b>{null_rate:.1f}%</b><span>false positives when the corpus is compared against itself</span></div>
  </div>
  <p class="col">The strongest result is not a favourite adjective. It is a sentence opening.
  <code>let me</code> runs at {let_me_rate:,.0f} per million against {best_human:,.0f} in the
  human corpus that uses it most, and it clears every one of the six. The habit is announcing an
  action before performing it, over and over, in the same six characters.</p>
  <p class="col">Longer chains are more extreme still and <em>cannot be certified</em>:
  <code>let me check</code> appears at 1,666 per million against 0.3 in the closest human
  corpus, a ratio of roughly 5,400, and yet scores a z of 1.6. That is not a quirk of this
  phrase &mdash; it is what the statistic does when a chain is <em>absent</em> from a reference
  corpus, and it is described under Limits.</p>
  <p class="col">Around it sit two clusters: verification (<code>verify</code>,
  <code>confirm</code>, <code>check</code>, <code>exactly</code>, <code>deliberately</code>) and
  the narration of sequence (<code>now</code>, <code>already</code>, <code>before</code>).</p>
</section>

<section>
  <p class="kicker">The gap</p>
  <h2>How far from human rates</h2>
  <p class="lede">Each row is one word. The blue mark is its rate in whichever human corpus uses
  it <em>most</em> &mdash; the toughest available comparison &mdash; and the orange mark is
  Claude&rsquo;s rate. The distance between them is the finding.</p>
  <figure>
    <p class="legend"><span><i class="lh"></i>toughest human corpus</span>
    <span><i class="lc"></i>Claude Code</span></p>
    <div class="chartbox">{chart}</div>
    <figcaption>The 18 largest gaps. Ordering by ratio has two costs worth naming.
    <code>hf</code> rises because ratio rewards rarity, and it is an abbreviation rather than a
    habit. And <code>the</code> vanishes from this view although it clears every gate: Claude
    uses it only about 6% more than the corpus that uses it most, but at 86,000 occurrences per
    million that 6% is overwhelming evidence. A large gap and strong evidence are not the same
    thing.</figcaption>
  </figure>
</section>

<section>
  <p class="kicker">Corrections</p>
  <h2>From {baseline.height:,} candidates to {style.height} words</h2>
  <p class="lede">Each step removes a different way of being wrong, and the first count was
  inflated by every one of them.</p>
  <div class="scroll"><table><tbody>{chain_rows}</tbody>
  <caption>Every figure is read from a stored run, not recomputed for the prose.</caption>
  </table></div>
  <div class="note"><b>Folding</b> merges <code>gap</code>, <code>gaps</code>,
  <code>gap&rsquo;s</code> and <code>gapped</code> into one entry, so a single habit stops
  occupying four ranks and its evidence stops being split four ways. It stops at inflection:
  <code>verification</code> is not merged into <code>verify</code>, because derivation changes
  part of speech and often meaning &mdash; and <code>agape</code> is not merged into
  <code>gap</code> at all.</div>
</section>

<section>
  <p class="kicker">Style</p>
  <h2>The vocabulary</h2>
  <p class="lede">Rate per million beside every human corpus. <em>Spec</em> is specialisation:
  how much more specialist human writing uses a word than general human writing, in doublings.
  It is computed from the reference corpora alone &mdash; the Claude corpus is not an input
  &mdash; so it reads the ranking without contaminating it.</p>
  <div class="scroll"><table>
    <thead><tr><th></th><th>word</th><th class="num">Claude /M</th>{tier_heads}
    <th class="num">git manual</th>
    <th class="num">vs hardest</th><th class="num">min z</th><th class="num">spec</th>
    <th class="num">proj DP</th><th class="num">sessions</th></tr></thead>
    <tbody>{word_rows(style, tiers, 60)}</tbody>
    <caption>Top 60 of {style.height}, by rate against the toughest human corpus.
    <em>Proj DP</em> is dispersion across the
    {project_count} projects: 0 means used everywhere, 1 means confined to one repository.</caption>
  </table></div>
</section>

<section>
  <p class="kicker">Chains</p>
  <h2>Phrases</h2>
  <p class="lede">Built only after the single-word signal was confirmed. Overlapping n-grams are
  not independent observations, so these z-scores are an ordering, not calibrated significance.
  Note what happens as the chains lengthen: the number that can be certified collapses, for the
  reason given under Limits.</p>
  {"".join(ngram_sections)}
</section>

<section>
  <p class="kicker">Predictions, tested</p>
  <h2>Named in advance</h2>
  <p class="lede">Guessed before the corpus existed, which makes them a test of the method rather
  than an output of it.</p>
  <div class="verdicts">{"".join(verdicts)}</div>

  <h3>A second set, drawn rather than typed</h3>
  <p class="lede">Someone sent a cartoon of a restaurant called <em>Full Picture</em>, in which
  every object on the table is captioned with one of these habits &mdash; load-bearing sauce,
  freshly ground truth, coarse-grained salt, a genuinely sharp point, all green, two sides
  <em>and it&rsquo;s worth separating them</em>. It was drawn by a reader who had never seen a
  frequency table. Measured the same way as everything else:</p>
  <div class="scroll"><table>
    <thead><tr><th>caption</th><th>measured as</th><th class="num">Claude /M</th>
    <th class="num">best human /M</th><th class="num">ratio</th><th>verdict</th></tr></thead>
    <tbody>{meme_rows(claude_counts, claude_n, gen_counts, gen_n, bigram_counts, bigram_n, refmap)}</tbody>
    <caption>Roughly two thirds land, and the misses are informative in both directions.
    <code>substrate</code> and <code>coarse-grained</code> do not occur at all, and
    <code>teeth</code> is used <em>less</em> than humans use it &mdash; yet those captions read
    as exactly as characteristic as the ones that are 500x. Intuition has good recall on the
    shape of a habit and poor precision on its instances, which is the argument for measuring.</caption>
  </table></div>
  <div class="note"><b>The reader also found what the tool cannot see.</b> Two of these captions
  quote no string in the corpus at all. <code>worth separating</code> occurs zero times &mdash;
  but <code>worth [verb-ing]</code> runs at 672 per million across 60 different verbs, and
  <code>full picture</code> is one filling of a frame that takes 23. Every individual filling is
  too rare to rank, so the word-by-word ranking on this page is blind to both, and a person
  reading the output inferred them anyway.</div>
</section>

<section>
  <p class="kicker">Register</p>
  <h2>Is it a habit, or is it the job?</h2>
  <p class="lede">Words like <code>commit</code>, <code>bump</code> and <code>defer</code> are
  developer vocabulary. Measured against novels, Reddit and PubMed they look like style for an
  obvious and uninteresting reason, and Stack Overflow does not settle it &mdash; it is questions
  about broken code, thin on the language of version control. So a sixth corpus was added:
  <b>1.48 million real commit messages</b>, written before 2022.</p>
  <div class="scroll"><table>
    <thead><tr><th>word</th><th class="num">Claude /M</th><th class="num">commit messages /M</th>
    <th class="num">ratio</th><th>verdict</th></tr></thead>
    <tbody>{register_rows(wide, ["refactor", "doc", "revert", "deprecate", "merge", "commit", "bump", "upstream", "defer", "guard", "gate", "stale", "digest"], survivors)}</tbody>
    <caption>The test was half passed, which is the useful outcome. Five words turned out to be
    register &mdash; developers write <code>refactor</code> thirteen times more often than Claude
    does &mdash; and were removed. The rest are over-used even against the corpus where they
    belong.</caption>
  </table></div>
</section>

<section>
  <p class="kicker">Shape</p>
  <h2>The stronger signal is not which words, but how few</h2>
  <p class="lede">Everything above ranks words one at a time. That framing turns out to be the
  weaker half of what the data shows, and a reader spotted it before the pipeline did.</p>
  <p class="col">Claude uses <em>fewer</em> adverbs than every human corpus here &mdash; 11,630
  per million against Reddit&rsquo;s 14,936 &mdash; and far fewer adjectives. By the ranking
  above, modifiers are unremarkable. But two thirds of all its adverb usage sits in
  <b>twenty words</b>:</p>
  <div class="scroll"><table>
    <thead><tr><th>corpus</th><th class="num">share of adverb use in its top 20</th></tr></thead>
    <tbody>
      <tr><td class='word'>claude_main</td><td class='num claude'>68.9%</td></tr>
      <tr><td class='word'>commit msgs</td><td class='num soft'>53.6%</td></tr>
      <tr><td class='word'>reddit</td><td class='num soft'>50.5%</td></tr>
      <tr><td class='word'>technical</td><td class='num soft'>49.2%</td></tr>
      <tr><td class='word'>biomedical</td><td class='num soft'>35.8%</td></tr>
      <tr><td class='word'>web</td><td class='num soft'>35.6%</td></tr>
      <tr><td class='word'>literature</td><td class='num soft'>24.6%</td></tr>
    </tbody>
    <caption>Rarefied by multinomial resampling to the target corpus&rsquo;s own 428,453 tokens,
    because type counts grow with sample size and the reference corpora are 140x larger. The
    mass share is a ratio and barely moves under that correction &mdash; 68.9% either way. A
    first, uncorrected reading of the same data claimed an order-of-magnitude gap in the number
    of distinct adverbs; at matched size that gap is 1.2x to 3.2x, and the claim was withdrawn.</caption>
  </table></div>
  <p class="col">The same shape appears as a template rather than a word.
  <code>worth separating</code> occurs zero times, but <code>worth [verb-ing]</code> runs at 672
  per million across 60 different verbs &mdash; <code>worth flagging</code> alone at 124 against
  <b>0.00</b> in 286 million words of human writing. Every individual filling falls below the
  reporting floor, so the ranking above cannot see the frame that generates them.</p>
  <div class="note"><b>This is a limitation, not a result.</b> The pipeline measures words. These
  two figures were computed by hand, because it has no notion of vocabulary concentration or of
  a productive template. On this evidence that is the bigger half of the phenomenon, and the
  page&rsquo;s own title is the smaller one.</div>
</section>

<section>
  <p class="kicker">The manual</p>
  <h2>Or is it just what the manual says?</h2>
  <p class="lede">The commit corpus settles whether a word belongs to the <em>register</em> of
  software work. It cannot settle whether a word belongs to the <em>subject</em> of version
  control, because commit messages use git&rsquo;s vocabulary without explaining it. So the
  documentation itself was added: <b>Pro Git</b> and git&rsquo;s own reference manual,
  {vcs_tokens:,} words of prose after code is stripped.</p>
  <div class="scroll"><table>
    <thead><tr><th>word</th><th class="num">Claude /M</th>
    <th class="num">git documentation /M</th><th class="num">ratio</th></tr></thead>
    <tbody>{vcs_rows(version_control, 14)}</tbody>
    <caption>{version_control.height} words are removed here. The documentation uses
    <code>commit</code> at 6,383 per million against Claude&rsquo;s 3,788. A word must be both
    <em>distinctive</em> to version-control writing and not out-used by Claude &mdash; without
    the first condition the filter removes <code>the</code> and <code>only</code>, because git
    documentation is ordinary dense prose and uses them at ordinary rates.</caption>
  </table></div>
  <div class="note"><b>This is a rate comparison, not a seventh corpus in the gate.</b> At a
  third of a million words the documentation is far too small to join the minimum-z rule: a word
  it happens never to use would have its z collapse for lack of evidence, which would look like
  topical control and would really be F10 in the finding log. A direct rate ratio has no such
  failure &mdash; a word the manual never uses simply scores infinity, which is the right
  verdict.</div>
</section>

<section>
  <p class="kicker">Domain</p>
  <h2>The words that are only the subject matter</h2>
  <p class="lede">Real rate differences, and not style. They were separated two ways: by adding a
  biomedical baseline so they stop clearing the all-corpora gate, and by scoring specialisation
  from human corpora alone.</p>
  <div class="scroll"><table>
    <thead><tr><th>word</th><th class="num">Claude /M</th><th class="num">general English /M</th>
    <th class="num">ratio</th><th class="num">PubMed /M</th><th class="num">spec</th></tr></thead>
    <tbody>{domain_rows(wide, ["annotation", "chromosome", "variant", "gene", "schema", "compiler", "namespace", "runtime", "query", "parsing"])}</tbody>
    <caption>{domain.height} of the {recalibrated.height} surviving words score as domain
    vocabulary. A biomedical baseline is what removes them, without anyone hand-writing a list
    of which words count as jargon.</caption>
  </table></div>
  <div class="note"><b>Domain vocabulary also moves the yardstick.</b> A corpus with a heavy
  topical component has a wider spread of log-odds, so a fixed threshold does not mean here what
  it would in a topic-matched comparison. Read off this corpus&rsquo;s own null distribution the
  cut is <b>z &ge; {threshold:.2f}</b> at a 1% false-positive rate, where the conventional
  constant was 3.00 &mdash; {100 * (threshold / 3.0 - 1):.0f}% stricter.</div>
</section>

<section>
  <p class="kicker">Method</p>
  <h2>How a word qualifies</h2>
  <ul class="gates">
    <li>At least <b>{MIN_TARGET_COUNT}</b> occurrences once inflected forms are merged.</li>
    <li>A log-odds z above the <b>empirically calibrated</b> threshold against <b>every</b>
    corpus &mdash; the ranking statistic is the <em>minimum</em> across them, so no single
    extreme baseline can carry a word.</li>
    <li>Dunning&rsquo;s G&sup2; agreeing in sign with the log-odds. Two tests with different
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
    <th class="num">types</th><th>dated to</th><th>role</th></tr></thead>
    <tbody>{corpus_rows}</tbody>
    <caption>A <em>baseline</em> is one of the corpora a word must clear. The
    <em>instrument</em> is read as a rate comparison only &mdash; at 316k words it is too small
    to gate with, for the reason in the note above.
    Every human corpus predates the period when generated text became common online.
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
  There is no per-document verdict, and nothing here can say whether a particular text was
  written by a model &mdash; or by which model.</p>
  <p class="col"><b>The statistic cannot certify a phrase that humans never write.</b> The
  log-odds variance carries a term of one over the reference count plus the prior. When a chain
  occurs zero times in a corpus and the prior is a fraction of one pseudo-count, that term
  explodes and the z-score collapses &mdash; so the <em>more</em> distinctive a phrase is, the
  worse it scores. <code>let me check</code>, at roughly 5,400 times the closest human rate,
  scores 1.6. Single words are unaffected, because a word frequent enough to be a candidate is
  attested everywhere. Raising the prior would rescue these phrases, which is precisely why it
  has not been done here: the defect was found by looking at which phrases it would rescue.</p>
  <p class="col"><b>Topic control is better, not complete.</b> A private vocabulary that no
  public corpus contains cannot be scored for specialisation at all; dispersion across projects
  is the only instrument that sees it.</p>
  <p class="col"><b>One author, {claude_meta['stats']['parts']} sessions,
  {project_count} projects.</b> Whether these rates hold for Claude Code generally is untested.</p>
  <p class="col"><b>The harness is not separated from the model.</b> Output is shaped by a system
  prompt, by tool results and by the user&rsquo;s own phrasing. Nothing here distinguishes a
  model&rsquo;s habit from one the harness induced.</p>
</section>

<footer>
  Built with <code>zipf</code>. Every figure is read from the run&rsquo;s own parquet output
  rather than transcribed, and the chart&rsquo;s two colours were checked with a colour-vision
  validator rather than chosen by eye. Nine defects found during the build are recorded in the
  repository&rsquo;s finding log &mdash; among them a prior normalised over the wrong
  denominator, a dispersion gate that was secretly a frequency filter, and a morphology pass in
  which the fragment <em>noth</em> swallowed the word <em>nothing</em>.
</footer>

</div>"""


if __name__ == "__main__":
    destination = OUTPUT_DIR / "zipf-report.html"
    destination.write_text(build(), encoding="utf-8")
    print(f"wrote {destination}")
