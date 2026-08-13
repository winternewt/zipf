"""Regenerate `docs/CORPORA.md` from the metadata written by the last run.

Operational one-off, not importable code. It exists so that no corpus number in the docs is
ever typed by hand: every figure below is read from `data/output/meta_*.json`, which is
written by the counting step itself.

    uv run python scripts/build_corpora_doc.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from zipf.models import REFERENCE_TIERS
from zipf.paths import REPO_ROOT
from zipf.pipeline import meta_path

ORDER = ("claude_main", "claude_sidechain", *REFERENCE_TIERS)

HEADER = """# Reference corpora

Provenance, licence, date cutoff and known contamination of every corpus. **Generated from
`data/output/meta_*.json` by `scripts/build_corpora_doc.py`** — do not hand-edit the numbers,
regenerate them.

Generated {stamp}.

## Sizes as built

| corpus | register | tokens | types | parts | documents | capped |
|---|---|---:|---:|---:|---:|---|
"""


def main() -> None:
    rows: list[str] = []
    details: list[str] = []
    for corpus_id in ORDER:
        path = meta_path(corpus_id)
        if not path.exists():
            rows.append(f"| `{corpus_id}` | — | *not built* | | | | |")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec, stats = payload["spec"], payload["stats"]
        capped = payload.get("capped")
        rows.append(
            f"| `{corpus_id}` | {spec['text_register']} | {stats['tokens']:,} | "
            f"{stats['types']:,} | {stats['parts']:,} | {stats['documents']:,} | "
            f"{'yes' if capped else 'no'} |"
        )
        details.append(
            f"### `{corpus_id}`\n\n"
            f"- **Source:** {spec['source']}\n"
            f"- **Licence:** {spec['licence']}\n"
            f"- **Date cutoff:** {spec['date_cutoff'] or 'unknown (not the same as none applied)'}\n"
            f"- **Contamination:** {spec['contamination_note']}\n"
            f"- **Built:** {stats['built_at']}\n"
        )

    body = HEADER.format(stamp=datetime.now(UTC).isoformat(timespec="seconds"))
    body += "\n".join(rows)
    body += (
        "\n\nA capped tier stopped at the token cap rather than exhausting its files; the cap "
        "is recorded in the metadata and is never applied silently.\n\n"
        "## Per corpus\n\n"
    )
    body += "\n".join(details)
    body += (
        "\n## A note on the direction of contamination\n\n"
        "Every human tier is dated to before generated text became common. Where any "
        "contamination remains it makes a baseline **more** Claude-like, which **shrinks** the "
        "measured gap. The bias runs toward the null, so surviving findings are conservative.\n\n"
        "The literary tier carries the opposite risk: archaic vocabulary makes ordinary modern "
        "words look novel, so it over-reports. That is exactly why a word must clear every "
        "tier rather than any one.\n"
    )

    destination = REPO_ROOT / "docs" / "CORPORA.md"
    destination.write_text(body, encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
