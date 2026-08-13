"""Build the version-control documentation corpus.

    uv run python scripts/fetch_vcs_corpus.py

Two freely licensed sources, both AsciiDoc, downloaded as release tarballs rather than cloned:

* **Pro Git**, Chacon & Straub — CC BY-NC-SA 3.0, from the `progit/progit2` book source.
* **git's own reference documentation** — GPL-2.0, `Documentation/*.txt` from a pinned git tag.

Why this is not a Hugging Face tier like the others: it is not on Hugging Face, and it is small
enough that the download is a couple of tarballs rather than a dataset. `zipf fetch` therefore
does not build it, and `zipf count --corpus vcs` refuses with a pointer here if it is missing —
rather than silently counting a corpus that is not there.

**This corpus is an instrument, not a baseline.** It answers "is this word simply the vocabulary
of version control?" — it is deliberately *not* one of the tiers a word must clear, because at
under a million words a great many candidate words occur in it zero times, and a zero reference
count collapses the log-odds z regardless of the effect size. That mechanism is written up as
F10 in `docs/dogfooding.md`. Used as a gate it would drop words for lack of evidence while
looking like topical control; used as an instrument it answers the question directly.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import urllib.request

from zipf.paths import INPUT_DIR

logger = logging.getLogger(__name__)

PROGIT_URL = "https://github.com/progit/progit2/archive/refs/heads/main.tar.gz"
GIT_TAG = "v2.43.0"
GIT_URL = f"https://github.com/git/git/archive/refs/tags/{GIT_TAG}.tar.gz"

DESTINATION = INPUT_DIR / "vcs" / "vcs_docs.jsonl"

#: Files in the git tree that are documentation prose rather than reference material we want.
GIT_DOC_SKIP = ("RelNotes", "technical/api-index", "CodingGuidelines")


def fetch(url: str) -> tarfile.TarFile:
    logger.info("downloading %s", url)
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URLs above
        payload = response.read()
    logger.info("  %.1f MB", len(payload) / 1e6)
    return tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz")


def progit_documents(archive: tarfile.TarFile) -> list[dict]:
    """Chapter files from the book source, excluding the repo's own meta-documentation."""
    out = []
    for member in archive.getmembers():
        name = member.name
        if not member.isfile() or not name.endswith(".asc"):
            continue
        # `book/` holds the per-section sources; the top-level files are assembly stubs, and
        # README/LICENSE/TRANSLATION_NOTES are about the book rather than about git.
        if "/book/" not in name:
            continue
        data = archive.extractfile(member)
        if data is None:
            continue
        out.append(
            {
                "source": "progit2",
                "path": name.split("/", 1)[-1],
                "text": data.read().decode("utf-8", errors="replace"),
            }
        )
    return out


def git_documents(archive: tarfile.TarFile) -> list[dict]:
    """`Documentation/*.txt` from the pinned git tag."""
    out = []
    for member in archive.getmembers():
        name = member.name
        if not member.isfile() or "/Documentation/" not in name or not name.endswith(".txt"):
            continue
        if any(skip in name for skip in GIT_DOC_SKIP):
            continue
        data = archive.extractfile(member)
        if data is None:
            continue
        out.append(
            {
                "source": f"git {GIT_TAG}",
                "path": name.split("/Documentation/", 1)[-1],
                "text": data.read().decode("utf-8", errors="replace"),
            }
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    documents = progit_documents(fetch(PROGIT_URL)) + git_documents(fetch(GIT_URL))
    if not documents:
        raise SystemExit("no documents extracted; refusing to write an empty corpus")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    # Sorted so two runs over the same releases produce byte-identical output.
    documents.sort(key=lambda d: (d["source"], d["path"]))
    with DESTINATION.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    words = sum(len(d["text"].split()) for d in documents)
    by_source: dict[str, int] = {}
    for d in documents:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
    logger.info("wrote %s", DESTINATION)
    logger.info("  %d documents, ~%d whitespace-words before code stripping", len(documents), words)
    for source, count in sorted(by_source.items()):
        logger.info("  %-14s %d files", source, count)


if __name__ == "__main__":
    main()
