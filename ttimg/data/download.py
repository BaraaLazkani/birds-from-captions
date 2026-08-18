"""Resumable file download with mirror fallback.

Kept dependency-free (urllib only) so data preparation does not drag in a
request stack we would otherwise never use.
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

_USER_AGENT = "ttimg/0.1 (text-to-image research project)"


def _report(name: str, done: int, total: int) -> None:
    if total > 0:
        pct = 100.0 * done / total
        bar = "#" * int(pct // 4)
        sys.stdout.write(f"\r[download] {name} {pct:5.1f}% {done / 1e6:8.1f}/{total / 1e6:.1f} MB {bar:<25}")
    else:
        sys.stdout.write(f"\r[download] {name} {done / 1e6:8.1f} MB")
    sys.stdout.flush()


def download_file(urls: str | tuple[str, ...], dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Download the first URL that works to ``dest``.

    Returns immediately if ``dest`` already exists, so callers can treat this as
    "ensure present".  A partial download is written to ``dest.part`` and only
    moved into place once complete, meaning an interrupted run can never leave a
    truncated archive that later fails to extract with a confusing error.
    """
    dest = Path(dest)
    if dest.exists():
        print(f"[download] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    if isinstance(urls, str):
        urls = (urls,)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    errors: list[str] = []

    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                done = 0
                with open(part, "wb") as handle:
                    while chunk := response.read(chunk_size):
                        handle.write(chunk)
                        done += len(chunk)
                        _report(dest.name, done, total)
            sys.stdout.write("\n")

            if total and done < total:
                raise OSError(f"truncated: got {done} of {total} bytes")

            shutil.move(str(part), str(dest))
            return dest

        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            sys.stdout.write("\n")
            errors.append(f"  {url}\n    -> {exc}")
            part.unlink(missing_ok=True)
            continue

    raise RuntimeError(f"could not download {dest.name}; all mirrors failed:\n" + "\n".join(errors))
