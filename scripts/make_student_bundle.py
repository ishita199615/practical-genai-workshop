"""Build the ZIP handed to workshop students.

Deliberately excludes anything that is a secret, a build artifact, or a local
machine detail. The instructor's ``.env`` holds live API keys and must never
travel in the bundle, so it is excluded by name and the result is re-scanned to
prove no key string escaped.

Run it from the project root::

    .venv\\Scripts\\python.exe scripts/make_student_bundle.py
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_NAME = "practical-genai-workshop-offline.zip"

# Directories never shipped: virtual environments are machine-specific, caches
# and outputs are regenerated, and .git may hold historical secrets.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".claude",
        ".idea",
        ".vscode",
        "node_modules",
        ".ruff_cache",
        ".mypy_cache",
    }
)

# Files never shipped. ".env" is the important one: it holds live keys.
EXCLUDED_FILES: frozenset[str] = frozenset(
    {".env", BUNDLE_NAME, ".DS_Store", "Thumbs.db"}
)

EXCLUDED_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".log"})

# Output packages can contain generated application text; students start clean.
EXCLUDED_RELATIVE: frozenset[str] = frozenset({"output"})

# Shapes of the API keys this project uses, checked against the built bundle.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Firecrawl key", re.compile(rb"fc-[0-9a-f]{16,}")),
    ("Google API key", re.compile(rb"AIza[0-9A-Za-z_\-]{20,}")),
    ("Google OAuth token", re.compile(rb"AQ\.[0-9A-Za-z_\-]{20,}")),
)


def should_include(path: Path) -> bool:
    """True when a path belongs in the student bundle."""
    relative = path.relative_to(PROJECT_ROOT)
    parts = relative.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if parts and parts[0] in EXCLUDED_RELATIVE and path.name != ".gitkeep":
        return False
    if path.name in EXCLUDED_FILES:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return True


def collect_files() -> list[Path]:
    """Return every file that belongs in the bundle, sorted for a stable ZIP."""
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and should_include(path)
    )


def scan_for_secrets(bundle: Path) -> list[str]:
    """Re-open the finished ZIP and look for anything key-shaped.

    A belt-and-braces check: exclusion by filename is easy to get wrong, and a
    leaked key in a file handed to thirty students is not recoverable.
    """
    findings: list[str] = []
    with zipfile.ZipFile(bundle) as archive:
        for entry in archive.namelist():
            try:
                blob = archive.read(entry)
            except Exception:  # noqa: BLE001 - unreadable entry is not a secret
                continue
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(blob):
                    findings.append(f"{label} found in {entry}")
    return findings


def main() -> int:
    """Build the bundle and refuse to leave one behind that contains a key."""
    bundle_path = PROJECT_ROOT / BUNDLE_NAME
    files = collect_files()
    if not files:
        print("No files collected; is the script running from the project?")
        return 1

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("practical-genai-workshop") / path.relative_to(PROJECT_ROOT))

    findings = scan_for_secrets(bundle_path)
    if findings:
        bundle_path.unlink(missing_ok=True)
        print("SECRET DETECTED - bundle deleted, nothing was written:")
        for finding in findings:
            print(f"  {finding}")
        return 2

    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    print(f"Built {bundle_path.name}")
    print(f"  files : {len(files)}")
    print(f"  size  : {size_mb:.1f} MB")
    print("  secret scan: clean (no API keys found in the archive)")
    print()
    print("Share this file with students. It excludes .env, .venv, and output/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
