"""Check a resume JSON file before the app tries to load it.

Hand-written JSON fails in boring ways — a trailing comma, a missing bullet ID,
a typo in a field name — and Pydantic's raw errors are not friendly to someone
who has never seen them. This turns those into plain instructions.

Run it on your own file::

    python scripts/check_resume.py data/my_resume.json

Add ``--fix-ids`` to write missing bullet and entry IDs into the file. Every
bullet needs a stable ID because the truthfulness validator uses it to trace a
revised line back to the original you wrote.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.resume import ResumeProfile  # noqa: E402

#: Fields the template ships with, so an untouched copy can be spotted.
PLACEHOLDERS: dict[str, str] = {
    "name": "Your Name",
    "email": "you@example.com",
}


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read the file, returning either the parsed object or a readable error."""
    if not path.exists():
        return None, f"No file at {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        hint = ""
        if "Expecting property name" in exc.msg:
            hint = "  Likely a trailing comma after the last item in a list or object."
        elif "Expecting value" in exc.msg:
            hint = "  Likely a missing value, or a stray comma."
        return None, (
            f"{path} is not valid JSON.\n"
            f"  Line {exc.lineno}, column {exc.colno}: {exc.msg}.{hint}"
        )


def assign_missing_ids(payload: dict[str, Any]) -> list[str]:
    """Fill in any missing entry or bullet IDs, reporting what changed."""
    changes: list[str] = []
    for section, prefix in (("experience", "experience"), ("projects", "project")):
        for index, entry in enumerate(payload.get(section) or [], start=1):
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if not entry_id:
                entry_id = f"{prefix}_{index}"
                entry["id"] = entry_id
                changes.append(f"added id '{entry_id}'")
            for b_index, bullet in enumerate(entry.get("bullets") or [], start=1):
                if isinstance(bullet, dict) and not bullet.get("id"):
                    bullet_id = f"{entry_id}_bullet_{b_index}"
                    bullet["id"] = bullet_id
                    changes.append(f"added id '{bullet_id}'")
    return changes


def duplicate_ids(profile: ResumeProfile) -> list[str]:
    """Return any bullet ID used more than once.

    Duplicates silently break claim tracing: two bullets answering to one ID
    means a revision cannot be attributed to the right original.
    """
    seen: set[str] = set()
    dupes: list[str] = []
    for entry in profile.experience:
        for bullet in entry.bullets:
            if bullet.id in seen:
                dupes.append(bullet.id)
            seen.add(bullet.id)
    for project in profile.projects:
        for bullet in project.get("bullets", []) or []:
            bullet_id = bullet.get("id")
            if bullet_id:
                if bullet_id in seen:
                    dupes.append(bullet_id)
                seen.add(bullet_id)
    return dupes


def advisories(profile: ResumeProfile) -> list[str]:
    """Return non-fatal notes that would weaken scoring."""
    notes: list[str] = []
    if not profile.skills:
        notes.append("No skills listed. Skill coverage is 45% of the match score.")
    if not profile.experience and not profile.projects:
        notes.append(
            "No experience or projects. Every claim the agent makes must trace "
            "to a bullet, so it will have almost nothing to work with."
        )
    if not profile.target_roles:
        notes.append("No target_roles. Role alignment is 15% of the match score.")
    if len(profile.professional_summary.split()) < 8:
        notes.append("The professional summary is very short.")
    for field, placeholder in PLACEHOLDERS.items():
        if getattr(profile, field, None) == placeholder:
            notes.append(f"'{field}' still holds the template value {placeholder!r}.")
    return notes


def main() -> int:
    """Validate the file and print either the problems or a summary."""
    parser = argparse.ArgumentParser(description="Validate a resume JSON file.")
    parser.add_argument("path", nargs="?", default="data/my_resume.json")
    parser.add_argument(
        "--fix-ids",
        action="store_true",
        help="write missing entry and bullet IDs back into the file",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    payload, error = load_json(path)
    if error:
        print(f"FAILED\n\n{error}")
        return 1
    assert payload is not None

    if args.fix_ids:
        changes = assign_missing_ids(payload)
        if changes:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {len(changes)} missing ID(s) to {path.name}:")
            for change in changes:
                print(f"  {change}")
            print()

    try:
        profile = ResumeProfile.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - reformatted for a human
        print("FAILED - the file is valid JSON but not a valid resume.\n")
        for line in str(exc).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("For further information"):
                continue
            # Pydantic appends "[type=..., input_value=..., input_type=...]",
            # which would print the resume's own contents to the terminal.
            cleaned = re.sub(r"\s*\[type=.*$", "", line)
            if "Field required" in stripped:
                print(f"{cleaned}   <- add this field")
            else:
                print(cleaned)
        print("\nCompare against data/my_resume.template.json.")
        return 1

    dupes = duplicate_ids(profile)
    if dupes:
        print("FAILED - duplicate bullet IDs, which breaks claim tracing:\n")
        for dupe in sorted(set(dupes)):
            print(f"  '{dupe}' is used more than once")
        print("\nEvery bullet needs its own ID. --fix-ids will not repair this.")
        return 1

    bullets = profile.bullet_index()
    print(f"OK - {path.name} is a valid resume.\n")
    print(f"  Name           : {profile.name}")
    print(f"  Location       : {profile.location}")
    print(f"  Target roles   : {', '.join(profile.target_roles) or 'none'}")
    print(f"  Skills         : {len(profile.skills)}")
    print(f"  Experience     : {len(profile.experience)} entr(ies)")
    print(f"  Projects       : {len(profile.projects)}")
    print(f"  Traceable bullets: {len(bullets)}")

    notes = advisories(profile)
    if notes:
        print("\nWorth improving (not errors):")
        for note in notes:
            print(f"  - {note}")

    relative = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    print(f"\nTo use it, set this in your .env file:\n  RESUME_FILE={relative.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
