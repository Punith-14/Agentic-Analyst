"""Which database the tools are pointed at. One source of truth.

    from tools.db import set_db, current_db
    set_db("/uploads/sales.db")        # a user's own file
    set_db("chinook_1")                # shorthand for data/db/chinook_1.db

THERE IS NO DEFAULT, AND THAT IS DELIBERATE.

The tools used to fall back to data/db/analytics.db — a generated demo
database — and `_ensure_db_exists()` would create it if it was missing. So a
query against a database nobody had selected would silently succeed against
invented data, and the agent would answer confidently:

    "North America had the highest sales at $180,500."

That looks like a real answer. It isn't. If a user's upload fails, or the
frontend forgets to select their file, the right outcome is an error telling
them to provide a dataset — not a fabricated number nobody catches.

So: nothing selected -> every tool returns status="error" with a message the
UI can show. Demo data is still available, but it has to be asked for by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from contracts import ToolResult

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "db"

_current: Optional[str] = None


def set_db(target: str | Path) -> str:
    """Point the tools at a database. Returns the resolved path.

    `target` is either:
      - a path to any SQLite file          "/uploads/sales.db"
      - the name of one already in data/db "chinook_1"

    The name form is only a convenience for the benchmark databases; nothing
    about the tools is specific to them.
    """
    global _current

    named = DB_DIR / f"{target}.db"
    path = named if named.exists() else Path(target)

    if not path.exists():
        raise FileNotFoundError(
            f"no database at {path}. Pass a path to a .db file, or one of: "
            f"{', '.join(sorted(p.stem for p in DB_DIR.glob('*.db'))) or '(none)'}")

    _current = str(path)
    return _current


def clear_db() -> None:
    """Forget the selection. Mostly for tests."""
    global _current
    _current = None


def current_db() -> Optional[str]:
    return _current


def resolve(db_path: str | None) -> Optional[str]:
    """An explicit db_path wins; otherwise use the selected one."""
    return db_path if db_path else _current


def no_database(tool: str) -> ToolResult:
    """The error every tool returns when nothing has been selected.

    Phrased for the person reading it in the UI, not for a developer reading
    a stack trace.
    """
    return ToolResult(
        status="error",
        tool=tool,
        error="No dataset provided. Upload a database or CSV file to continue.",
        error_full="tools.db.current_db() is None — set_db() was never called",
        error_category="permission",
        hint="Upload a .db or .csv file, or pass an explicit db_path.",
    )
