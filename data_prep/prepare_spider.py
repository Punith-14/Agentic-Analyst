"""Turn the Spider download into the databases and task suite we run against.

Databases are kept separate rather than merged. Merging the three would put ~35
tables into every prompt (~1500 tokens of mostly irrelevant schema) versus ~500
for one. It also makes the held-out-domain split fall out naturally.

    python data_prep/prepare_spider.py --spider-dir Data/spider_data --list
    python data_prep/prepare_spider.py --spider-dir Data/spider_data \
        --dbs formula_1,college_2 --holdout chinook_1 --n 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# SQL analysis helpers
# --------------------------------------------------------------------------

def sql_difficulty(sql: str) -> str:
    """Difficulty from SQL shape.

    Derived rather than read from Spider so it stays consistent across the
    train and dev files, which label things differently.
    """
    s = sql.lower()
    score = 0
    score += 2 * len(re.findall(r"\bjoin\b", s))
    score += 2 * (s.count("select") - 1)          # subqueries
    score += 1 if "group by" in s else 0
    score += 1 if "having" in s else 0
    score += 1 if "order by" in s else 0
    score += 1 if re.search(r"\b(union|intersect|except)\b", s) else 0
    score += 1 if re.search(r"\b(avg|sum|min|max|count)\s*\(", s) else 0
    score += 1 if re.search(r"\bnot\s+in\b|\bexists\b", s) else 0
    if score <= 1:
        return "easy"
    if score <= 4:
        return "medium"
    return "hard"


def sql_tags(sql: str) -> list[str]:
    s = sql.lower()
    tags = []
    if re.search(r"\b(avg|sum|min|max|count)\s*\(", s):
        tags.append("aggregation")
    if "join" in s:
        tags.append("join")
    if "group by" in s:
        tags.append("groupby")
    if "order by" in s:
        tags.append("sort")
    if "where" in s:
        tags.append("filter")
    if s.count("select") > 1:
        tags.append("subquery")
    if re.search(r"\b(union|intersect|except)\b", s):
        tags.append("setop")
    if "having" in s:
        tags.append("having")
    return tags or ["simple"]


def join_count(sql: str) -> int:
    return len(re.findall(r"\bjoin\b", sql.lower()))


def table_count(sql: str) -> int:
    s = sql.lower()
    names = re.findall(r"\bfrom\s+([a-zA-Z_]\w*)", s) + re.findall(r"\bjoin\s+([a-zA-Z_]\w*)", s)
    return len(set(names))


# --------------------------------------------------------------------------
# reading the Spider download
# --------------------------------------------------------------------------

QUESTION_FILES = ["train_spider.json", "dev.json", "train_others.json"]


def load_questions(spider_dir: Path) -> list[dict]:
    rows: list[dict] = []
    found: list[str] = []
    for name in QUESTION_FILES:
        p = spider_dir / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows.extend(json.load(f))
            found.append(name)
    if not rows:
        raise SystemExit(
            f"No Spider question files under {spider_dir}. Looked for {QUESTION_FILES}."
        )
    print(f"  loaded {len(rows):,} questions from {found}")
    return rows


def gold_of(row: dict) -> str:
    """Spider calls it 'query', BIRD calls it 'SQL'."""
    return (row.get("query") or row.get("SQL") or "").strip()


def db_file(spider_dir: Path, db_id: str) -> Path | None:
    for sub in ("database", "test_database"):
        p = spider_dir / sub / db_id / f"{db_id}.sqlite"
        if p.exists():
            return p
    return None


def read_schema(db_path: Path) -> dict:
    """Tables, columns, FKs and row counts."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    schema: dict = {"tables": {}, "foreign_keys": [], "total_rows": 0}
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for t in tables:
        cols = [{"name": r[1], "type": r[2] or "TEXT", "pk": bool(r[5])}
                for r in con.execute(f'PRAGMA table_info("{t}")')]
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except sqlite3.Error:
            n = 0
        schema["tables"][t] = {"columns": cols, "rows": n}
        schema["total_rows"] += n
        for r in con.execute(f'PRAGMA foreign_key_list("{t}")'):
            schema["foreign_keys"].append(
                {"from_table": t, "from_col": r[3], "to_table": r[2], "to_col": r[4]})
    con.close()
    return schema


def db_overview(spider_dir: Path, rows: list[dict]) -> list[dict]:
    by_db = Counter(r["db_id"] for r in rows)
    out = []
    for db_id, n_q in by_db.items():
        f = db_file(spider_dir, db_id)
        if not f:
            continue
        try:
            s = read_schema(f)
        except sqlite3.Error:
            continue
        out.append({
            "db_id": db_id,
            "questions": n_q,
            "tables": len(s["tables"]),
            "rows": s["total_rows"],
            "fks": len(s["foreign_keys"]),
        })
    out.sort(key=lambda d: (-d["tables"], -d["questions"]))
    return out


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

def copy_db(spider_dir: Path, db_id: str, out_dir: Path) -> Path:
    src = db_file(spider_dir, db_id)
    if not src:
        raise SystemExit(f"No sqlite file for {db_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{db_id}.db"
    shutil.copyfile(src, dst)

    # index the obvious keys so the agent's queries aren't slow
    con = sqlite3.connect(dst)
    for t in [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")]:
        for c in [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]:
            if c.lower().endswith("id") or c.lower() in ("date", "year", "name"):
                try:
                    con.execute(f'CREATE INDEX IF NOT EXISTS "ix_{t}_{c}" ON "{t}"("{c}")')
                except sqlite3.Error:
                    pass
    con.commit()
    con.close()
    return dst


def curate(rows: list[dict], db_paths: dict[str, Path], n: int,
           split: str, seed: int = 23) -> list[dict]:
    """Pick n questions balanced across difficulty.

    Drops anything whose gold SQL doesn't execute — a few in Spider don't.
    """
    random.seed(seed)
    db_ids = list(db_paths)
    pool = [r for r in rows if r["db_id"] in db_ids]

    cons = {d: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for d, p in db_paths.items()}

    buckets: dict[str, list[dict]] = defaultdict(list)
    checked = failed = 0
    for r in pool:
        gold = gold_of(r)
        if not gold:
            continue
        checked += 1
        try:
            cons[r["db_id"]].execute(gold).fetchall()
        except sqlite3.Error:
            failed += 1
            continue
        buckets[sql_difficulty(gold)].append({
            "question": r["question"].strip(),
            "gold_sql": " ".join(gold.split()),
            "db": r["db_id"],
        })
    for c in cons.values():
        c.close()

    usable = sum(len(v) for v in buckets.values())
    print(f"    {checked:,} checked · {failed:,} failed to execute · {usable:,} usable")

    want = {"easy": round(n * 0.375), "medium": round(n * 0.375)}
    want["hard"] = n - want["easy"] - want["medium"]

    picked: list[dict] = []
    for diff in ("easy", "medium", "hard"):
        avail = buckets.get(diff, [])
        random.shuffle(avail)
        take = avail[: want[diff]]
        if len(take) < want[diff]:
            print(f"    !! only {len(take)} {diff} available (wanted {want[diff]})")
        picked.extend({**t, "difficulty": diff} for t in take)

    random.shuffle(picked)
    prefix = "h" if split == "holdout" else "t"
    tasks = []
    for i, t in enumerate(picked, 1):
        tasks.append({
            "task_id": f"{prefix}{i:03d}",
            "question": t["question"],
            "gold_sql": t["gold_sql"],
            "difficulty": t["difficulty"],
            "db": t["db"],
            "split": split,
            "tags": sql_tags(t["gold_sql"]),
            "source": "spider",
            "n_tables": table_count(t["gold_sql"]),
            "n_joins": join_count(t["gold_sql"]),
        })
    return tasks


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-dir", required=True, type=Path)
    ap.add_argument("--out-db-dir", type=Path, default=Path("data/db"))
    ap.add_argument("--out-tasks", type=Path, default=Path("data/tasks/task_suite.json"))
    ap.add_argument("--out-schemas", type=Path, default=Path("data/tasks/schemas.json"))
    ap.add_argument("--dbs", type=str, default="", help="comma-separated main db_ids")
    ap.add_argument("--holdout", type=str, default="",
                    help="db_id reserved for the generalisation test — never trained on")
    ap.add_argument("--n", type=int, default=40, help="questions for the main split")
    ap.add_argument("--n-holdout", type=int, default=20)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    print("Reading Spider…")
    rows = load_questions(args.spider_dir)

    if args.list:
        ov = db_overview(args.spider_dir, rows)
        print(f"\n{'db_id':<30}{'questions':>10}{'tables':>8}{'rows':>10}{'FKs':>6}")
        print("-" * 64)
        for d in ov[:40]:
            flag = "   << EMPTY" if d["rows"] == 0 else ""
            print(f"{d['db_id']:<30}{d['questions']:>10}{d['tables']:>8}"
                  f"{d['rows']:>10}{d['fks']:>6}{flag}")
        print("\nAvoid EMPTY databases — they have schemas but no data.")
        return

    if not args.dbs:
        raise SystemExit("Pass --dbs a,b  (and optionally --holdout c). Use --list to browse.")

    main_ids = [d.strip() for d in args.dbs.split(",") if d.strip()]
    hold_id = args.holdout.strip()

    print("\nCopying databases…")
    db_paths: dict[str, Path] = {}
    for db_id in main_ids + ([hold_id] if hold_id else []):
        p = copy_db(args.spider_dir, db_id, args.out_db_dir)
        s = read_schema(p)
        db_paths[db_id] = p
        role = "HOLDOUT" if db_id == hold_id else "train"
        print(f"  {db_id:<28} {len(s['tables']):>3} tables  "
              f"{s['total_rows']:>8,} rows  {len(s['foreign_keys']):>3} FKs   [{role}]")

    # schema summary — used to build prompts and by C's knowledge graph
    schemas = {d: read_schema(p) for d, p in db_paths.items()}
    args.out_schemas.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_schemas, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2)

    print("\nCurating main task suite…")
    tasks = curate(rows, {d: db_paths[d] for d in main_ids}, args.n, "main")

    if hold_id:
        print("Curating holdout task suite…")
        tasks += curate(rows, {hold_id: db_paths[hold_id]}, args.n_holdout, "holdout")

    with open(args.out_tasks, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"\n  {len(tasks)} tasks -> {args.out_tasks}")
    print(f"  schemas    -> {args.out_schemas}")
    print(f"  databases  -> {args.out_db_dir}/")
    print(f"  difficulty: {dict(Counter(t['difficulty'] for t in tasks))}")
    print(f"  split:      {dict(Counter(t['split'] for t in tasks))}")
    print("\nSanity-check a few questions by hand before you trust them.")


if __name__ == "__main__":
    main()
