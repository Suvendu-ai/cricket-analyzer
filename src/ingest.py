"""Parses Cricsheet JSON match files into the SQLite database.

Cricsheet JSON structure (format 1.1.0, see https://cricsheet.org/format/json/):
  info.teams, info.venue, info.toss, info.outcome, ...
  innings[] -> overs[] -> deliveries[] with batter/bowler/runs/wickets
"""

import json
from pathlib import Path

from db import get_connection, init_db

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get_or_create(conn, table, name_col, name_val, extra=None):
    """Get an existing row's id by name, or insert it and return the new id."""
    cur = conn.execute(f"SELECT rowid FROM {table} WHERE {name_col} = ?", (name_val,))
    row = cur.fetchone()
    if row:
        return row[0]
    cols = [name_col] + list(extra.keys()) if extra else [name_col]
    vals = [name_val] + list(extra.values()) if extra else [name_val]
    placeholders = ",".join("?" * len(vals))
    cur = conn.execute(
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", vals
    )
    return cur.lastrowid


def _parse_outcome(outcome: dict):
    """Returns (result_type, result_margin) from an info.outcome block."""
    if "by" in outcome:
        by = outcome["by"]
        if "innings" in by:
            return "innings", by.get("runs") or by.get("wickets")
        if "runs" in by:
            return "runs", by["runs"]
        if "wickets" in by:
            return "wickets", by["wickets"]
        return next(iter(by.items()))
    if "result" in outcome:
        return outcome["result"], None  # 'draw', 'tie', 'no result'
    return None, None


def parse_match(conn, path: Path, match_type_label: str) -> bool:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    info = data["info"]
    teams = info["teams"]
    if len(teams) != 2:
        return False  # skip anything that isn't a standard two-team match

    match_id = path.stem

    cur = conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,))
    if cur.fetchone():
        return False  # already ingested

    team1_id = get_or_create(conn, "teams", "name", teams[0])
    team2_id = get_or_create(conn, "teams", "name", teams[1])

    venue_id = None
    if info.get("venue"):
        venue_id = get_or_create(
            conn, "venues", "name", info["venue"], {"city": info.get("city")}
        )

    toss = info.get("toss", {})
    toss_winner_id = (
        get_or_create(conn, "teams", "name", toss["winner"]) if toss.get("winner") else None
    )

    outcome = info.get("outcome", {})
    winner_id = (
        get_or_create(conn, "teams", "name", outcome["winner"])
        if outcome.get("winner")
        else None
    )
    result_type, result_margin = _parse_outcome(outcome)

    dates = info.get("dates", [])

    conn.execute(
        """INSERT INTO matches (
            match_id, match_type, team_type, event_name, match_number, gender,
            date_start, date_end, season, venue_id, team1_id, team2_id,
            toss_winner_id, toss_decision, winner_id, result_type,
            result_margin, player_of_match
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            match_id,
            match_type_label,
            info.get("team_type"),
            info.get("event", {}).get("name"),
            info.get("event", {}).get("match_number"),
            info.get("gender"),
            dates[0] if dates else None,
            dates[-1] if dates else None,
            info.get("season"),
            venue_id,
            team1_id,
            team2_id,
            toss_winner_id,
            toss.get("decision"),
            winner_id,
            result_type,
            result_margin,
            ",".join(info.get("player_of_match", [])),
        ),
    )

    for innings_number, innings in enumerate(data.get("innings", []), start=1):
        batting_team_id = get_or_create(conn, "teams", "name", innings["team"])
        cur = conn.execute(
            "INSERT INTO innings (match_id, innings_number, batting_team_id) VALUES (?,?,?)",
            (match_id, innings_number, batting_team_id),
        )
        innings_id = cur.lastrowid

        for over in innings.get("overs", []):
            over_number = over["over"]
            for ball_in_over, delivery in enumerate(over["deliveries"], start=1):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                extras_type = next(iter(extras), None)
                wickets = delivery.get("wickets", [])
                is_wicket = 1 if wickets else 0
                wicket_kind = wickets[0]["kind"] if wickets else None
                player_out = wickets[0]["player_out"] if wickets else None

                conn.execute(
                    """INSERT INTO deliveries (
                        innings_id, over_number, ball_in_over, batter, bowler,
                        non_striker, runs_batter, runs_extras, runs_total,
                        extras_type, is_wicket, wicket_kind, player_out
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        innings_id,
                        over_number,
                        ball_in_over,
                        delivery["batter"],
                        delivery["bowler"],
                        delivery.get("non_striker"),
                        runs.get("batter", 0),
                        runs.get("extras", 0),
                        runs.get("total", 0),
                        extras_type,
                        is_wicket,
                        wicket_kind,
                        player_out,
                    ),
                )

    return True


def ingest_format(fmt: str, match_type_label: str) -> None:
    folder = RAW_DIR / fmt
    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"No files found in {folder}. Run download_data.py first.")
        return

    conn = get_connection()
    loaded = 0
    for path in files:
        try:
            if parse_match(conn, path, match_type_label):
                loaded += 1
        except Exception as e:
            print(f"  skipped {path.name}: {e}")
    conn.commit()

    matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    deliveries = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    conn.close()

    print(f"Ingested {loaded} new {match_type_label} matches from {folder}")
    print(f"Database now has {matches} matches, {deliveries} deliveries total")


if __name__ == "__main__":
    import sys

    init_db()
    fmt_map = {"test": "Test", "odi": "ODI", "t20i": "T20I"}
    formats = sys.argv[1:] or ["test"]
    for fmt in formats:
        ingest_format(fmt, fmt_map[fmt])