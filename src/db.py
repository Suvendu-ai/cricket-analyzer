"""SQLite schema for the cricket analyzer's core data layer.

Shared by all three formats (Test, ODI, T20I) -- match_type on the
`matches` row is what distinguishes them.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cricket.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS venues (
    venue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    city TEXT,
    UNIQUE(name, city)
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    match_type TEXT NOT NULL,
    team_type TEXT,
    event_name TEXT,
    match_number INTEGER,
    gender TEXT,
    date_start TEXT,
    date_end TEXT,
    season TEXT,
    venue_id INTEGER REFERENCES venues(venue_id),
    team1_id INTEGER REFERENCES teams(team_id),
    team2_id INTEGER REFERENCES teams(team_id),
    toss_winner_id INTEGER REFERENCES teams(team_id),
    toss_decision TEXT,
    winner_id INTEGER REFERENCES teams(team_id),
    result_type TEXT,
    result_margin INTEGER,
    player_of_match TEXT
);

CREATE TABLE IF NOT EXISTS innings (
    innings_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    innings_number INTEGER NOT NULL,
    batting_team_id INTEGER REFERENCES teams(team_id),
    UNIQUE(match_id, innings_number)
);

CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    innings_id INTEGER NOT NULL REFERENCES innings(innings_id),
    over_number INTEGER NOT NULL,
    ball_in_over INTEGER NOT NULL,
    batter TEXT NOT NULL,
    bowler TEXT NOT NULL,
    non_striker TEXT,
    runs_batter INTEGER NOT NULL DEFAULT 0,
    runs_extras INTEGER NOT NULL DEFAULT 0,
    runs_total INTEGER NOT NULL DEFAULT 0,
    extras_type TEXT,
    is_wicket INTEGER NOT NULL DEFAULT 0,
    wicket_kind TEXT,
    player_out TEXT
);

CREATE INDEX IF NOT EXISTS idx_matches_type ON matches(match_type);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date_start);
CREATE INDEX IF NOT EXISTS idx_innings_match ON innings(match_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_innings ON deliveries(innings_id);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Schema ready at {DB_PATH}")


if __name__ == "__main__":
    init_db()