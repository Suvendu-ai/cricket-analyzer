"""
Leakage-safe feature engineering for the Test match win/draw/loss model.

For every Test match, builds TWO rows -- one from each team's perspective --
so the model learns general "for_team vs opp_team" patterns instead of being
anchored to which team happened to be listed first in the data.

Every feature is computed using only matches strictly BEFORE the match being
featurized ("as of" the match date). Nothing here is allowed to see the
result of the match it's predicting, or any match that happened after it.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from db import get_connection

NEUTRAL_PRIOR = 0.5  # used when a team/pairing/venue has no prior history yet
RECENT_N = 10  # rolling window size for "recent form"
EXCLUDE_RESULTS = ("no result",)  # abandoned matches carry no real signal


def _team_matches(conn, team_id, before_date):
    """A team's decided matches strictly before `before_date`, oldest first."""
    placeholders = ",".join("?" * len(EXCLUDE_RESULTS))
    return conn.execute(
        f"""
        SELECT match_id, team1_id, team2_id, winner_id, result_type
        FROM matches
        WHERE (team1_id = ? OR team2_id = ?)
          AND date_start < ?
          AND result_type IS NOT NULL
          AND result_type NOT IN ({placeholders})
        ORDER BY date_start
        """,
        (team_id, team_id, before_date, *EXCLUDE_RESULTS),
    ).fetchall()


def _team_venue_matches(conn, team_id, venue_id, before_date):
    placeholders = ",".join("?" * len(EXCLUDE_RESULTS))
    return conn.execute(
        f"""
        SELECT winner_id
        FROM matches
        WHERE (team1_id = ? OR team2_id = ?)
          AND venue_id = ?
          AND date_start < ?
          AND result_type IS NOT NULL
          AND result_type NOT IN ({placeholders})
        """,
        (team_id, team_id, venue_id, before_date, *EXCLUDE_RESULTS),
    ).fetchall()


def _venue_matches(conn, venue_id, before_date):
    """All decided matches at a venue (either team), strictly before `before_date`."""
    placeholders = ",".join("?" * len(EXCLUDE_RESULTS))
    return conn.execute(
        f"""
        SELECT result_type
        FROM matches
        WHERE venue_id = ?
          AND date_start < ?
          AND result_type IS NOT NULL
          AND result_type NOT IN ({placeholders})
        """,
        (venue_id, before_date, *EXCLUDE_RESULTS),
    ).fetchall()


def _win_rate(rows, team_id):
    if not rows:
        return NEUTRAL_PRIOR
    wins = sum(1 for r in rows if r["winner_id"] == team_id)
    return wins / len(rows)


def _draw_rate(rows):
    if not rows:
        return NEUTRAL_PRIOR
    draws = sum(1 for r in rows if r["result_type"] in ("draw", "tie"))
    return draws / len(rows)


def _outcome_for(match_row, team_id):
    if match_row["result_type"] in ("draw", "tie") or match_row["winner_id"] is None:
        return "draw"
    return "win" if match_row["winner_id"] == team_id else "loss"


def _batted_first(match_row, team_id):
    """Whether `team_id` was the team that actually batted first -- correct
    regardless of who won the toss, unlike a raw "toss decision" field."""
    toss_winner, decision = match_row["toss_winner_id"], match_row["toss_decision"]
    if toss_winner is None or decision is None:
        return 0  # missing toss info (rare); neutral fallback
    if decision == "bat":
        batting_first = toss_winner
    else:
        batting_first = match_row["team2_id"] if toss_winner == match_row["team1_id"] else match_row["team1_id"]
    return int(batting_first == team_id)


def build_dataset(conn) -> pd.DataFrame:
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(EXCLUDE_RESULTS))
    matches = conn.execute(
        f"""
        SELECT match_id, date_start, venue_id, team1_id, team2_id,
               toss_winner_id, toss_decision, winner_id, result_type
        FROM matches
        WHERE match_type = 'Test'
          AND date_start IS NOT NULL
          AND result_type IS NOT NULL
          AND result_type NOT IN ({placeholders})
        ORDER BY date_start
        """,
        EXCLUDE_RESULTS,
    ).fetchall()

    rows = []
    for m in matches:
        team1_history = _team_matches(conn, m["team1_id"], m["date_start"])
        team2_history = _team_matches(conn, m["team2_id"], m["date_start"])
        venue_draw_rate = _draw_rate(_venue_matches(conn, m["venue_id"], m["date_start"]))

        perspectives = (
            (m["team1_id"], m["team2_id"], team1_history, team2_history),
            (m["team2_id"], m["team1_id"], team2_history, team1_history),
        )
        for for_team, opp_team, for_history, opp_history in perspectives:
            recent_form = _win_rate(for_history[-RECENT_N:], for_team)
            opp_recent_form = _win_rate(opp_history[-RECENT_N:], opp_team)
            h2h = [r for r in for_history if opp_team in (r["team1_id"], r["team2_id"])]
            h2h_win_rate = _win_rate(h2h, for_team)
            venue_hist = _team_venue_matches(conn, for_team, m["venue_id"], m["date_start"])
            venue_win_rate = _win_rate(venue_hist, for_team)

            rows.append(
                {
                    "match_id": m["match_id"],
                    "date": m["date_start"],
                    "for_team": for_team,
                    "opp_team": opp_team,
                    "recent_form": recent_form,
                    "opp_recent_form": opp_recent_form,
                    "h2h_win_rate": h2h_win_rate,
                    "venue_win_rate": venue_win_rate,
                    "venue_draw_rate": venue_draw_rate,
                    "won_toss": int(m["toss_winner_id"] == for_team),
                    "batted_first": _batted_first(m, for_team),
                    "outcome": _outcome_for(m, for_team),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    conn = get_connection()
    df = build_dataset(conn)
    conn.close()

    out_path = Path(__file__).resolve().parent.parent / "data" / "features.csv"
    df.to_csv(out_path, index=False)

    n_matches = df["match_id"].nunique()
    print(f"Built {len(df)} rows from {n_matches} matches (2 rows per match)")
    print("\nOutcome balance:")
    print(df["outcome"].value_counts(normalize=True).round(3).to_string())
    print(f"\nSaved to {out_path}")