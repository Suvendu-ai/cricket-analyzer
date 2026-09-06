"""
Projects the WTC 2025-27 table forward using the trained win/draw/loss
model, given a list of remaining fixtures.

Two features the model was trained on -- won_toss, batted_first -- are
genuinely unknowable in advance for a fixture that hasn't been played yet
(the toss hasn't happened). Both default to 0.5, i.e. "unknown, treat as
50-50", rather than guessed. This is a real simplification: it means the
model can't express "team X does especially well when they win the toss",
only the average case.
"""

import sqlite3
from datetime import date

import pandas as pd

from db import get_connection
from features import (
    NEUTRAL_PRIOR,
    RECENT_N,
    _draw_rate,
    _team_matches,
    _team_venue_matches,
    _venue_matches,
    _win_rate,
)
from wtc import POINTS, compute_standings


def _team_id(conn, name):
    row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def _venue_id(conn, name):
    if not name:
        return None
    row = conn.execute("SELECT venue_id FROM venues WHERE name = ?", (name,)).fetchone()
    return row[0] if row else None


def predict_fixture(model, conn, team1_name: str, team2_name: str, venue_name: str = None, as_of: str = None) -> dict:
    """Win/draw/loss probabilities for team1, for a not-yet-played fixture."""
    conn.row_factory = sqlite3.Row
    as_of = as_of or date.today().isoformat()

    team1_id, team2_id = _team_id(conn, team1_name), _team_id(conn, team2_name)
    if team1_id is None or team2_id is None:
        raise ValueError(f"Unknown team name(s): {team1_name!r}, {team2_name!r}")

    history1 = _team_matches(conn, team1_id, as_of)
    history2 = _team_matches(conn, team2_id, as_of)

    recent_form = _win_rate(history1[-RECENT_N:], team1_id)
    opp_recent_form = _win_rate(history2[-RECENT_N:], team2_id)
    h2h = [r for r in history1 if team2_id in (r["team1_id"], r["team2_id"])]
    h2h_win_rate = _win_rate(h2h, team1_id)

    venue_id = _venue_id(conn, venue_name)
    if venue_id:
        venue_win_rate = _win_rate(_team_venue_matches(conn, team1_id, venue_id, as_of), team1_id)
        venue_draw_rate = _draw_rate(_venue_matches(conn, venue_id, as_of))
    else:
        venue_win_rate = NEUTRAL_PRIOR
        venue_draw_rate = NEUTRAL_PRIOR

    row = pd.DataFrame([{
        "recent_form": recent_form,
        "opp_recent_form": opp_recent_form,
        "h2h_win_rate": h2h_win_rate,
        "venue_win_rate": venue_win_rate,
        "venue_draw_rate": venue_draw_rate,
        "won_toss": 0.5,
        "batted_first": 0.5,
    }])
    proba = model.predict_proba(row)[0]
    return dict(zip(model.classes_, proba))


def project_standings(model, conn, remaining_fixtures: list[dict], as_of: str = None) -> list[dict]:
    """
    remaining_fixtures: [{"team1": ..., "team2": ..., "venue": ... (optional)}, ...]

    Combines current earned points with EXPECTED points from remaining
    fixtures (win=12, draw=4 to both sides). This is a partial-cycle
    projection, not the full 2025-27 table -- it only covers whatever
    fixtures are passed in.
    """
    projected = {r["team"]: dict(r) for r in compute_standings(conn)}

    for fx in remaining_fixtures:
        t1, t2 = fx["team1"], fx["team2"]
        proba = predict_fixture(model, conn, t1, t2, venue_name=fx.get("venue"), as_of=as_of)

        exp_points_t1 = proba.get("win", 0) * POINTS["win"] + proba.get("draw", 0) * POINTS["draw"]
        exp_points_t2 = proba.get("loss", 0) * POINTS["win"] + proba.get("draw", 0) * POINTS["draw"]

        for team, exp_pts in ((t1, exp_points_t1), (t2, exp_points_t2)):
            projected.setdefault(team, {"team": team, "matches": 0, "points": 0.0})
            projected[team]["matches"] += 1
            projected[team]["points"] += exp_pts

    rows = []
    for team, s in projected.items():
        available = s["matches"] * POINTS["win"]
        pct = (s["points"] / available * 100) if available else 0.0
        rows.append({"team": team, "matches": s["matches"], "points": round(s["points"], 1), "pct": round(pct, 2)})

    rows.sort(key=lambda r: r["pct"], reverse=True)
    return rows


# A snapshot of real, confirmed near-term fixtures (through Dec 2026) as of
# this writing -- NOT the complete remaining 2025-27 schedule. Extending
# this to the full ~58 remaining matches through June 2027 is the natural
# next step, from an authoritative source rather than search snippets.
KNOWN_UPCOMING_FIXTURES = [
    {"team1": "England", "team2": "Pakistan", "venue": "Edgbaston"},
    {"team1": "South Africa", "team2": "Australia", "venue": "Kingsmead"},
    {"team1": "South Africa", "team2": "Australia", "venue": None},  # Gqeberha, venue name may differ in Cricsheet
    {"team1": "South Africa", "team2": "Australia", "venue": None},  # Cape Town
    {"team1": "South Africa", "team2": "Bangladesh", "venue": None},  # Johannesburg
    {"team1": "South Africa", "team2": "Bangladesh", "venue": None},  # Centurion
    {"team1": "New Zealand", "team2": "India", "venue": None},  # Wellington
    {"team1": "New Zealand", "team2": "India", "venue": None},  # Christchurch
    {"team1": "Australia", "team2": "New Zealand", "venue": None},  # Perth
]


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    model_path = Path(__file__).resolve().parent.parent / "data" / "win_draw_loss_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    conn = get_connection()
    print("Projected standings including known near-term fixtures (partial cycle):\n")
    projected = project_standings(model, conn, KNOWN_UPCOMING_FIXTURES)
    conn.close()

    print(f"{'#':<3}{'Team':<15}{'M':>4}{'Pts':>7}{'PCT':>8}")
    for i, r in enumerate(projected, 1):
        print(f"{i:<3}{r['team']:<15}{r['matches']:>4}{r['points']:>7}{r['pct']:>7.2f}%")