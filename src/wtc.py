"""
Computes the current WTC 2025-27 points table from ingested Test match data.

Ranked by PCT (percentage of points won), not raw points, since teams play
a different number of matches across the cycle -- same rule the ICC uses.

Assumption: every Test between two full members during the cycle window
counts toward WTC standings (true for the current cycle's format). A
match with result_type='no result' is counted as played but earns 0 points
for both sides -- a reasonable default, flagged here since I haven't found
an authoritative source confirming ICC's exact rule for that edge case.
"""

import sqlite3

from db import get_connection

WTC_TEAMS = {
    "Australia", "Bangladesh", "England", "India", "New Zealand",
    "Pakistan", "South Africa", "Sri Lanka", "West Indies",
}
CYCLE_START = "2025-06-17"  # confirmed start of the 2025-27 cycle; the previous
# cycle's final (South Africa beat Australia, ~June 11-15 2025) falls just
# before this and must NOT be counted here
CYCLE_END = "2027-06-30"

POINTS = {"win": 12, "tie": 6, "draw": 4}  # loss / no result = 0


def compute_standings(conn) -> list[dict]:
    conn.row_factory = sqlite3.Row
    matches = conn.execute(
        """
        SELECT m.result_type, m.winner_id, m.team1_id, m.team2_id,
               t1.name AS team1, t2.name AS team2
        FROM matches m
        JOIN teams t1 ON m.team1_id = t1.team_id
        JOIN teams t2 ON m.team2_id = t2.team_id
        WHERE m.match_type = 'Test'
          AND m.date_start >= ? AND m.date_start <= ?
        """,
        (CYCLE_START, CYCLE_END),
    ).fetchall()

    stats = {}
    for m in matches:
        if m["team1"] not in WTC_TEAMS or m["team2"] not in WTC_TEAMS:
            continue  # only bilateral Tests between the 9 full members count

        for team_name, team_id in ((m["team1"], m["team1_id"]), (m["team2"], m["team2_id"])):
            s = stats.setdefault(team_name, {"matches": 0, "points": 0})
            s["matches"] += 1

            if m["result_type"] == "draw":
                s["points"] += POINTS["draw"]
            elif m["result_type"] == "tie":
                s["points"] += POINTS["tie"]
            elif m["winner_id"] == team_id:
                s["points"] += POINTS["win"]
            # loss / no result: +0, nothing to add

    rows = []
    for team, s in stats.items():
        available = s["matches"] * POINTS["win"]
        pct = (s["points"] / available * 100) if available else 0.0
        rows.append({"team": team, "matches": s["matches"], "points": s["points"], "pct": round(pct, 2)})

    rows.sort(key=lambda r: r["pct"], reverse=True)
    return rows


if __name__ == "__main__":
    conn = get_connection()
    standings = compute_standings(conn)
    conn.close()

    print(f"WTC 2025-27 standings (as of matches ingested so far)\n")
    print(f"{'#':<3}{'Team':<15}{'M':>4}{'Pts':>6}{'PCT':>8}")
    for i, r in enumerate(standings, 1):
        print(f"{i:<3}{r['team']:<15}{r['matches']:>4}{r['points']:>6}{r['pct']:>7.2f}%")