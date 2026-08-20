# Cricket Analyzer — Data Layer (Phase 1)

The foundation of a cricket intelligence platform: a normalized SQLite
database of match, innings, and ball-by-ball data, built from
[Cricsheet](https://cricsheet.org)'s free, open archive.

This is Phase 1 of a larger project — ML-driven predictions across Test/WTC,
ODI/2027 World Cup, and T20I. This phase just gets clean, queryable
historical data in place. No live API needed yet: the WTC points table,
qualification odds, and a first win-probability model can all be built
straight from match results.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download Cricsheet's match archive (test / odi / t20i)
python src/download_data.py test

# 2. Parse it into the database
python src/ingest.py test
```

Both scripts accept multiple formats at once, e.g. `python src/ingest.py test odi t20i`.
Re-running `ingest.py` is safe — already-loaded matches are skipped, not duplicated.

This creates `data/cricket.db`, a SQLite database you can open directly in
VS Code's SQLite viewer or query with the `sqlite3` CLI.

## Schema

| Table | Contents |
|---|---|
| `matches` | one row per match — teams, venue, toss, result, dates |
| `innings` | one row per innings per match |
| `deliveries` | one row per ball — batter, bowler, runs, wicket details |
| `teams`, `venues` | lookup tables |

## Verified against

`src/ingest.py` was tested against a hand-built sample matching Cricsheet's
documented JSON schema (format version 1.1.0) — this sandbox can't reach
`cricsheet.org` directly, so run `download_data.py` yourself to pull the
real archive; it's a plain HTTPS download, nothing else needed.

## Known simplifications (Phase 2 candidates)

- Batter/bowler are stored as raw names, not resolved against Cricsheet's
  player registry — name variants across seasons aren't merged yet.
- "Won by an innings" results store the runs/wickets margin but not the
  innings count itself.

## Data source

[Cricsheet](https://cricsheet.org) is released under the Open Database
License (ODbL) — free to use, with attribution.