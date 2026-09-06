"""
Trains a Test-match win/draw/loss classifier on Cricsheet history.

Uses a time-based split (train on older matches, evaluate on the most
recent slice) rather than a random split. This is a forecasting task --
what matters is whether the model generalizes to matches it hasn't seen
the outcome of yet, and a random split wouldn't actually test that.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from db import get_connection
from features import build_dataset

FEATURE_COLS = ["recent_form", "opp_recent_form", "h2h_win_rate", "venue_win_rate", "venue_draw_rate"]
# won_toss/batted_first are deliberately excluded here: they're real signal for a
# match that's already happened, but genuinely unknowable in advance for a future
# fixture (the toss hasn't happened). Training on them and approximating with 0.5
# at inference time created a train/predict mismatch rather than solving anything --
# features.py still computes them for exploration, just not used for training.
TEST_FRACTION = 0.2  # most recent 20% of matches, by date, held out for testing
HALF_LIFE_YEARS = 3  # a training match this many years before the reference date gets half the weight
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "win_draw_loss_model.pkl"


def _recency_weights(dates: pd.Series, reference_date) -> np.ndarray:
    """Exponential decay: weight 1.0 right at `reference_date`, halving every
    HALF_LIFE_YEARS further back. Lets training lean toward the current era
    without discarding older matches outright."""
    days_before = (pd.to_datetime(reference_date) - pd.to_datetime(dates)).dt.days
    years_before = (days_before / 365.25).clip(lower=0)
    return (0.5 ** (years_before / HALF_LIFE_YEARS)).to_numpy()


def time_split(df: pd.DataFrame):
    """Split by MATCH date, not by row -- both perspective-rows of a match
    must land on the same side, or the split leaks across itself."""
    match_dates = df.groupby("match_id")["date"].first().sort_values()
    cutoff_idx = int(len(match_dates) * (1 - TEST_FRACTION))
    cutoff_date = match_dates.iloc[cutoff_idx]

    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]
    return train, test, cutoff_date


def train_and_evaluate(df: pd.DataFrame):
    train, test, cutoff_date = time_split(df)
    print(f"Train: {len(train)} rows (before {cutoff_date}) | Test: {len(test)} rows (from {cutoff_date})")

    X_train, y_train = train[FEATURE_COLS], train["outcome"]
    X_test, y_test = test[FEATURE_COLS], test["outcome"]
    sample_weight = _recency_weights(train["date"], cutoff_date)
    print(f"Recency weights range {sample_weight.min():.3f} (oldest) to {sample_weight.max():.3f} (near cutoff)")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train, clf__sample_weight=sample_weight)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=model.classes_)

    baseline_class = y_train.value_counts().idxmax()
    baseline_acc = (y_test == baseline_class).mean()

    print(f"\nAccuracy: {acc:.3f}   (baseline -- always predict '{baseline_class}': {baseline_acc:.3f})")
    print(f"Log loss: {ll:.3f}")
    print("\n" + classification_report(y_test, pred, zero_division=0))

    print("Calibration check -- mean predicted probability vs actual frequency in the test set.")
    print("(This matters more than the classification report above: the WTC simulator uses")
    print(" these probabilities directly for expected points, it never needs a single predicted label.)")
    proba_df = pd.DataFrame(proba, columns=model.classes_)
    for cls in model.classes_:
        mean_pred = proba_df[cls].mean()
        actual = (y_test == cls).mean()
        print(f"  {cls:>5}:  predicted {mean_pred:.3f}   |   actual {actual:.3f}")

    return model


if __name__ == "__main__":
    conn = get_connection()
    df = build_dataset(conn)
    conn.close()

    model = train_and_evaluate(df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")