"""Shared data loading for IGR prediction experiments."""

import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELIGIBLE_FILE = os.path.join(PROJECT_ROOT, "data", "eligible_tweets.csv")
GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "data.tsv")

EMOTIONS = [
    "Admiration", "Anger", "Disgust", "Fear",
    "Interest", "Joy", "Sadness", "Surprise",
]


def load_gold_standard(split=None):
    """Load gold standard labels from data.tsv.

    Returns list of dicts with: tweet_id, username, mentname, gold_igr,
    gold_emotions, party, split.
    """
    rows = []
    with open(GOLD_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if split and row["Split"] != split:
                continue
            gold_emotions = [e for e in EMOTIONS if row[e] == "True"]
            rows.append({
                "tweet_id": row["TweetId"],
                "username": row["username"],
                "mentname": row["mentname"],
                "gold_igr": "In-Group" if row["group"] == "1" else "Out-Group",
                "gold_emotions": gold_emotions,
                "party": row["party"],
                "split": row["Split"],
            })
    return rows


def load_tweet_texts():
    """Build tweet_id -> tweet_text lookup from eligible_tweets.csv."""
    lookup = {}
    with open(ELIGIBLE_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["tweet_id"]] = row["tweet_text"]
    return lookup
