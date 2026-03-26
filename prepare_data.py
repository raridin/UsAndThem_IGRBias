"""Preprocess congressional tweet data into a clean CSV for IGR labeling.

Scans raw tweet CSVs and congress handle files to produce a single
eligible_tweets.csv containing only tweets that mention exactly one
other congress member. This shared dataset can be used by any LLM
labeling pipeline (Claude, GPT, Gemini, etc.).

Usage:
    python3 prepare_data.py
"""

import csv
import os
import glob
import json
import ast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HANDLES_DIR = os.path.join(SCRIPT_DIR, "twitter-handles")
TWEETS_DIR = os.path.join(SCRIPT_DIR, "tweets")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "eligible_tweets.csv")


def load_handle_to_party():
    """Parse all congress handle CSVs into a lowercase_handle -> party dict.

    Iterates 110->117 in order so latest session wins on conflicts.
    Skips 116 (no party column).
    """
    handle_to_party = {}

    for congress_num in range(110, 118):
        if congress_num == 116:
            continue

        filepath = os.path.join(
            HANDLES_DIR,
            f"Congress Twitter Handles - {congress_num}.csv"
        )
        if not os.path.exists(filepath):
            continue

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                handle = None
                party = None

                if congress_num in (110, 111):
                    handle = row.get("twitter_handle", "").strip()
                    party = row.get("party", "").strip()

                elif congress_num in (112, 113, 114, 115):
                    handle = row.get("twitter_account", "").strip()
                    party = row.get("party", "").strip()

                elif congress_num == 117:
                    link = row.get("Link", "").strip()
                    if link:
                        handle = link.rstrip("/").split("/")[-1]
                    party = row.get("Party", "").strip()

                if handle and party:
                    party = {"ID": "I"}.get(party, party)
                    handle_to_party[handle.lower()] = party

    return handle_to_party


def extract_party_from_filename(filename):
    """Extract party from filename like Adams_Alma_D_NC.csv -> 'D'."""
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    if len(parts) >= 3:
        candidate = parts[-2]
        if candidate in ("D", "R", "I"):
            return candidate
    return None


def parse_mentions(mentions_str):
    """Parse mentions column — Python dict syntax, not JSON."""
    if not mentions_str or mentions_str.strip() == "[]":
        return []
    try:
        return ast.literal_eval(mentions_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(mentions_str)
        except (json.JSONDecodeError, ValueError):
            return []


def scan_tweets(handle_to_party):
    """Scan tweet CSVs for tweets mentioning exactly one congress member."""
    eligible = []
    csv_files = sorted(glob.glob(os.path.join(TWEETS_DIR, "*.csv")))

    for filepath in csv_files:
        filename_party = extract_party_from_filename(filepath)

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                tweet_text = row.get("tweet", "").strip()
                tweet_id = row.get("id", "").strip()
                username = row.get("username", "").strip()

                if not tweet_text or not tweet_id:
                    continue

                tweeter_party = handle_to_party.get(username.lower())
                if not tweeter_party:
                    tweeter_party = filename_party
                if not tweeter_party:
                    continue

                mentions = parse_mentions(row.get("mentions", ""))
                congress_mentions = []
                for m in mentions:
                    screen_name = m.get("screen_name", "").strip()
                    if screen_name and screen_name.lower() in handle_to_party:
                        congress_mentions.append(screen_name)

                if len(congress_mentions) != 1:
                    continue

                mentioned_handle = congress_mentions[0]
                if mentioned_handle.lower() == username.lower():
                    continue

                mentioned_party = handle_to_party[mentioned_handle.lower()]

                eligible.append({
                    "tweet_id": tweet_id,
                    "tweeter_handle": username,
                    "tweeter_party": tweeter_party,
                    "mentioned_handle": mentioned_handle,
                    "mentioned_party": mentioned_party,
                    "tweet_text": tweet_text,
                })

    return eligible


if __name__ == "__main__":
    print("Loading handle-to-party lookup...")
    lookup = load_handle_to_party()
    print(f"  Loaded {len(lookup)} handles")

    print("Scanning tweets for eligible mentions...")
    eligible = scan_tweets(lookup)
    print(f"  Found {len(eligible)} eligible tweets")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fieldnames = [
        "tweet_id", "tweeter_handle", "tweeter_party",
        "mentioned_handle", "mentioned_party", "tweet_text",
    ]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eligible)

    print(f"\nSaved to {OUTPUT_FILE}")
