import csv
import os
import glob
import json
import ast
import re
import random
import argparse


def load_handle_to_party(handles_dir="twitter-handles"):
    """Parse all congress handle CSVs into a lowercase_handle -> party dict.

    Iterates 110->117 in order so latest session wins on conflicts.
    Skips 116 (no party column).
    """
    handle_to_party = {}

    for congress_num in range(110, 118):
        if congress_num == 116:
            continue  # no party column in this file

        filepath = os.path.join(
            handles_dir,
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
                    # Format: first_name,last_name,party,state,twitter_handle,position
                    handle = row.get("twitter_handle", "").strip()
                    party = row.get("party", "").strip()

                elif congress_num in (112, 113, 114, 115):
                    # Format: ID,Label,Interval,dw_nominate,inferred_party,party,...,twitter_account,...
                    handle = row.get("twitter_account", "").strip()
                    party = row.get("party", "").strip()

                elif congress_num == 117:
                    # Format: Name,Link,State,Party,Position
                    # Link is full URL like https://twitter.com/SenatorBaldwin
                    link = row.get("Link", "").strip()
                    if link:
                        handle = link.rstrip("/").split("/")[-1]
                    party = row.get("Party", "").strip()

                if handle and party:
                    # Normalize non-standard party codes (e.g., "ID" -> "I")
                    party = {"ID": "I"}.get(party, party)
                    handle_to_party[handle.lower()] = party

    return handle_to_party


def extract_party_from_filename(filename):
    """Extract party from filename like Adams_Alma_D_NC.csv -> 'D'."""
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    # Format: Last_First_Party_State
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


def scan_tweets(tweets_dir, handle_to_party):
    """Scan tweet CSVs for tweets mentioning exactly one congress member.

    Returns list of dicts with keys:
        tweet_id, tweeter_handle, tweeter_party, mentioned_handle,
        mentioned_party, tweet_text
    """
    eligible = []
    csv_files = sorted(glob.glob(os.path.join(tweets_dir, "*.csv")))

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

                # Determine tweeter party: handle lookup first, filename fallback
                tweeter_party = handle_to_party.get(username.lower())
                if not tweeter_party:
                    tweeter_party = filename_party
                if not tweeter_party:
                    continue

                # Parse mentions and find congress members
                mentions = parse_mentions(row.get("mentions", ""))
                congress_mentions = []
                for m in mentions:
                    screen_name = m.get("screen_name", "").strip()
                    if screen_name and screen_name.lower() in handle_to_party:
                        congress_mentions.append(screen_name)

                # Keep only tweets with exactly one congress-member mention
                if len(congress_mentions) != 1:
                    continue

                mentioned_handle = congress_mentions[0]
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict IGR and emotion for congressional tweets using Claude API"
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of tweets to sample (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv",
        help="Output CSV path (default: predictions.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading handle-to-party lookup...")
    lookup = load_handle_to_party()
    print(f"  Loaded {len(lookup)} handles")

    print("Scanning tweets for eligible mentions...")
    eligible = scan_tweets("tweets", lookup)
    print(f"  Found {len(eligible)} eligible tweets")

    # Sample
    random.seed(args.seed)
    sample_size = min(args.sample_size, len(eligible))
    sample = random.sample(eligible, sample_size)
    print(f"  Sampled {sample_size} tweets (seed={args.seed})")
