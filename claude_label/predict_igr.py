"""Predict IGR and emotion for congressional tweets using Claude API.

Joins gold standard labels (data/data.tsv) with tweet text from
eligible_tweets.csv, runs Claude predictions, and compares against
ground truth for both IGR and Plutchik emotions.

Usage:
    python3 claude_label/predict_igr.py [--split dev] [--output predictions.csv]
"""

import csv
import os
import json
import re
import argparse
import time
import logging

from dotenv import load_dotenv
from anthropic import Anthropic

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ELIGIBLE_FILE = os.path.join(PROJECT_ROOT, "data", "eligible_tweets.csv")
GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "data.tsv")

EMOTIONS = ["Admiration", "Anger", "Disgust", "Fear", "Interest", "Joy", "Sadness", "Surprise"]

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(
    filename=os.path.join(SCRIPT_DIR, "prediction_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(message)s",
)


def load_gold_standard(split=None):
    """Load gold standard labels from data.tsv.

    Returns list of dicts with: tweet_id, username, mentname, group (1/-1),
    party, and gold_emotions (list of emotion strings that are True).
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict IGR and emotion for gold standard tweets using Claude API"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["train", "dev", "test"],
        help="Only predict on this split (default: all splits)"
    )
    parser.add_argument(
        "--output", type=str, default=os.path.join(SCRIPT_DIR, "predictions.csv"),
        help="Output CSV path (default: claude_label/predictions.csv)"
    )
    return parser.parse_args()


def mask_handle(tweet_text, handle):
    """Replace all occurrences of the mentioned handle with @Doe (case-insensitive)."""
    masked = re.sub(re.escape(f"@{handle}"), "@Doe", tweet_text, flags=re.IGNORECASE)
    masked = re.sub(r'(?<!\w)' + re.escape(handle) + r'(?!\w)', "Doe", masked, flags=re.IGNORECASE)
    return masked


def predict_tweet(client, tweet_text, mentioned_handle):
    """Call Claude API to predict IGR and emotions for a tweet."""
    masked = mask_handle(tweet_text, mentioned_handle)

    prompt = f"""You are annotating tweets for a political communication research study.

Given this tweet, determine:
1. Interpersonal Group Relationship (IGR): Does the speaker appear to be talking about someone in their own group (In-Group) or a different group (Out-Group)? Base this only on linguistic cues, tone, and context.
2. Emotions toward @Doe: For each of the 8 Plutchik emotions below, indicate True or False. Multiple emotions can be True. If no emotion applies, mark all as False.

Emotions: Admiration, Anger, Disgust, Fear, Interest, Joy, Sadness, Surprise

Tweet:
"{masked}"

Respond with ONLY this JSON, no other text:
{{"igr": "In-Group" or "Out-Group", "Admiration": true/false, "Anger": true/false, "Disgust": true/false, "Fear": true/false, "Interest": true/false, "Joy": true/false, "Sadness": true/false, "Surprise": true/false, "reasoning": "<1-2 sentences>"}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    return {
        "igr": result["igr"],
        "emotions": [e for e in EMOTIONS if result.get(e, False)],
        "reasoning": result.get("reasoning", ""),
    }


def run_predictions(gold_rows, tweet_texts):
    """Run Claude API predictions on gold standard tweets."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or set as env var.")
        return []

    client = Anthropic(api_key=api_key)
    results = []
    errors = 0
    skipped = 0
    total = len(gold_rows)

    for i, gold in enumerate(gold_rows):
        tweet_text = tweet_texts.get(gold["tweet_id"])
        if not tweet_text:
            skipped += 1
            print(f"  [{i + 1}/{total}] SKIP {gold['tweet_id']} — not in eligible_tweets.csv")
            continue

        print(f"  [{i + 1}/{total}] Processing tweet {gold['tweet_id']}...")

        try:
            prediction = predict_tweet(client, tweet_text, gold["mentname"])

            results.append({
                "tweet_id": gold["tweet_id"],
                "split": gold["split"],
                "tweeter_handle": gold["username"],
                "mentioned_handle": gold["mentname"],
                "tweet_text": tweet_text,
                "predicted_igr": prediction["igr"],
                "gold_igr": gold["gold_igr"],
                "predicted_emotions": "|".join(prediction["emotions"]) if prediction["emotions"] else "None",
                "gold_emotions": "|".join(gold["gold_emotions"]) if gold["gold_emotions"] else "None",
                "model_reasoning": prediction["reasoning"],
            })

        except Exception as e:
            errors += 1
            logging.error(f"Tweet {gold['tweet_id']}: {type(e).__name__}: {e}")
            print(f"    ERROR: {e}")

        if i < total - 1:
            time.sleep(1.0)

    print(f"\n  Completed: {len(results)} successful, {errors} errors, {skipped} skipped")
    return results


def write_csv(results, output_path):
    """Write prediction results to CSV."""
    fieldnames = [
        "tweet_id", "split", "tweeter_handle", "mentioned_handle", "tweet_text",
        "predicted_igr", "gold_igr",
        "predicted_emotions", "gold_emotions",
        "model_reasoning",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    """Print accuracy summary comparing predictions to gold standard."""
    if not results:
        return

    total = len(results)

    # --- IGR accuracy ---
    igr_correct = sum(1 for r in results if r["predicted_igr"] == r["gold_igr"])
    igr_accuracy = igr_correct / total * 100

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total predictions: {total}")
    print(f"\nIGR accuracy: {igr_correct}/{total} ({igr_accuracy:.1f}%)")

    # IGR confusion
    tp = sum(1 for r in results if r["predicted_igr"] == "In-Group" and r["gold_igr"] == "In-Group")
    fp = sum(1 for r in results if r["predicted_igr"] == "In-Group" and r["gold_igr"] == "Out-Group")
    fn = sum(1 for r in results if r["predicted_igr"] == "Out-Group" and r["gold_igr"] == "In-Group")
    tn = sum(1 for r in results if r["predicted_igr"] == "Out-Group" and r["gold_igr"] == "Out-Group")
    print(f"  TP (pred=In, gold=In):   {tp}")
    print(f"  FP (pred=In, gold=Out):  {fp}")
    print(f"  FN (pred=Out, gold=In):  {fn}")
    print(f"  TN (pred=Out, gold=Out): {tn}")

    # --- Emotion accuracy (exact match and per-emotion) ---
    exact_match = 0
    per_emotion_correct = {e: 0 for e in EMOTIONS}
    per_emotion_total = {e: 0 for e in EMOTIONS}

    for r in results:
        pred_set = set(r["predicted_emotions"].split("|")) if r["predicted_emotions"] != "None" else set()
        gold_set = set(r["gold_emotions"].split("|")) if r["gold_emotions"] != "None" else set()

        if pred_set == gold_set:
            exact_match += 1

        for e in EMOTIONS:
            pred_has = e in pred_set
            gold_has = e in gold_set
            if pred_has == gold_has:
                per_emotion_correct[e] += 1

    print(f"\nEmotion exact match: {exact_match}/{total} ({exact_match / total * 100:.1f}%)")
    print("\nPer-emotion accuracy:")
    for e in EMOTIONS:
        acc = per_emotion_correct[e] / total * 100
        print(f"  {e:12s}: {per_emotion_correct[e]}/{total} ({acc:.1f}%)")

    # Predicted emotion distribution
    pred_counts = {e: 0 for e in EMOTIONS}
    gold_counts = {e: 0 for e in EMOTIONS}
    for r in results:
        pred_set = set(r["predicted_emotions"].split("|")) if r["predicted_emotions"] != "None" else set()
        gold_set = set(r["gold_emotions"].split("|")) if r["gold_emotions"] != "None" else set()
        for e in EMOTIONS:
            if e in pred_set:
                pred_counts[e] += 1
            if e in gold_set:
                gold_counts[e] += 1

    print("\nEmotion distribution (predicted vs gold):")
    for e in EMOTIONS:
        print(f"  {e:12s}: pred={pred_counts[e]:4d}  gold={gold_counts[e]:4d}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = parse_args()

    print("Loading gold standard labels...")
    gold_rows = load_gold_standard(split=args.split)
    split_label = args.split or "all"
    print(f"  Loaded {len(gold_rows)} gold standard tweets (split={split_label})")

    print("Loading tweet texts...")
    tweet_texts = load_tweet_texts()
    print(f"  Loaded {len(tweet_texts)} tweet texts")

    print(f"\nRunning predictions on {len(gold_rows)} tweets...")
    results = run_predictions(gold_rows, tweet_texts)

    if results:
        write_csv(results, args.output)
        print(f"\nPredictions saved to {args.output}")
        print_summary(results)
    else:
        print("\nNo predictions generated. Check prediction_errors.log for details.")
