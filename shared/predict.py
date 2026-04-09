"""Shared prediction loop, CSV output, and CLI for IGR experiments.

Each provider implements a `predict_fn(client, masked_tweet, condition)` and
passes it to `run_predictions()`. Retry logic and rate limiting are handled here.
"""

import csv
import json
import os
import re
import time
import logging
import argparse

from shared.data import EMOTIONS
from shared.prompts import CONDITIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TOKENS = 512
TEMPERATURE = 0
RATE_LIMIT_PAUSE = 2.0

FIELDNAMES = [
    "tweet_id", "split", "condition", "tweeter_handle", "mentioned_handle",
    "tweet_text", "predicted_igr", "gold_igr",
    "predicted_emotions", "gold_emotions", "model_reasoning",
]

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw_text):
    """Parse a JSON response from any LLM, stripping markdown fences if present.

    Returns dict with keys: igr, emotions (list), reasoning (str).
    """
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    reasoning = result.get("reasoning", "") or result.get("step2_reasoning", "")

    return {
        "igr": result["igr"],
        "emotions": [e for e in EMOTIONS if result.get(e, False)],
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def call_with_retry(fn, *args, max_retries=5, **kwargs):
    """Call fn with exponential backoff on rate limit / server errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as err:
            err_str = str(err).lower()
            is_retryable = any(k in err_str for k in [
                "rate", "429", "overloaded", "529", "500", "timeout",
            ])
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"    Rate limited/server error (attempt {attempt + 1}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Prediction loop
# ---------------------------------------------------------------------------

def run_predictions(gold_rows, tweet_texts, condition, predict_fn, client):
    """Run predictions on gold standard tweets for one condition.

    Args:
        gold_rows: list of gold standard dicts from load_gold_standard()
        tweet_texts: dict of tweet_id -> tweet_text from load_tweet_texts()
        condition: one of CONDITIONS
        predict_fn: callable(client, masked_tweet, condition) -> raw text str
        client: provider-specific API client instance
    """
    from shared.prompts import mask_handle

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
            masked = mask_handle(tweet_text, gold["mentname"])
            raw_text = call_with_retry(predict_fn, client, masked, condition)
            prediction = parse_response(raw_text)

            results.append({
                "tweet_id": gold["tweet_id"],
                "split": gold["split"],
                "condition": condition,
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
            logging.error(f"[{condition}] Tweet {gold['tweet_id']}: {type(e).__name__}: {e}")
            print(f"    ERROR: {e}")

        if i < total - 1:
            time.sleep(RATE_LIMIT_PAUSE)

    print(f"\n  Completed: {len(results)} successful, {errors} errors, {skipped} skipped")
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def output_path_for_condition(condition, base_dir):
    slug = condition.replace("-", "_")
    return os.path.join(base_dir, f"predictions_{slug}.csv")


def write_csv(results, output_path):
    """Write prediction results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(script_dir):
    """Parse command-line arguments. script_dir is the provider's directory."""
    parser = argparse.ArgumentParser(
        description="Predict IGR and emotion for gold standard tweets"
    )
    parser.add_argument(
        "--condition", type=str, default="all",
        choices=CONDITIONS + ["all"],
        help="Prompting condition to run (default: all)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "dev", "test"],
        help="Data split to predict on (default: test)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=script_dir,
        help=f"Directory for output CSVs (default: {os.path.basename(script_dir)}/)",
    )
    return parser.parse_args()
