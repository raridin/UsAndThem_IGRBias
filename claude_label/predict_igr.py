"""Predict IGR and Plutchik emotions for congressional tweets using Claude API.

Runs 3 prompting conditions (zero-shot, few-shot, chain-of-thought) against
gold standard labels from data/data.tsv, comparing Claude predictions to
human annotations for both IGR and emotion classification.

Usage:
    python3 claude_label/predict_igr.py --condition all --split test
    python3 claude_label/predict_igr.py --condition zero-shot --split dev
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

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ELIGIBLE_FILE = os.path.join(PROJECT_ROOT, "data", "eligible_tweets.csv")
GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "data.tsv")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(
    filename=os.path.join(SCRIPT_DIR, "prediction_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITIONS = ["zero-shot", "few-shot", "cot-thinking"]
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 512
TEMPERATURE = 0
RATE_LIMIT_PAUSE = 2.0
EMOTIONS = [
    "Admiration", "Anger", "Disgust", "Fear",
    "Interest", "Joy", "Sadness", "Surprise",
]

# ---------------------------------------------------------------------------
# Few-shot examples (from train split, pre-masked)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "label": "positive in-group",
        "tweet": "Great seeing so many people from #CT in our DC office this morning for a constituent coffee w me and @Doe:  http://t.co/Ud9HeV2S3B",
        "answer": {
            "igr": "In-Group",
            "Admiration": False, "Anger": False, "Disgust": False, "Fear": False,
            "Interest": False, "Joy": True, "Sadness": False, "Surprise": False,
            "reasoning": "The speaker is sharing a positive, collaborative experience with @Doe, suggesting they are allies in the same group.",
        },
    },
    {
        "label": "positive out-group",
        "tweet": "Thanks to @Doe for kind words. Enjoy serving w you on IP subcommittee.",
        "answer": {
            "igr": "Out-Group",
            "Admiration": False, "Anger": False, "Disgust": False, "Fear": False,
            "Interest": False, "Joy": True, "Sadness": False, "Surprise": False,
            "reasoning": "The speaker thanks @Doe politely but the phrasing suggests a collegial cross-group relationship rather than close in-group solidarity.",
        },
    },
    {
        "label": "negative in-group",
        "tweet": "Lifting my friend and colleague @Doe and his wife up in prayer as they heal from this accident. Get well soon, friends.  https://t.co/MWVX5THsTg",
        "answer": {
            "igr": "In-Group",
            "Admiration": False, "Anger": False, "Disgust": False, "Fear": False,
            "Interest": False, "Joy": False, "Sadness": True, "Surprise": False,
            "reasoning": "The speaker expresses deep personal concern for @Doe, calling them 'my friend and colleague', indicating a strong in-group bond.",
        },
    },
    {
        "label": "negative out-group",
        "tweet": ".@Doe is voting against confirmation of Lynch, a North Carolinian and a proven litigator. Very disappointing.#ConfirmLorettaLynch",
        "answer": {
            "igr": "Out-Group",
            "Admiration": False, "Anger": True, "Disgust": True, "Fear": False,
            "Interest": False, "Joy": False, "Sadness": False, "Surprise": False,
            "reasoning": "The speaker criticizes @Doe's vote as 'very disappointing', indicating opposition and out-group dynamics.",
        },
    },
    {
        "label": "neutral in-group (no emotion)",
        "tweet": "Tomorrow I'm hosting a Spring Break reception with @Doe for Iowans visiting DC. More details here:  https://t.co/9MAIFCxCJd",
        "answer": {
            "igr": "In-Group",
            "Admiration": False, "Anger": False, "Disgust": False, "Fear": False,
            "Interest": False, "Joy": False, "Sadness": False, "Surprise": False,
            "reasoning": "The speaker mentions @Doe in a purely informational announcement with no emotional language, but the co-hosting implies an in-group relationship.",
        },
    },
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gold_standard(split=None):
    """Load gold standard labels from data.tsv."""
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


# ---------------------------------------------------------------------------
# Handle masking
# ---------------------------------------------------------------------------

def mask_handle(tweet_text, handle):
    """Replace all occurrences of the mentioned handle with @Doe."""
    masked = re.sub(re.escape(f"@{handle}"), "@Doe", tweet_text, flags=re.IGNORECASE)
    masked = re.sub(
        r"(?<!\w)" + re.escape(handle) + r"(?!\w)", "Doe", masked, flags=re.IGNORECASE
    )
    return masked


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_TASK_FRAMING = """You are annotating tweets for a political communication research study.

Given a tweet, determine:
1. Interpersonal Group Relationship (IGR): Does the speaker appear to be talking about someone in their own group (In-Group) or a different group (Out-Group)? Base this only on linguistic cues, tone, and context.
2. Emotions toward @Doe: For each of the 8 Plutchik emotions below, indicate true or false. Multiple emotions can be true. if no clear emotion is expressed toward @Doe, mark ALL emotions as false.

Emotions: Admiration, Anger, Disgust, Fear, Interest, Joy, Sadness, Surprise"""

_JSON_SCHEMA = '{"igr": "In-Group" or "Out-Group", "Admiration": true/false, "Anger": true/false, "Disgust": true/false, "Fear": true/false, "Interest": true/false, "Joy": true/false, "Sadness": true/false, "Surprise": true/false, "reasoning": "<1-2 sentences>"}'


def build_zero_shot_prompt(masked_tweet):
    return f"""{_TASK_FRAMING}

Tweet:
"{masked_tweet}"

Respond with ONLY this JSON, no other text:
{_JSON_SCHEMA}"""


def build_few_shot_prompt(masked_tweet):
    examples_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        answer_json = json.dumps(ex["answer"], ensure_ascii=False)
        examples_block += f"\nExample {i} ({ex['label']}):\nTweet: \"{ex['tweet']}\"\nAnswer: {answer_json}\n"

    return f"""{_TASK_FRAMING}

Here are some examples:
{examples_block}
Now classify this tweet:
"{masked_tweet}"

Respond with ONLY this JSON, no other text:
{_JSON_SCHEMA}"""


def build_cot_thinking_prompt(masked_tweet):
    """Same task as zero-shot, but uses the API's native extended thinking."""
    return f"""{_TASK_FRAMING}

Tweet:
"{masked_tweet}"

Respond with ONLY this JSON, no other text:
{_JSON_SCHEMA}"""


PROMPT_BUILDERS = {
    "zero-shot": build_zero_shot_prompt,
    "few-shot": build_few_shot_prompt,
    "cot-thinking": build_cot_thinking_prompt,
}

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_tweet(client, masked_tweet, condition, max_retries=5):
    """Call Claude API to predict IGR and emotions for a masked tweet.

    Retries with exponential backoff on rate limit (429) or server errors (5xx).
    """
    prompt = PROMPT_BUILDERS[condition](masked_tweet)

    use_thinking = condition == "cot-thinking"

    for attempt in range(max_retries):
        try:
            if use_thinking:
                message = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS + 1024,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 1024,
                    },
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                message = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
            break
        except Exception as api_err:
            err_str = str(api_err)
            is_retryable = "rate" in err_str.lower() or "429" in err_str or "overloaded" in err_str.lower() or "529" in err_str or "500" in err_str
            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"    Rate limited/server error (attempt {attempt + 1}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    # Extended thinking responses have thinking blocks before the text block
    raw = None
    for block in message.content:
        if block.type == "text":
            raw = block.text.strip()
            break
    if raw is None:
        raise ValueError("No text block in API response")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    # Normalise reasoning field across conditions
    reasoning = result.get("reasoning", "") or result.get("step2_reasoning", "")

    return {
        "igr": result["igr"],
        "emotions": [e for e in EMOTIONS if result.get(e, False)],
        "reasoning": reasoning,
    }


def run_predictions(gold_rows, tweet_texts, condition):
    """Run Claude predictions on gold standard tweets for one condition."""
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
            masked = mask_handle(tweet_text, gold["mentname"])
            prediction = predict_tweet(client, masked, condition)

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
# Output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "tweet_id", "split", "condition", "tweeter_handle", "mentioned_handle",
    "tweet_text", "predicted_igr", "gold_igr",
    "predicted_emotions", "gold_emotions", "model_reasoning",
]


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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict IGR and emotion for gold standard tweets using Claude API"
    )
    parser.add_argument(
        "--condition", type=str, default="all",
        choices=["zero-shot", "few-shot", "cot-thinking", "all"],
        help="Prompting condition to run (default: all)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "dev", "test"],
        help="Data split to predict on (default: test)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=SCRIPT_DIR,
        help="Directory for output CSVs (default: claude_label/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("Loading gold standard labels...")
    gold_rows = load_gold_standard(split=args.split)
    print(f"  Loaded {len(gold_rows)} gold standard tweets (split={args.split})")

    print("Loading tweet texts...")
    tweet_texts = load_tweet_texts()
    print(f"  Loaded {len(tweet_texts)} tweet texts")

    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    for condition in conditions:
        print(f"\n{'#' * 60}")
        print(f"# CONDITION: {condition}")
        print(f"{'#' * 60}")

        results = run_predictions(gold_rows, tweet_texts, condition)

        if results:
            out_path = output_path_for_condition(condition, args.output_dir)
            write_csv(results, out_path)
            print(f"\nSaved {len(results)} predictions to {out_path}")
        else:
            print("\nNo predictions generated. Check prediction_errors.log.")
