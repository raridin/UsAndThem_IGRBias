"""Predict IGR and Plutchik emotions for congressional tweets using DeepSeek.

Usage:
    python3 deepseek_label/predict_igr.py --condition all --split test
    python3 deepseek_label/predict_igr.py --condition zero-shot --split dev
    python3 deepseek_label/predict_igr.py --condition zero-shot --split test --workers 8 --pause-seconds 0

Setup:
    pip install openai python-dotenv
    Add DEEPSEEK_API_KEY=... to .env

Note: DeepSeek uses an OpenAI-compatible API, so we use the openai SDK
      with a custom base_url.

Parallel workers / --limit / --pause-seconds live in this file only so
shared/predict.py stays unchanged for other providers (e.g. Claude).
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data import load_gold_standard, load_tweet_texts
from shared.prompts import PROMPT_BUILDERS, CONDITIONS
from shared.predict import (
    call_with_retry,
    parse_response,
    write_csv,
    output_path_for_condition,
    MAX_TOKENS,
    TEMPERATURE,
    RATE_LIMIT_PAUSE,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(
    filename=os.path.join(SCRIPT_DIR, "prediction_errors.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(message)s",
)

# ---------------------------------------------------------------------------
# DeepSeek-specific model config
# ---------------------------------------------------------------------------
MODEL = "deepseek-chat"  # TODO: confirm model name (deepseek-chat or deepseek-reasoner)
BASE_URL = "https://api.deepseek.com"
# Reasoner spends tokens on internal chain-of-thought before emitting JSON; 512 is too small.
REASONER_MAX_TOKENS = 8192


def parse_args(script_dir):
    """CLI for DeepSeek only (includes workers/limit/pause)."""
    parser = argparse.ArgumentParser(
        description="Predict IGR and emotion for gold standard tweets (DeepSeek)"
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
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional max number of tweets to process (default: all in split)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--pause-seconds", type=float, default=RATE_LIMIT_PAUSE,
        help=f"Delay between requests when workers=1 (default: {RATE_LIMIT_PAUSE})",
    )
    return parser.parse_args()


def run_predictions_deepseek(
    gold_rows,
    tweet_texts,
    condition,
    predict_fn,
    client,
    limit=None,
    workers=1,
    pause_seconds=RATE_LIMIT_PAUSE,
):
    """Same as shared run_predictions, plus optional limit and parallel workers."""
    from shared.prompts import mask_handle

    results = []
    errors = 0
    skipped = 0
    if limit is not None:
        gold_rows = gold_rows[:limit]
    total = len(gold_rows)

    def build_result_row(i, gold):
        tweet_text = tweet_texts.get(gold["tweet_id"])
        if not tweet_text:
            return ("skip", i, gold, None, None)

        masked = mask_handle(tweet_text, gold["mentname"])
        raw_text = call_with_retry(predict_fn, client, masked, condition)
        prediction = parse_response(raw_text)

        row = {
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
        }
        return ("ok", i, gold, row, None)

    if workers <= 1:
        for i, gold in enumerate(gold_rows):
            print(f"  [{i + 1}/{total}] Processing tweet {gold['tweet_id']}...")
            try:
                status, _, _, row, _ = build_result_row(i, gold)
                if status == "skip":
                    skipped += 1
                    print(f"  [{i + 1}/{total}] SKIP {gold['tweet_id']} — not in eligible_tweets.csv")
                else:
                    results.append(row)
            except Exception as e:
                errors += 1
                logging.error(f"[{condition}] Tweet {gold['tweet_id']}: {type(e).__name__}: {e}")
                print(f"    ERROR: {e}")

            if i < total - 1 and pause_seconds > 0:
                time.sleep(pause_seconds)
    else:
        print(f"  Running with {workers} parallel workers...")
        indexed_results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(build_result_row, i, gold): (i, gold)
                for i, gold in enumerate(gold_rows)
            }
            done = 0
            for fut in as_completed(future_map):
                i, gold = future_map[fut]
                done += 1
                print(f"  [{done}/{total}] Finished tweet {gold['tweet_id']}...")
                try:
                    status, idx, _, row, _ = fut.result()
                    if status == "skip":
                        skipped += 1
                    else:
                        indexed_results.append((idx, row))
                except Exception as e:
                    errors += 1
                    logging.error(f"[{condition}] Tweet {gold['tweet_id']}: {type(e).__name__}: {e}")
                    print(f"    ERROR: {e}")
        indexed_results.sort(key=lambda x: x[0])
        results = [row for _, row in indexed_results]

    print(f"\n  Completed: {len(results)} successful, {errors} errors, {skipped} skipped")
    return results


def predict_tweet(client, masked_tweet, condition):
    """Call DeepSeek API and return the raw response text.

    DeepSeek uses an OpenAI-compatible API (openai SDK with custom base_url).
    For cot-thinking, switch to deepseek-reasoner.
    """
    prompt = PROMPT_BUILDERS[condition](masked_tweet)
    model = "deepseek-reasoner" if condition == "cot-thinking" else MODEL

    # deepseek-reasoner ignores temperature, so only pass it for chat.
    max_out = REASONER_MAX_TOKENS if model == "deepseek-reasoner" else MAX_TOKENS
    request_kwargs = {
        "model": model,
        "max_tokens": max_out,
        "messages": [
            {"role": "system", "content": "You are a research annotation assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    if model != "deepseek-reasoner":
        request_kwargs["temperature"] = TEMPERATURE

    response = client.chat.completions.create(**request_kwargs)
    msg = response.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        raise ValueError(
            "Empty model content (reasoner may need more max_tokens or returned only reasoning)"
        )
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args(SCRIPT_DIR)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set. Add it to .env or set as env var.")
        sys.exit(1)

    print("Loading gold standard labels...")
    gold_rows = load_gold_standard(split=args.split)
    print(f"  Loaded {len(gold_rows)} gold standard tweets (split={args.split})")

    print("Loading tweet texts...")
    tweet_texts = load_tweet_texts()
    print(f"  Loaded {len(tweet_texts)} tweet texts")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    for condition in conditions:
        print(f"\n{'#' * 60}")
        active_model = "deepseek-reasoner" if condition == "cot-thinking" else MODEL
        print(f"# CONDITION: {condition} (model: {active_model})")
        print(f"{'#' * 60}")

        results = run_predictions_deepseek(
            gold_rows,
            tweet_texts,
            condition,
            predict_tweet,
            client,
            limit=args.limit,
            workers=args.workers,
            pause_seconds=args.pause_seconds,
        )

        if results:
            out_path = output_path_for_condition(condition, args.output_dir)
            write_csv(results, out_path)
            print(f"\nSaved {len(results)} predictions to {out_path}")
        else:
            print("\nNo predictions generated. Check prediction_errors.log.")
