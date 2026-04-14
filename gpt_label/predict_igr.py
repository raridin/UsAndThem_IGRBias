"""Predict IGR and Plutchik emotions for congressional tweets using OpenAI GPT.

Usage:
    python3 gpt_label/predict_igr.py --condition all --split test
    python3 gpt_label/predict_igr.py --condition zero-shot --split dev

Setup:
    pip install openai python-dotenv
    Add OPENAI_API_KEY=sk-... to .env
"""

import os
import sys
import logging

from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data import load_gold_standard, load_tweet_texts
from shared.prompts import PROMPT_BUILDERS, CONDITIONS
from shared.predict import (
    run_predictions, write_csv, output_path_for_condition, parse_args,
    MAX_TOKENS, TEMPERATURE,
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
# GPT-specific model config
# ---------------------------------------------------------------------------
MODEL = "gpt-4o"

def predict_tweet(client, masked_tweet, condition):
    """Call OpenAI API and return the raw response text."""

    prompt = PROMPT_BUILDERS[condition](masked_tweet)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": "You are a research annotation assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in predict_tweets: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args(SCRIPT_DIR)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or set as env var.")
        sys.exit(1)

    print("Loading gold standard labels...")
    gold_rows = load_gold_standard(split=args.split)
    print(f"  Loaded {len(gold_rows)} gold standard tweets (split={args.split})")

    print("Loading tweet texts...")
    tweet_texts = load_tweet_texts()
    print(f"  Loaded {len(tweet_texts)} tweet texts")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    for condition in conditions:
        print(f"\n{'#' * 60}")
        print(f"# CONDITION: {condition} (model: {MODEL})")
        print(f"{'#' * 60}")

        results = run_predictions(gold_rows, tweet_texts, condition, predict_tweet, client)

        if results:
            out_path = output_path_for_condition(condition, args.output_dir)
            write_csv(results, out_path)
            print(f"\nSaved {len(results)} predictions to {out_path}")
        else:
            print("\nNo predictions generated. Check prediction_errors.log.")
