"""Predict IGR and Plutchik emotions for congressional tweets using DeepSeek.

Usage:
    python3 deepseek_label/predict_igr.py --condition all --split test
    python3 deepseek_label/predict_igr.py --condition zero-shot --split dev

Setup:
    pip install openai python-dotenv
    Add DEEPSEEK_API_KEY=... to .env

Note: DeepSeek uses an OpenAI-compatible API, so we use the openai SDK
      with a custom base_url.
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
# DeepSeek-specific model config
# ---------------------------------------------------------------------------
MODEL = "deepseek-chat"  # TODO: confirm model name (deepseek-chat or deepseek-reasoner)
BASE_URL = "https://api.deepseek.com"


def predict_tweet(client, masked_tweet, condition):
    """Call DeepSeek API and return the raw response text.

    TODO: Implement this function.
    - DeepSeek uses OpenAI-compatible API (openai SDK with custom base_url)
    - For cot-thinking: use deepseek-reasoner model which has native CoT
    """
    prompt = PROMPT_BUILDERS[condition](masked_tweet)

    # TODO: uncomment and implement
    # model = "deepseek-reasoner" if condition == "cot-thinking" else MODEL
    # response = client.chat.completions.create(
    #     model=model,
    #     temperature=TEMPERATURE,
    #     max_tokens=MAX_TOKENS,
    #     messages=[
    #         {"role": "system", "content": "You are a research annotation assistant."},
    #         {"role": "user", "content": prompt},
    #     ],
    # )
    # return response.choices[0].message.content

    raise NotImplementedError(
        "DeepSeek predict_tweet not yet implemented. "
        "See the TODO comments above for guidance."
    )


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

    # TODO: uncomment after pip install openai
    # from openai import OpenAI
    # client = OpenAI(api_key=api_key, base_url=BASE_URL)
    client = None

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
