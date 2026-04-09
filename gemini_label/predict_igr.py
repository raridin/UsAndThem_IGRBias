"""Predict IGR and Plutchik emotions for congressional tweets using Google Gemini.

Usage:
    python3 gemini_label/predict_igr.py --condition all --split test
    python3 gemini_label/predict_igr.py --condition zero-shot --split dev

Setup:
    pip install google-generativeai python-dotenv
    Add GOOGLE_API_KEY=... to .env
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
# Gemini-specific model config
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-pro"  # TODO: confirm model name


def predict_tweet(client, masked_tweet, condition):
    """Call Gemini API and return the raw response text.

    TODO: Implement this function.
    - Use client.generate_content() or the chat API
    - For cot-thinking: Gemini has a thinking mode — check google-generativeai docs
    """
    prompt = PROMPT_BUILDERS[condition](masked_tweet)

    # TODO: uncomment and implement
    # response = client.generate_content(
    #     prompt,
    #     generation_config=genai.types.GenerationConfig(
    #         temperature=TEMPERATURE,
    #         max_output_tokens=MAX_TOKENS,
    #     ),
    # )
    # return response.text

    raise NotImplementedError(
        "Gemini predict_tweet not yet implemented. "
        "See the TODO comments above for guidance."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args(SCRIPT_DIR)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set. Add it to .env or set as env var.")
        sys.exit(1)

    print("Loading gold standard labels...")
    gold_rows = load_gold_standard(split=args.split)
    print(f"  Loaded {len(gold_rows)} gold standard tweets (split={args.split})")

    print("Loading tweet texts...")
    tweet_texts = load_tweet_texts()
    print(f"  Loaded {len(tweet_texts)} tweet texts")

    # TODO: uncomment after pip install google-generativeai
    # import google.generativeai as genai
    # genai.configure(api_key=api_key)
    # client = genai.GenerativeModel(MODEL)
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
