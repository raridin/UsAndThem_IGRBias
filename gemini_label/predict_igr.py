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

import google.generativeai as genai
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
MODEL = "gemini-2.5-flash"   # fast + stable (you can switch to gemini-2.5-pro)


def predict_tweet(client, masked_tweet, condition):
    base_prompt = PROMPT_BUILDERS[condition](masked_tweet)

    #Force JSON output
    strict_instruction = """
Return ONLY valid JSON in this exact format:
{
  "igr": "<In-Group or Out-Group>",
  "emotion": "<one of: Admiration, Anger, Disgust, Fear, Interest, Joy, Sadness, Surprise, No Emotion>",
  "reasoning": "<1-2 sentences>"
}

Do NOT include any extra text.
Do NOT use markdown.
Do NOT explain anything outside JSON.
"""

    prompt = base_prompt + "\n\n" + strict_instruction

    try:
        response = client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=MAX_TOKENS,
            ),
        )

        if not response or not response.text:
            logging.error(f"[{condition}] Empty response from Gemini")
            return '{"igr": "Unknown", "emotion": "None", "reasoning": "No response"}'

        return response.text.strip()

    except Exception as e:
        logging.error(f"[{condition}] Error: {e}")
        return '{"igr": "Unknown", "emotion": "None", "reasoning": "API failure"}'
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

    #Initialize Gemini
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(MODEL)

    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    for condition in conditions:
        print(f"\n{'#' * 60}")
        print(f"# CONDITION: {condition} (model: {MODEL})")
        print(f"{'#' * 60}")

        results = run_predictions(
            gold_rows,
            tweet_texts,
            condition,
            predict_tweet,
            client
        )

        if results:
            out_path = output_path_for_condition(condition, args.output_dir)
            write_csv(results, out_path)
            print(f"\nSaved {len(results)} predictions to {out_path}")
        else:
            print("\nNo predictions generated. Check prediction_errors.log.")