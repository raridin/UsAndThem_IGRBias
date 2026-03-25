# IGR Prediction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `predict_igr.py` — a single Python script that samples congressional tweets mentioning other congress members, calls Claude API to predict IGR and emotion, and outputs `predictions.csv` with predictions vs ground truth.

**Architecture:** Single-file pipeline with five stages: load handle→party lookup from congress CSVs, scan tweet CSVs for eligible tweets (exactly one congress-member mention), sample N tweets, call Claude API with masked handles, write CSV output.

**Tech Stack:** Python 3, `anthropic` SDK, `python-dotenv`, standard library (`csv`, `json`, `ast`, `os`, `glob`, `time`, `random`, `logging`, `argparse`, `re`)

---

### Task 1: Handle Loader — Build handle-to-party lookup

**Files:**
- Create: `predict_igr.py`

This task creates the script with the handle loading function. It parses 7 of the 8 congress handle CSVs (skipping 116 which has no party data) into a single `dict[str, str]` mapping `lowercase_handle → party`.

- [ ] **Step 1: Create `predict_igr.py` with handle loader function**

```python
import csv
import os
import glob


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
                    handle_to_party[handle.lower()] = party

    return handle_to_party


if __name__ == "__main__":
    lookup = load_handle_to_party()
    print(f"Loaded {len(lookup)} handles")
    # Show a few examples
    for h, p in list(lookup.items())[:5]:
        print(f"  {h} -> {p}")
```

- [ ] **Step 2: Run to verify handle loading works**

Run: `python predict_igr.py`
Expected: Output like:
```
Loaded <N> handles
  neilabercrombie -> D
  repgaryackerman -> D
  ...
```

- [ ] **Step 3: Commit**

```bash
git add predict_igr.py
git commit -m "feat: add handle-to-party loader for congress CSVs"
```

---

### Task 2: Tweet Scanner — Find eligible tweets

**Files:**
- Modify: `predict_igr.py`

Add function to scan all tweet CSVs, find tweets with exactly one congress-member mention, and collect them with party info.

- [ ] **Step 1: Add tweet scanning function**

Add these imports at the top of `predict_igr.py`:

```python
import json
import ast
import re
```

Add this function after `load_handle_to_party`:

```python
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
```

- [ ] **Step 2: Update `__main__` to test scanning**

Replace the `if __name__` block:

```python
if __name__ == "__main__":
    lookup = load_handle_to_party()
    print(f"Loaded {len(lookup)} handles")

    eligible = scan_tweets("tweets", lookup)
    print(f"Found {len(eligible)} eligible tweets")
    if eligible:
        ex = eligible[0]
        print(f"  Example: @{ex['tweeter_handle']} ({ex['tweeter_party']}) mentions @{ex['mentioned_handle']} ({ex['mentioned_party']})")
        print(f"  Tweet: {ex['tweet_text'][:120]}...")
```

- [ ] **Step 3: Run to verify scanning works**

Run: `python predict_igr.py`
Expected: Output showing handle count and eligible tweet count (likely thousands), plus one example.

- [ ] **Step 4: Commit**

```bash
git add predict_igr.py
git commit -m "feat: add tweet scanner to find single-mention congress tweets"
```

---

### Task 3: Sampler + CLI — Random sampling with CLI args

**Files:**
- Modify: `predict_igr.py`

Add argparse CLI and random sampling.

- [ ] **Step 1: Add imports and CLI setup**

Add `import random` and `import argparse` to the top imports (add to the existing import block).

Add this function after `scan_tweets`:

```python
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
```

- [ ] **Step 2: Update `__main__` with sampling**

Replace the `if __name__` block:

```python
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

    for t in sample[:3]:
        print(f"    @{t['tweeter_handle']} ({t['tweeter_party']}) -> @{t['mentioned_handle']} ({t['mentioned_party']})")
```

- [ ] **Step 3: Run to verify sampling**

Run: `python predict_igr.py --sample-size 10`
Expected: Shows 10 sampled tweets with party info, 3 examples printed.

- [ ] **Step 4: Commit**

```bash
git add predict_igr.py
git commit -m "feat: add CLI args and random sampling for tweet selection"
```

---

### Task 4: Claude API Caller — Predict IGR and emotion

**Files:**
- Modify: `predict_igr.py`

Add the API calling function with handle masking, structured prompting, rate limiting, and error handling.

- [ ] **Step 1: Add imports**

Add to the top imports:

```python
import time
import logging

from dotenv import load_dotenv
from anthropic import Anthropic
```

Add right after the imports:

```python
load_dotenv()

logging.basicConfig(
    filename="prediction_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(message)s",
)
```

- [ ] **Step 2: Add the masking and prediction functions**

Add after `parse_args`:

```python
def mask_handle(tweet_text, handle):
    """Replace all occurrences of the mentioned handle with @Doe (case-insensitive)."""
    # Replace @handle variant
    masked = re.sub(re.escape(f"@{handle}"), "@Doe", tweet_text, flags=re.IGNORECASE)
    # Replace bare handle if it appears without @
    masked = re.sub(r'(?<!\w)' + re.escape(handle) + r'(?!\w)', "Doe", masked, flags=re.IGNORECASE)
    return masked


def predict_tweet(client, tweet_text, mentioned_handle):
    """Call Claude API to predict IGR and emotion for a tweet.

    Returns dict with keys: igr, emotion, reasoning
    Raises on API or parse errors.
    """
    masked = mask_handle(tweet_text, mentioned_handle)

    prompt = f"""You are annotating tweets for a political communication research study.

Given this tweet, determine:
1. Interpersonal Group Relationship (IGR): Does the speaker appear to be talking about someone in their own group (In-Group) or a different group (Out-Group)? Base this only on linguistic cues, tone, and context.
2. Emotion toward @Doe: Choose exactly one from: Admiration, Anger, Disgust, Fear, Interest, Joy, Sadness, Surprise, No Emotion.

Tweet:
"{masked}"

Respond with ONLY this JSON, no other text:
{{"igr": "In-Group" or "Out-Group", "emotion": "<one emotion>", "reasoning": "<1-2 sentences>"}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    return {
        "igr": result["igr"],
        "emotion": result["emotion"],
        "reasoning": result.get("reasoning", ""),
    }


def run_predictions(sample):
    """Run Claude API predictions on sampled tweets with rate limiting.

    Returns list of result dicts (one per successful prediction).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or set as env var.")
        return []

    client = Anthropic(api_key=api_key)
    results = []
    errors = 0
    total = len(sample)

    for i, tweet in enumerate(sample):
        print(f"  [{i + 1}/{total}] Processing tweet {tweet['tweet_id']}...")

        try:
            prediction = predict_tweet(client, tweet["tweet_text"], tweet["mentioned_handle"])

            actual_igr = "In-Group" if tweet["tweeter_party"] == tweet["mentioned_party"] else "Out-Group"

            results.append({
                "tweet_id": tweet["tweet_id"],
                "tweeter_handle": tweet["tweeter_handle"],
                "mentioned_handle": tweet["mentioned_handle"],
                "tweet_text": tweet["tweet_text"],
                "predicted_igr": prediction["igr"],
                "predicted_emotion": prediction["emotion"],
                "actual_igr": actual_igr,
                "model_reasoning": prediction["reasoning"],
            })

        except Exception as e:
            errors += 1
            logging.error(f"Tweet {tweet['tweet_id']}: {type(e).__name__}: {e}")
            print(f"    ERROR: {e}")

        # Rate limit — skip sleep after last tweet
        if i < total - 1:
            time.sleep(1.0)

    print(f"\n  Completed: {len(results)} successful, {errors} errors")
    return results
```

- [ ] **Step 3: Update `__main__` to call predictions**

Replace the `if __name__` block:

```python
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

    # Predict
    print("\nRunning predictions...")
    results = run_predictions(sample)

    if results:
        print(f"\nResults ready: {len(results)} predictions")
    else:
        print("\nNo predictions generated.")
```

- [ ] **Step 4: Test with `--sample-size 2` (requires valid API key)**

Run: `ANTHROPIC_API_KEY=your-key python predict_igr.py --sample-size 2`
Expected: Two tweets processed with predictions printed, no errors.

- [ ] **Step 5: Commit**

```bash
git add predict_igr.py
git commit -m "feat: add Claude API prediction with handle masking and rate limiting"
```

---

### Task 5: CSV Output + Summary Stats

**Files:**
- Modify: `predict_igr.py`

Add CSV writing and accuracy summary.

- [ ] **Step 1: Add output and summary functions**

Add after `run_predictions`:

```python
def write_csv(results, output_path):
    """Write prediction results to CSV."""
    fieldnames = [
        "tweet_id", "tweeter_handle", "mentioned_handle", "tweet_text",
        "predicted_igr", "predicted_emotion", "actual_igr", "model_reasoning",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    """Print accuracy summary stats."""
    if not results:
        return

    total = len(results)
    igr_correct = sum(1 for r in results if r["predicted_igr"] == r["actual_igr"])
    igr_accuracy = igr_correct / total * 100

    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total predictions: {total}")
    print(f"IGR accuracy:      {igr_correct}/{total} ({igr_accuracy:.1f}%)")

    # Emotion distribution
    emotions = {}
    for r in results:
        e = r["predicted_emotion"]
        emotions[e] = emotions.get(e, 0) + 1

    print(f"\nEmotion distribution:")
    for emotion, count in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"  {emotion}: {count} ({count / total * 100:.1f}%)")

    # IGR breakdown
    in_group_pred = sum(1 for r in results if r["predicted_igr"] == "In-Group")
    out_group_pred = total - in_group_pred
    in_group_actual = sum(1 for r in results if r["actual_igr"] == "In-Group")
    out_group_actual = total - in_group_actual

    print(f"\nIGR breakdown:")
    print(f"  Predicted: {in_group_pred} In-Group, {out_group_pred} Out-Group")
    print(f"  Actual:    {in_group_actual} In-Group, {out_group_actual} Out-Group")
    print(f"{'=' * 50}")
```

- [ ] **Step 2: Update `__main__` with CSV output and summary**

Replace the `if __name__` block:

```python
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

    # Predict
    print("\nRunning predictions...")
    results = run_predictions(sample)

    # Output
    if results:
        write_csv(results, args.output)
        print(f"\nPredictions saved to {args.output}")
        print_summary(results)
    else:
        print("\nNo predictions generated. Check prediction_errors.log for details.")
```

- [ ] **Step 3: Run full pipeline with small sample**

Run: `python predict_igr.py --sample-size 3`
Expected: 3 tweets processed, `predictions.csv` created with 8 columns, summary stats printed.

- [ ] **Step 4: Verify CSV output**

Run: `head -5 predictions.csv`
Expected: Header row + up to 3 data rows with all 8 columns populated.

- [ ] **Step 5: Commit**

```bash
git add predict_igr.py
git commit -m "feat: add CSV output and accuracy summary stats"
```

---

### Task 6: Final polish — add .env template and gitignore updates

**Files:**
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: Create `.env.example`**

```
ANTHROPIC_API_KEY=your-api-key-here
```

- [ ] **Step 2: Update `.gitignore`**

Add these lines to the existing `.gitignore`:

```
.env
prediction_errors.log
predictions.csv
```

- [ ] **Step 3: Commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add .env template and update gitignore for output files"
```
