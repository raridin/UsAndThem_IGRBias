# IGR Prediction Pipeline Design

## Purpose

Build a Python script (`predict_igr.py`) that predicts interpersonal group relationship (IGR) and emotion for congressional tweets using the Claude API. The model predicts from text alone (no party info provided), and predictions are compared against ground truth derived from party affiliations.

## Architecture

Single-file script with five sequential stages: Handle Loading → Tweet Scanning → Sampling → API Prediction → CSV Output.

### Stage 1: Handle Loader

Parses all 8 congress handle CSVs (`twitter-handles/Congress Twitter Handles - {110..117}.csv`) into a unified `dict[str, str]` mapping `lowercase_handle → party`.

Three column format variants exist:

| Sessions | Handle column | Party column | Notes |
|----------|--------------|--------------|-------|
| 110-111 | `twitter_handle` | `party` | Simple CSV |
| 112-115 | `twitter_account` | `party` | Includes dw_nominate scores |
| 116 | — | — | **No party column — skip this file** |
| 117 | `Link` (full URL) | `Party` | Handle extracted from URL tail; `Name` is "Last, First" |

Deduplication: if a handle appears in multiple sessions with different parties, the latest session wins (iterate 110→117 in order).

### Stage 2: Tweet Scanner

For each CSV in `tweets/`:

1. **Tweeter party**: Cross-reference the `username` column (lowercased) against the handle lookup. Fall back to the party character in the filename (e.g., `Adams_Alma_D_NC.csv` → `D`).
2. **Parse mentions**: The `mentions` column contains a JSON list of dicts, each with a `screen_name` key. Parse it via `json.loads()` (handle `ast.literal_eval()` as fallback for Python-style dicts).
3. **Filter**: Keep tweets where exactly one mention's `screen_name` resolves to a congressperson in the handle lookup.
4. **Collect**: Store `(tweet_id, tweeter_handle, tweeter_party, mentioned_handle, mentioned_party, tweet_text)` tuples.

### Stage 3: Sampler

- Collect all eligible tweets from Stage 2 into a list.
- Random sample of `N` tweets (default 50, configurable via `--sample-size` CLI arg).
- `random.seed(42)` for reproducibility.

### Stage 4: Claude API Caller

For each sampled tweet:

**Handle masking**: Replace all occurrences of the mentioned handle (case-insensitive) with `@Doe` in the tweet text. Also replace `@`-prefixed variants.

**Prompt** (JSON-output optimized):

```
You are annotating tweets for a political communication research study.

Given this tweet, determine:
1. Interpersonal Group Relationship (IGR): Does the speaker appear to be talking about someone in their own group (In-Group) or a different group (Out-Group)? Base this only on linguistic cues, tone, and context.
2. Emotion toward @Doe: Choose exactly one from: Admiration, Anger, Disgust, Fear, Interest, Joy, Sadness, Surprise, No Emotion.

Tweet:
"{masked_tweet}"

Respond with ONLY this JSON, no other text:
{"igr": "In-Group" or "Out-Group", "emotion": "<one emotion>", "reasoning": "<1-2 sentences>"}
```

**API parameters**:
- Model: `claude-sonnet-4-6`
- `temperature=0`
- `max_tokens=256`

**Rate limiting**: `time.sleep(1.0)` between calls.

**Error handling**: On any exception (API error, JSON parse failure, timeout), log the error with tweet_id to `prediction_errors.log` and skip the tweet. Continue processing remaining tweets.

**Response parsing**: Parse the response text as JSON. If the response contains markdown code fences, strip them before parsing.

### Stage 5: Output Writer

Write `predictions.csv` with columns:

| Column | Source |
|--------|--------|
| `tweet_id` | Tweet CSV `id` column |
| `tweeter_handle` | Tweet CSV `username` column |
| `mentioned_handle` | From parsed mentions |
| `tweet_text` | Original (unmasked) tweet text |
| `predicted_igr` | Model output `igr` field |
| `predicted_emotion` | Model output `emotion` field |
| `actual_igr` | `"In-Group"` if tweeter_party == mentioned_party, else `"Out-Group"` |
| `model_reasoning` | Model output `reasoning` field |

## CLI Interface

```
python predict_igr.py [--sample-size 50] [--output predictions.csv] [--seed 42]
```

All arguments optional with sensible defaults.

## Dependencies

- `anthropic` — Claude API client
- `python-dotenv` — loads `.env` for `ANTHROPIC_API_KEY`
- Standard library: `csv`, `json`, `os`, `glob`, `time`, `random`, `logging`, `argparse`, `re`, `ast`

## Progress Reporting

Print a progress line for each API call: `[12/50] Processing tweet 1397953591642624005...`

Print a summary at the end: total processed, successful predictions, errors, and accuracy breakdown (predicted vs actual IGR match rate).
