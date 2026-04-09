"""Shared prompt templates and handle masking for IGR prediction experiments.

All providers use the same prompts to ensure a fair comparison.
"""

import json
import re

from shared.data import EMOTIONS

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = ["zero-shot", "few-shot", "cot-thinking"]

# ---------------------------------------------------------------------------
# Prompt building blocks
# ---------------------------------------------------------------------------

TASK_FRAMING = """You are annotating tweets for a political communication research study.

Given a tweet, determine:
1. Interpersonal Group Relationship (IGR): Does the speaker appear to be talking about someone in their own group (In-Group) or a different group (Out-Group)? Base this only on linguistic cues, tone, and context.
2. Emotions toward @Doe: For each of the 8 Plutchik emotions below, indicate true or false. Multiple emotions can be true. if no clear emotion is expressed toward @Doe, mark ALL emotions as false.

Emotions: Admiration, Anger, Disgust, Fear, Interest, Joy, Sadness, Surprise"""

JSON_SCHEMA = '{"igr": "In-Group" or "Out-Group", "Admiration": true/false, "Anger": true/false, "Disgust": true/false, "Fear": true/false, "Interest": true/false, "Joy": true/false, "Sadness": true/false, "Surprise": true/false, "reasoning": "<1-2 sentences>"}'

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
# Prompt builders
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(masked_tweet):
    return f"""{TASK_FRAMING}

Tweet:
"{masked_tweet}"

Respond with ONLY this JSON, no other text:
{JSON_SCHEMA}"""


def build_few_shot_prompt(masked_tweet):
    examples_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        answer_json = json.dumps(ex["answer"], ensure_ascii=False)
        examples_block += f"\nExample {i} ({ex['label']}):\nTweet: \"{ex['tweet']}\"\nAnswer: {answer_json}\n"

    return f"""{TASK_FRAMING}

Here are some examples:
{examples_block}
Now classify this tweet:
"{masked_tweet}"

Respond with ONLY this JSON, no other text:
{JSON_SCHEMA}"""


def build_cot_thinking_prompt(masked_tweet):
    """Same task as zero-shot — the 'thinking' happens via the provider's
    native extended thinking / reasoning feature, not the prompt."""
    return f"""{TASK_FRAMING}

Tweet:
"{masked_tweet}"

Respond with ONLY this JSON, no other text:
{JSON_SCHEMA}"""


PROMPT_BUILDERS = {
    "zero-shot": build_zero_shot_prompt,
    "few-shot": build_few_shot_prompt,
    "cot-thinking": build_cot_thinking_prompt,
}

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
