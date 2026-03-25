import anthropic

# ============ CONFIG ============
API_KEY = ""
MODEL = "claude-sonnet-4-20250514"

# ============ EXAMPLE DATA ============
tweet = """With more than 36 million Americans out of a job, @Doe and House Democrats 
should stop wasting time on radical legislation that will never become law and instead 
focus on ways to safely re-open our economy and get Americans back to work."""

gold_label = {
    "igr": "Out-Group",
    "emotion": "Anger"
}

# ============ PROMPT ============
prompt = f"""You are annotating tweets for a research study on interpersonal group relationships.

Given the following tweet, determine:
1. **Interpersonal Group Relationship (IGR)**: Is the speaker talking about someone in their "In-Group" (same social/political group) or "Out-Group" (different group)?
2. **Interpersonal Emotion**: What emotion is the speaker expressing TOWARD @Doe? Choose from: Admiration, Anger, Disgust, Fear, Interest, Joy, Sadness, Surprise, or No Emotion.

Tweet:
"{tweet.strip()}"

Respond in exactly this format:
IGR: [In-Group or Out-Group]
Emotion: [emotion label]
Reasoning: [1-2 sentences explaining your decision]"""

# ============ API CALL ============
client = anthropic.Anthropic(api_key=API_KEY)

message = client.messages.create(
    model=MODEL,
    max_tokens=256,
    temperature=0,  # deterministic output
    messages=[
        {"role": "user", "content": prompt}
    ]
)

response = message.content[0].text

# ============ OUTPUT ============
print("=" * 50)
print("TWEET:")
print(tweet.strip())
print("=" * 50)
print("\nMODEL PREDICTION:")
print(response)
print("\n" + "=" * 50)
print("GOLD LABELS:")
print(f"  IGR: {gold_label['igr']}")
print(f"  Emotion: {gold_label['emotion']}")
print("=" * 50)

# ============ SIMPLE EVAL ============
response_lower = response.lower()
igr_correct = gold_label["igr"].lower() in response_lower
emotion_correct = gold_label["emotion"].lower() in response_lower

print("\nQUICK CHECK:")
print(f"  IGR Match: {'✓' if igr_correct else '✗'}")
print(f"  Emotion Match: {'✓' if emotion_correct else '✗'}")