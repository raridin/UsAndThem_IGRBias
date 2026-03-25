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
                    # Normalize non-standard party codes (e.g., "ID" -> "I")
                    party = {"ID": "I"}.get(party, party)
                    handle_to_party[handle.lower()] = party

    return handle_to_party


if __name__ == "__main__":
    lookup = load_handle_to_party()
    print(f"Loaded {len(lookup)} handles")
    # Show a few examples
    for h, p in list(lookup.items())[:5]:
        print(f"  {h} -> {p}")
