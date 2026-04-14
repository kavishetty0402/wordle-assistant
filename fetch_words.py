"""
Downloads the standard Wordle word list (~12,972 five-letter words) and saves to words.txt.
Run once:  python fetch_words.py
"""

import urllib.request
import sys

# Two sources: NYT Wordle allowed guesses + answers (combined ~14k words).
# We use the well-known open-source Wordle word lists from cfreshman on GitHub.
SOURCES = [
    # valid guesses (non-answer words)
    "https://raw.githubusercontent.com/cfreshman/wordle-5757/main/words.txt",
    # answer list
    "https://raw.githubusercontent.com/cfreshman/wordle-5757/main/words-answers.txt",
]

FALLBACK = [
    "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words",
]


def fetch(url: str) -> list[str]:
    print(f"  Fetching {url} ...", end=' ', flush=True)
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode()
        words = [w.strip().lower() for w in text.splitlines()]
        words = [w for w in words if len(w) == 5 and w.isalpha()]
        print(f"{len(words)} words")
        return words
    except Exception as e:
        print(f"FAILED ({e})")
        return []


def main():
    all_words: set[str] = set()

    for url in SOURCES:
        all_words |= set(fetch(url))

    if not all_words:
        print("Primary sources failed, trying fallback...")
        for url in FALLBACK:
            all_words |= set(fetch(url))

    if not all_words:
        print("All sources failed. Check your internet connection.")
        sys.exit(1)

    words = sorted(all_words)
    with open('words.txt', 'w') as f:
        f.write('\n'.join(words) + '\n')

    print(f"\nSaved {len(words)} words to words.txt")


if __name__ == '__main__':
    main()
