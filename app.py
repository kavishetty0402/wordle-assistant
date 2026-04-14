"""
Wordle Assistant & Simulator
-----------------------------
Usage:
  python app.py                       # interactive assistant mode
  python app.py --simulate            # run solver on every word, print stats
  python app.py --simulate --sample 500  # simulate on 500 random words (faster)
  python app.py --first CRANE         # force a specific first guess
  python app.py --word STARE          # simulate a single target word (verbose)
  python app.py --best-opener         # find the best opening words
  python app.py --analyze SLATE       # deep analysis of one guess word

Pattern input format (after each guess):
  G = green  (right letter, right position)
  Y = yellow (right letter, wrong position)
  B = black  (letter not in word)
  e.g.  GYBBB  or  GYBBG
"""

import sys
import math
import argparse
import random
from collections import defaultdict
from typing import List, Tuple, Optional

from wordle_solver import (
    WORD_LEN, BLACK, YELLOW, GREEN, ALL_GREEN, NUM_PATTERNS,
    get_pattern, pattern_to_int, int_to_pattern,
    compute_entropy, actual_information,
    filter_words, best_guesses, play_game, run_simulation,
    pattern_to_emoji, pattern_to_label,
    build_pattern_matrix, best_guesses_matrix, entropy_from_matrix,
)
import numpy as np


# ─── Word list ────────────────────────────────────────────────────────────────

def load_words(path: str = 'words.txt') -> List[str]:
    try:
        with open(path) as f:
            words = [line.strip().lower() for line in f]
        words = [w for w in words if len(w) == WORD_LEN and w.isalpha()]
        if not words:
            raise ValueError("Empty word list")
        return words
    except FileNotFoundError:
        print(f"Word list not found at '{path}'.")
        print("Run:  python fetch_words.py  to download it.")
        sys.exit(1)


# ─── Pattern parsing ──────────────────────────────────────────────────────────

_PATTERN_MAP = {
    'g': GREEN, 'G': GREEN, '2': GREEN,
    'y': YELLOW, 'Y': YELLOW, '1': YELLOW,
    'b': BLACK,  'B': BLACK,  '0': BLACK,
}

def parse_pattern(s: str) -> Optional[Tuple[int, ...]]:
    s = s.strip()
    if len(s) != WORD_LEN or not all(c in _PATTERN_MAP for c in s):
        return None
    return tuple(_PATTERN_MAP[c] for c in s)


# ─── Display helpers ─────────────────────────────────────────────────────────

DIVIDER = '─' * 62
THICK   = '═' * 62


def print_top_guesses(top: List[Tuple[str, float, bool]]) -> None:
    print(f"\n  {'WORD':<10}{'EXPECTED INFO':>16}{'IN ANSWER SET':>16}")
    print(f"  {'─'*10}{'─'*16}{'─'*16}")
    for word, entropy, is_poss in top:
        poss_tag = '  ✓' if is_poss else ''
        print(f"  {word.upper():<10}{entropy:>14.4f} bits{poss_tag:>13}")


def print_game_state(possible: List[str], show_words_limit: int = 30) -> None:
    n = len(possible)
    print(f"\n  Words remaining: {n}")
    if n <= show_words_limit:
        cols = 6
        for i in range(0, n, cols):
            row = possible[i:i+cols]
            print("    " + "  ".join(w.upper() for w in row))
    else:
        preview = "  ".join(w.upper() for w in possible[:show_words_limit])
        print(f"    {preview}  ... (+{n - show_words_limit} more)")


# ─── Interactive assistant ────────────────────────────────────────────────────

def _matrix_top_guesses(matrix: np.ndarray, word_to_idx: dict, all_words: List[str],
                         possible: List[str], top_n: int) -> List[Tuple[str, float, bool]]:
    """Rank guesses using the precomputed matrix — fast even for the full word set."""
    n = len(all_words)
    possible_set = set(possible)
    mask = np.zeros(n, dtype=bool)
    for w in possible:
        if w in word_to_idx:
            mask[word_to_idx[w]] = True

    ranked = best_guesses_matrix(matrix, mask, n, top_n=top_n)
    return [(all_words[idx], ent, all_words[idx] in possible_set) for idx, ent in ranked]


def _matrix_entropy_for_guess(matrix: np.ndarray, word_to_idx: dict, all_words: List[str],
                               guess: str, possible: List[str]) -> float:
    """Compute expected entropy for one specific guess using the matrix."""
    if guess not in word_to_idx:
        # Word not in matrix (user typed something outside the word list)
        return compute_entropy(guess, possible)
    n = len(all_words)
    mask = np.zeros(n, dtype=bool)
    for w in possible:
        if w in word_to_idx:
            mask[word_to_idx[w]] = True
    return entropy_from_matrix(matrix, word_to_idx[guess], mask)


def assistant_mode(all_words: List[str], matrix: np.ndarray, word_to_idx: dict,
                   forced_first: Optional[str] = None) -> None:
    print()
    print(THICK)
    print("  WORDLE ASSISTANT  —  Entropy-based solver")
    print(THICK)
    n = len(all_words)
    max_info = math.log2(n)
    print(f"  Word space : {n:,} words")
    print(f"  Max info   : {max_info:.2f} bits  (= log2({n}))")
    print()

    possible = list(all_words)

    # Rank first guesses using the matrix (fast — uses cached precomputed patterns)
    print("  Computing top first guesses...")
    top = _matrix_top_guesses(matrix, word_to_idx, all_words, possible, top_n=10)
    print(f"\n  Best opening guesses (maximise expected information):")
    print_top_guesses(top)

    if forced_first:
        recommended = forced_first.lower()
        if recommended not in {w for w, _, _ in top}:
            h = _matrix_entropy_for_guess(matrix, word_to_idx, all_words, recommended, possible)
            print(f"\n  Forced first guess: {recommended.upper()}  ({h:.4f} bits expected)")
    else:
        recommended = top[0][0]

    total_info = 0.0

    for attempt in range(1, 7):
        current_max = math.log2(len(possible)) if len(possible) > 1 else 0
        print(f"\n{DIVIDER}")
        print(f"  Guess #{attempt}  │  {len(possible):,} words remaining"
              f"  │  log2 = {current_max:.2f} bits")
        print(DIVIDER)

        # Prompt for guess
        while True:
            raw = input(f"\n  Your guess (or ENTER for '{recommended.upper()}'): ").strip().lower()
            if raw == '':
                guess = recommended
                print(f"  Using: {guess.upper()}")
                break
            if raw == 'quit':
                print("  Goodbye!")
                return
            if len(raw) == WORD_LEN and raw.isalpha():
                guess = raw
                break
            print("  Please enter a 5-letter word.")

        expected = _matrix_entropy_for_guess(matrix, word_to_idx, all_words, guess, possible)
        if current_max > 0:
            print(f"\n  Expected information from {guess.upper()}: {expected:.4f} bits"
                  f"  ({100*expected/current_max:.1f}% of max)")

        # Build pattern distribution over current possible set
        pattern_dist = defaultdict(list)
        for w in possible:
            p = get_pattern(guess, w)
            pattern_dist[p].append(w)
        print(f"  Produces {len(pattern_dist)} distinct patterns out of 243")

        # Prompt for pattern
        while True:
            raw_pat = input("  Result pattern (G/Y/B for each letter, e.g. GYBBB): ").strip()
            pattern = parse_pattern(raw_pat)
            if pattern is not None:
                break
            print("  Invalid pattern. Use exactly 5 characters from G/Y/B.")

        # Compute actual information
        n_before = len(possible)
        bucket = pattern_dist.get(pattern, [])
        n_after = len(bucket)

        if n_after == 0:
            print("\n  ⚠ No words match this pattern! Double-check your colors.")
            continue

        possible = bucket
        info = actual_information(n_before, n_after)
        total_info += info

        print(f"\n  {pattern_to_emoji(pattern)}  {guess.upper()}")
        print(f"\n  ┌─ Information Analysis ───────────────────────────────")
        print(f"  │ Expected info (E[I]) : {expected:.4f} bits")
        print(f"  │ Actual info (I)      : {info:.4f} bits")
        delta = info - expected
        if delta >= 0:
            print(f"  │ Above expected       : +{delta:.4f} bits  (lucky!)")
        else:
            print(f"  │ Below expected       : {delta:.4f} bits")
        print(f"  │ p(this pattern)      : {n_after}/{n_before}"
              f" = {n_after/n_before:.6f}")
        print(f"  │ I = -log2(p)         : {info:.4f} bits")
        print(f"  │ Words: {n_before} → {n_after}")
        print(f"  │ Cumulative info      : {total_info:.4f} / {max_info:.2f} bits")
        print(f"  └─────────────────────────────────────────────────────")

        if pattern == ALL_GREEN:
            print(f"\n  ★ Solved in {attempt} guess{'es' if attempt > 1 else ''}!")
            return

        if not possible:
            print("\n  ✗ No valid words remain — the target may not be in the word list.")
            return

        print_game_state(possible)

        # Recommend next guess using the matrix
        if len(possible) > 1:
            print(f"\n  Computing next best guesses...")
            top = _matrix_top_guesses(matrix, word_to_idx, all_words, possible, top_n=10)
            print(f"\n  Best next guesses:")
            print_top_guesses(top)
            recommended = top[0][0]
        else:
            recommended = possible[0]

    print("\n  ✗ Could not solve in 6 guesses.")
    if possible:
        print(f"  Remaining candidates: {', '.join(w.upper() for w in possible)}")


# ─── Single-word verbose simulation ──────────────────────────────────────────

def single_word_sim(target: str, all_words: List[str], first_guess: Optional[str] = None) -> None:
    word_set = set(all_words)
    if target not in word_set:
        print(f"  '{target}' is not in the word list.")
        return

    if first_guess is None:
        print("  Computing best first guess...")
        first_guess = best_guesses(all_words, all_words, top_n=1)[0][0]

    result = play_game(target, all_words, first_guess=first_guess)

    print()
    print(THICK)
    print(f"  SIMULATION  —  Target: {target.upper()}  │  First guess: {first_guess.upper()}")
    print(THICK)

    total_info = 0.0
    for i, g in enumerate(result['guesses'], 1):
        total_info += g['actual_info']
        print(f"\n  Guess #{i}: {g['guess'].upper()}  {g['emoji']}")
        print(f"    Words before : {g['n_before']:>5}  →  after: {g['n_after']:>5}")
        print(f"    Expected info: {g['expected_info']:.4f} bits")
        print(f"    Actual info  : {g['actual_info']:.4f} bits   (cumulative: {total_info:.4f} bits)")

    outcome = f"Solved in {result['attempts']}" if result['solved'] else "FAILED (>6 guesses)"
    print(f"\n  Result: {outcome}")
    print()


# ─── Best opener analysis ────────────────────────────────────────────────────

def best_opener_mode(all_words: List[str], top_n: int = 20) -> None:
    n = len(all_words)
    max_info = math.log2(n)

    print(f"\n  Ranking all {n:,} words by expected information...")
    print(f"  Each word is scored against all {n:,} possible targets.")
    print(f"  Max possible entropy: {max_info:.4f} bits\n")

    top = best_guesses(all_words, all_words, top_n=top_n)

    print(f"\n  Top {top_n} opening guesses:")
    print(f"  {'Rank':<6}{'Word':<10}{'E[I] (bits)':<16}{'Efficiency':<12}")
    print(f"  {'─'*44}")
    for rank, (word, ent, _) in enumerate(top, 1):
        pct = 100 * ent / max_info
        print(f"  {rank:<6}{word.upper():<10}{ent:<16.4f}{pct:.1f}%")


# ─── Analyze a specific word ─────────────────────────────────────────────────

def analyze_mode(word: str, all_words: List[str]) -> None:
    word = word.lower()
    if word not in set(all_words):
        print(f"  '{word}' is not in the word list.")
        return

    n = len(all_words)
    max_info = math.log2(n)
    entropy = compute_entropy(word, all_words)

    # Build pattern distribution
    dist = defaultdict(list)
    for w in all_words:
        p = get_pattern(word, w)
        dist[p].append(w)

    print(f"\n{'═'*62}")
    print(f"  ANALYSIS: {word.upper()}")
    print(f"{'═'*62}")
    print(f"  Against {n:,} possible words")
    print(f"  E[I] = {entropy:.4f} bits  ({100*entropy/max_info:.1f}% of max {max_info:.2f})")
    print(f"  Distinct patterns: {len(dist)} / 243")

    # Expected remaining words
    exp_remaining = sum(len(ws)**2 for ws in dist.values()) / n
    print(f"  Expected remaining words: {exp_remaining:.1f}")
    print(f"  Expected reduction: {n} → {exp_remaining:.1f}"
          f" ({100*(1 - exp_remaining/n):.1f}% eliminated)")

    # Show all patterns sorted by frequency
    sorted_dist = sorted(dist.items(), key=lambda x: -len(x[1]))

    print(f"\n  {'Pattern':<8}{'Visual':<14}{'Count':<8}"
          f"{'p(x)':<12}{'I=-log2(p)':<13}{'p·I'}")
    print(f"  {'─'*62}")
    for pat, words in sorted_dist[:30]:
        count = len(words)
        prob = count / n
        info = -math.log2(prob)
        contrib = prob * info
        print(f"  {pattern_to_label(pat):<8}{pattern_to_emoji(pat):<14}{count:<8}"
              f"{prob:<12.6f}{info:<13.4f}{contrib:.4f}")
        if count <= 5:
            print(f"           → {', '.join(w.upper() for w in sorted(words))}")

    if len(sorted_dist) > 30:
        print(f"  ... and {len(sorted_dist) - 30} more patterns")

    # Worst-case bucket
    worst_pat, worst_words = sorted_dist[0]
    print(f"\n  Worst-case bucket: {pattern_to_label(worst_pat)}"
          f" ({len(worst_words)} words, {100*len(worst_words)/n:.1f}%)")
    if len(worst_words) <= 15:
        print(f"    {', '.join(w.upper() for w in sorted(worst_words))}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Wordle entropy-based solver — assistant and simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                           Interactive assistant
  python app.py --word crane              Simulate solving for CRANE
  python app.py --simulate                Simulate all 12,972 games
  python app.py --simulate --sample 500   Simulate 500 random games
  python app.py --best-opener             Find the best opening words
  python app.py --best-opener --top 50    Show top 50 openers
  python app.py --analyze slate           Deep analysis of SLATE as a guess
        """)

    parser.add_argument('--simulate', action='store_true',
                        help='Run solver on all words and print statistics')
    parser.add_argument('--word', metavar='TARGET',
                        help='Simulate a single target word verbosely')
    parser.add_argument('--first', metavar='GUESS',
                        help='Force a specific first guess word')
    parser.add_argument('--words', metavar='FILE', default='words.txt',
                        help='Path to word list file (default: words.txt)')
    parser.add_argument('--best-opener', action='store_true',
                        help='Find and rank the best opening words')
    parser.add_argument('--analyze', metavar='WORD',
                        help='Deep analysis of a specific guess word')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top results to display (default: 20)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Simulate on N random words instead of all')
    parser.add_argument('--answers', metavar='FILE', default='answers.txt',
                        help='Path to answer word list for simulation targets')

    args = parser.parse_args()

    all_words = load_words(args.words)
    first_guess = args.first.lower() if args.first else None

    if first_guess and first_guess not in set(all_words):
        print(f"Warning: '{first_guess}' is not in the word list. Using it anyway.")

    # Load the precomputed pattern matrix once — used by all modes
    print("  Loading pattern matrix (cached)...")
    matrix = build_pattern_matrix(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    print()

    if args.analyze:
        analyze_mode(args.analyze, all_words)
    elif args.best_opener:
        best_opener_mode(all_words, top_n=args.top)
    elif args.word:
        single_word_sim(args.word.lower(), all_words, first_guess=first_guess)
    elif args.simulate:
        # Determine target words
        targets = None
        try:
            answer_words = load_words(args.answers)
            print(f"  Using {len(answer_words)} answer words as targets")
            targets = answer_words
        except SystemExit:
            print(f"  No separate answer list found — simulating on all {len(all_words)} words")
            targets = all_words

        if args.sample:
            random.seed(42)
            targets = random.sample(targets, min(args.sample, len(targets)))
            print(f"  Sampled {len(targets)} random target words")

        run_simulation(all_words, first_guess=first_guess, targets=targets)
    else:
        assistant_mode(all_words, matrix, word_to_idx, forced_first=first_guess)


if __name__ == '__main__':
    main()
