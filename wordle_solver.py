"""
Wordle Entropy Solver
---------------------
Uses information theory to find the optimal guess at each step.

Key idea:
  For a guess G and remaining possible words P, the expected information is:
    E[I] = -Σ p(pattern) * log2(p(pattern))   (Shannon entropy)
  where p(pattern) = |words producing that pattern| / |P|

  When we actually observe pattern x, the information gained is:
    I(x) = log2(n_before / n_after)   (= log2 reduction in search space)

Two modes:
  1. Pure Python (no precomputation) — fine for interactive play
  2. Precomputed pattern matrix (numpy) — required for full simulation
"""

import math
import os
import time
import numpy as np
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

WORD_LEN = 5
BLACK, YELLOW, GREEN = 0, 1, 2
ALL_GREEN = (GREEN,) * WORD_LEN
NUM_PATTERNS = 3 ** 5  # 243
ALL_GREEN_INT = sum(GREEN * 3**i for i in range(WORD_LEN))  # 242


# ─── Core pattern computation (pure Python, cached) ─────────────────────────

@lru_cache(maxsize=None)
def get_pattern(guess: str, target: str) -> tuple:
    """
    Compute the Wordle color pattern for a guess against a target.
    Returns a tuple of 5 values: BLACK=0, YELLOW=1, GREEN=2.

    Handles duplicate letters correctly via two-pass approach:
      Pass 1 — mark exact matches (green), remove from target pool
      Pass 2 — mark present-but-wrong-position (yellow), consuming remaining letters
    """
    result = [BLACK] * WORD_LEN
    target_pool = list(target)

    for i in range(WORD_LEN):
        if guess[i] == target[i]:
            result[i] = GREEN
            target_pool[i] = None

    for i in range(WORD_LEN):
        if result[i] == GREEN:
            continue
        for j in range(WORD_LEN):
            if target_pool[j] == guess[i]:
                result[i] = YELLOW
                target_pool[j] = None
                break

    return tuple(result)


def pattern_to_int(pattern: tuple) -> int:
    """Convert (2,0,1,0,0) → base-3 integer."""
    return sum(p * 3**i for i, p in enumerate(pattern))


def int_to_pattern(n: int) -> tuple:
    """Convert base-3 integer → (B/Y/G, B/Y/G, ...)."""
    return tuple((n // 3**i) % 3 for i in range(WORD_LEN))


def pattern_to_emoji(pattern) -> str:
    if isinstance(pattern, int):
        pattern = int_to_pattern(pattern)
    return ''.join(['\u2B1B', '\U0001F7E8', '\U0001F7E9'][p] for p in pattern)


def pattern_to_label(pattern) -> str:
    if isinstance(pattern, int):
        pattern = int_to_pattern(pattern)
    return ''.join(['B', 'Y', 'G'][p] for p in pattern)


# ─── Fast pattern computation (for building the matrix) ─────────────────────

def _pattern_int(guess: str, target: str) -> int:
    """Pattern as base-3 int directly — faster for matrix construction."""
    g0, g1, g2, g3, g4 = guess
    t0, t1, t2, t3, t4 = target
    p = [0, 0, 0, 0, 0]
    rem = [t0, t1, t2, t3, t4]

    if g0 == t0: p[0] = 2; rem[0] = None
    if g1 == t1: p[1] = 2; rem[1] = None
    if g2 == t2: p[2] = 2; rem[2] = None
    if g3 == t3: p[3] = 2; rem[3] = None
    if g4 == t4: p[4] = 2; rem[4] = None

    for i in range(5):
        if p[i] == 2:
            continue
        c = guess[i]
        for j in range(5):
            if rem[j] == c:
                p[i] = 1
                rem[j] = None
                break

    return p[0] + 3*p[1] + 9*p[2] + 27*p[3] + 81*p[4]


# ─── Entropy and information ────────────────────────────────────────────────

def compute_entropy(guess: str, possible_words: list) -> float:
    """Expected bits of information from guessing `guess`."""
    counts = defaultdict(int)
    for target in possible_words:
        counts[get_pattern(guess, target)] += 1

    n = len(possible_words)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return entropy


def actual_information(n_before: int, n_after: int) -> float:
    """Bits of information actually conveyed: I = log2(n_before / n_after)."""
    if n_after == 0 or n_before == 0:
        return float('inf')
    return math.log2(n_before / n_after)


def filter_words(possible: list, guess: str, pattern: tuple) -> list:
    """Return words consistent with observing `pattern` for `guess`."""
    return [w for w in possible if get_pattern(guess, w) == pattern]


# ─── Ranking guesses (pure Python path) ─────────────────────────────────────

def best_guesses(possible: list, all_words: list, top_n: int = 5) -> list:
    """
    Rank words by expected information. Returns [(word, entropy, is_possible)].
    """
    if len(possible) <= 2:
        scored = [(w, compute_entropy(w, possible), True) for w in possible]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_n]

    possible_set = set(possible)
    scored = []
    for word in all_words:
        h = compute_entropy(word, possible)
        scored.append((word, h, word in possible_set))

    scored.sort(key=lambda x: (-x[1], not x[2], x[0]))
    return scored[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
#  PRECOMPUTED PATTERN MATRIX (numpy) — for fast simulation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_row_chunk(args):
    """Worker: compute pattern matrix rows for a chunk of guesses."""
    chunk_guesses, all_words = args
    rows = []
    for guess in chunk_guesses:
        rows.append([_pattern_int(guess, t) for t in all_words])
    return rows


def build_pattern_matrix(words: list, cache_path: str = "pattern_matrix.npy") -> np.ndarray:
    """
    Precompute pattern for every (guess, target) pair → uint8 array (N×N).
    Uses multiprocessing. Caches to disk (~160 MB for 12,972 words).
    """
    n = len(words)

    if os.path.exists(cache_path):
        print(f"  Loading cached pattern matrix from {cache_path}...")
        matrix = np.load(cache_path)
        if matrix.shape == (n, n):
            print(f"  Loaded {n}×{n} matrix.")
            return matrix
        print(f"  Cache shape mismatch ({matrix.shape} vs {(n,n)}), recomputing...")

    print(f"  Building {n}×{n} pattern matrix ({n*n:,} entries)...")
    print(f"  This is a one-time cost — result is cached to disk.")
    start = time.time()

    matrix = np.zeros((n, n), dtype=np.uint8)

    # Split into chunks for multiprocessing
    chunk_size = 100
    chunks = []
    for i in range(0, n, chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunks.append((chunk_words, words))

    done = 0
    with ProcessPoolExecutor() as executor:
        futures = {}
        for ci, chunk in enumerate(chunks):
            futures[executor.submit(_compute_row_chunk, chunk)] = ci

        for future in as_completed(futures):
            ci = futures[future]
            rows = future.result()
            start_row = ci * chunk_size
            for j, row in enumerate(rows):
                matrix[start_row + j] = row
            done += len(rows)
            if done % 1000 < chunk_size:
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (n - done) / rate if rate > 0 else 0
                print(f"\r  Progress: {done}/{n} ({100*done/n:.1f}%)"
                      f"  {rate:.0f} rows/s  ETA {eta:.0f}s", end="", flush=True)

    elapsed = time.time() - start
    print(f"\r  Built {n}×{n} matrix in {elapsed:.1f}s"
          f" ({n*n/elapsed:,.0f} patterns/s)              ")

    np.save(cache_path, matrix)
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"  Saved to {cache_path} ({size_mb:.1f} MB)")
    return matrix


def entropy_from_matrix(matrix: np.ndarray, guess_idx: int, target_mask: np.ndarray) -> float:
    """Compute entropy for one guess using the precomputed matrix. target_mask is a bool array."""
    patterns = matrix[guess_idx][target_mask]
    counts = np.bincount(patterns, minlength=NUM_PATTERNS)
    counts = counts[counts > 0]
    total = counts.sum()
    probs = counts / total
    return -float(np.sum(probs * np.log2(probs)))


def best_guesses_matrix(matrix: np.ndarray, target_mask: np.ndarray,
                        n_words: int, top_n: int = 5) -> list:
    """
    Rank ALL words by entropy using the precomputed pattern matrix.
    Returns [(index, entropy)] sorted by descending entropy.

    Key optimisation: iterate over the n_possible *columns* (remaining targets)
    rather than the n_words *rows*. When the possible set is small (which it is
    from guess 2 onward), this is hundreds of times faster than the naive loop.

    Shape: matrix (N, N), sub (N, n_possible).
    Counting: accumulate into a (N, 243) count array, one column at a time.
    Then compute entropy fully vectorised with numpy.
    """
    sub = matrix[:, target_mask]           # (n_words, n_possible)
    n_possible = sub.shape[1]

    if n_possible <= 1:
        top_indices = np.arange(min(top_n, n_words))
        return [(int(i), 0.0) for i in top_indices]

    # Build count matrix: counts[i, p] = how many targets give pattern p for guess i
    # Iterate over columns (possible targets) — O(n_possible) Python iterations
    counts = np.zeros((n_words, NUM_PATTERNS), dtype=np.int32)
    row_idx = np.arange(n_words)
    for j in range(n_possible):
        counts[row_idx, sub[:, j]] += 1

    # Vectorised entropy: H = -Σ p log2(p)
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = counts.astype(np.float64) / n_possible
        log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    entropy = -np.einsum('ij,ij->i', probs, log_probs)

    top_k = min(top_n, n_words)
    top_indices = np.argpartition(-entropy, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-entropy[top_indices])]
    return [(int(idx), float(entropy[idx])) for idx in top_indices]


# ─── Game simulation with matrix ────────────────────────────────────────────

def play_game(target: str, all_words: list, first_guess: str = None,
              max_guesses: int = 6) -> dict:
    """Simulate a single game (pure Python path)."""
    possible = list(all_words)
    guesses = []

    current_guess = first_guess or best_guesses(possible, all_words, top_n=1)[0][0]

    for attempt in range(1, max_guesses + 1):
        pattern = get_pattern(current_guess, target)
        n_before = len(possible)
        expected = compute_entropy(current_guess, possible)
        possible = filter_words(possible, current_guess, pattern)
        n_after = len(possible)

        guesses.append({
            'guess': current_guess,
            'pattern': pattern,
            'emoji': pattern_to_emoji(pattern),
            'n_before': n_before,
            'n_after': n_after,
            'expected_info': expected,
            'actual_info': actual_information(n_before, n_after),
        })

        if pattern == ALL_GREEN:
            return {'target': target, 'solved': True, 'attempts': attempt, 'guesses': guesses}

        if not possible:
            return {'target': target, 'solved': False, 'attempts': attempt, 'guesses': guesses}

        guesses_left = max_guesses - attempt
        n_after = len(possible)
        if n_after == 1 or guesses_left <= 1:
            current_guess = possible[0]
        elif n_after <= guesses_left:
            top_word = best_guesses(possible, all_words, top_n=1)[0][0]
            current_guess = top_word if top_word in set(possible) else possible[0]
        else:
            current_guess = best_guesses(possible, all_words, top_n=1)[0][0]

    return {'target': target, 'solved': False, 'attempts': max_guesses + 1, 'guesses': guesses}


def play_game_matrix(target_idx: int, matrix: np.ndarray, all_words: list,
                     first_guess_idx: int, max_guesses: int = 6) -> dict:
    """Simulate a single game using the precomputed pattern matrix (fast)."""
    n = len(all_words)
    target = all_words[target_idx]
    # Boolean mask of which words are still possible
    possible_mask = np.ones(n, dtype=bool)
    guesses = []

    guess_idx = first_guess_idx

    for attempt in range(1, max_guesses + 1):
        guess = all_words[guess_idx]

        # Get the pattern
        pattern_int = int(matrix[guess_idx, target_idx])
        pattern_tuple = int_to_pattern(pattern_int)

        n_before = int(possible_mask.sum())

        # Compute expected entropy before filtering
        expected = entropy_from_matrix(matrix, guess_idx, possible_mask)

        # Filter: keep only words that produce the same pattern against this guess
        # i.e., for each possible word w, matrix[guess_idx, w] == pattern_int
        possible_mask &= (matrix[guess_idx] == pattern_int)
        n_after = int(possible_mask.sum())

        guesses.append({
            'guess': guess,
            'pattern': pattern_tuple,
            'emoji': pattern_to_emoji(pattern_tuple),
            'n_before': n_before,
            'n_after': n_after,
            'expected_info': expected,
            'actual_info': actual_information(n_before, n_after),
        })

        if pattern_int == ALL_GREEN_INT:
            return {'target': target, 'solved': True, 'attempts': attempt, 'guesses': guesses}

        if n_after == 0:
            return {'target': target, 'solved': False, 'attempts': attempt, 'guesses': guesses}

        guesses_left = max_guesses - attempt
        if n_after == 1 or guesses_left <= 1:
            # Only one option or last guess: must pick from remaining answers
            guess_idx = int(np.where(possible_mask)[0][0])
        elif n_after <= guesses_left:
            # We have enough guesses to enumerate possible words directly.
            # Guessing from possible set guarantees a win; a non-answer guess
            # risks wasting a turn when the possible set is this small.
            top = best_guesses_matrix(matrix, possible_mask, n, top_n=1)
            if possible_mask[top[0][0]]:
                guess_idx = top[0][0]   # top pick is already a possible answer
            else:
                guess_idx = int(np.where(possible_mask)[0][0])
        else:
            # Pick the highest-entropy guess from ALL words (not just remaining)
            top = best_guesses_matrix(matrix, possible_mask, n, top_n=1)
            guess_idx = top[0][0]

    return {'target': target, 'solved': False, 'attempts': max_guesses + 1, 'guesses': guesses}


# ─── Full simulation ────────────────────────────────────────────────────────

def run_simulation(all_words: list, first_guess: str = None,
                   max_guesses: int = 6, verbose: bool = True,
                   targets: list = None) -> list:
    """
    Simulate the solver on all words (or a subset) using the precomputed matrix.
    """
    from collections import Counter

    n = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Build or load pattern matrix
    matrix = build_pattern_matrix(all_words)

    # Determine first guess
    if first_guess is None:
        if verbose:
            print("  Computing optimal first guess...")
        top = best_guesses_matrix(matrix, np.ones(n, dtype=bool), n, top_n=5)
        first_guess_idx = top[0][0]
        first_guess = all_words[first_guess_idx]
        if verbose:
            print(f"  Optimal first guess: {first_guess.upper()} ({top[0][1]:.4f} bits)")
            print(f"  Top 5 openers:")
            for idx, ent in top:
                print(f"    {all_words[idx].upper():<10} {ent:.4f} bits")
    else:
        first_guess = first_guess.lower()
        if first_guess not in word_to_idx:
            print(f"  Warning: '{first_guess}' not in word list")
            return []
        first_guess_idx = word_to_idx[first_guess]

    # Determine target words
    if targets is not None:
        target_indices = [word_to_idx[t] for t in targets if t in word_to_idx]
    else:
        target_indices = list(range(n))

    total = len(target_indices)
    if verbose:
        print(f"\n  Simulating {total:,} games | First guess: {first_guess.upper()}")
        print(f"  {'─'*55}")

    results = []
    dist = Counter()
    total_guesses = 0
    start = time.time()

    for i, tidx in enumerate(target_indices):
        result = play_game_matrix(tidx, matrix, all_words, first_guess_idx, max_guesses)
        results.append(result)
        dist[result['attempts']] += 1
        total_guesses += result['attempts']

        if verbose and ((i + 1) % 500 == 0 or i == 0):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            avg = total_guesses / (i + 1)
            print(f"\r  [{i+1:>5}/{total}] {100*(i+1)/total:5.1f}%"
                  f"  avg={avg:.3f}  {rate:.1f} games/s  ETA {eta:.0f}s",
                  end="", flush=True)

    elapsed = time.time() - start

    if verbose:
        solved = [r for r in results if r['solved']]
        avg = sum(r['attempts'] for r in solved) / len(solved) if solved else 0

        print(f"\r  {'─'*55}{'':>20}")
        print(f"\n{'═'*55}")
        print(f"  SIMULATION RESULTS  (first guess: {first_guess.upper()})")
        print(f"{'═'*55}")
        print(f"  Total games    : {total:,}")
        print(f"  Solved (≤{max_guesses})    : {len(solved)}  ({100*len(solved)/total:.2f}%)")
        print(f"  Failed (>{max_guesses})     : {total - len(solved)}")
        print(f"  Average guesses: {avg:.4f}")
        print(f"  Time           : {elapsed:.1f}s ({total/elapsed:.1f} games/s)")
        print()

        max_count = max(dist.values()) if dist else 1
        print(f"  Guess distribution:")
        for k in sorted(dist):
            bar_len = max(1, dist[k] * 50 // max_count)
            bar = '█' * bar_len
            label = f"  {k}" if k <= max_guesses else f"  {max_guesses}+"
            print(f"  {label} │ {bar}  {dist[k]:>5} ({100*dist[k]/total:.1f}%)")

        failed = [r for r in results if not r['solved']]
        if failed:
            print(f"\n  Hardest words (failed or 6 guesses):")
            for r in sorted(failed, key=lambda x: -x['attempts'])[:20]:
                chain = ' → '.join(g['guess'].upper() for g in r['guesses'])
                print(f"    {r['target'].upper()} ({r['attempts']} guesses): {chain}")

    return results
