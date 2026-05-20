"""
Wordle Assistant — Streamlit UI
--------------------------------
Run with:  streamlit run streamlit_app.py
"""

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from wordle_solver import (
    ALL_GREEN, BLACK, GREEN, WORD_LEN, YELLOW,
    actual_information, build_pattern_matrix, best_guesses_matrix,
    compute_entropy, entropy_from_matrix, filter_words, get_pattern,
    pattern_to_emoji,
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wordle Assistant",
    page_icon="🟩",
    layout="wide",
)

# ─── Resource loading (cached across reruns) ─────────────────────────────────

@st.cache_resource(show_spinner="Loading word list and pattern matrix…")
def load_resources():
    with open("words.txt") as f:
        all_words = [ln.strip().lower() for ln in f]
    all_words = [w for w in all_words if len(w) == WORD_LEN and w.isalpha()]
    matrix = build_pattern_matrix(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    return all_words, matrix, word_to_idx


@st.cache_resource(show_spinner="Loading answer list…")
def load_answers():
    """Load the official Wordle answer list and compute word commonality scores."""
    with open("answers.txt") as f:
        answers = [ln.strip().lower() for ln in f]
    answers = [w for w in answers if len(w) == WORD_LEN and w.isalpha()]
    answer_set = set(answers)

    # Compute letter frequency across the answer list (positional)
    # Words with common letters in common positions score higher
    pos_freq = [Counter() for _ in range(WORD_LEN)]
    for w in answers:
        for i, ch in enumerate(w):
            pos_freq[i][ch] += 1

    # Normalise: max frequency per position = 1.0
    for i in range(WORD_LEN):
        max_count = max(pos_freq[i].values()) if pos_freq[i] else 1
        for ch in pos_freq[i]:
            pos_freq[i][ch] /= max_count

    # Commonality score = sum of positional frequencies (0–5 range)
    # Bonus for using all distinct letters (more "common" feel)
    commonality = {}
    for w in answers:
        score = sum(pos_freq[i].get(ch, 0) for i, ch in enumerate(w))
        # Small bonus for distinct letters (common words rarely repeat letters)
        distinct_ratio = len(set(w)) / WORD_LEN
        score *= (0.8 + 0.2 * distinct_ratio)
        commonality[w] = score

    return answers, answer_set, commonality


# ─── Hard mode constraint logic ─────────────────────────────────────────────

def derive_hard_mode_constraints(history):
    """
    From the guess history, derive the hard mode constraints:
    - greens: dict of {position: letter} — must appear at this exact position
    - yellows: list of (letter, position) — letter must be in word but NOT at this position
    - blacks: set of letters — must NOT appear in the word at all
              (unless the same letter also appears as green/yellow elsewhere)
    """
    greens = {}         # pos → letter (must be here)
    yellows = []        # [(letter, pos_to_avoid), ...]  — must be in word, not at pos
    yellow_letters = set()
    green_letters = set()
    black_candidates = set()

    for entry in history:
        guess = entry["guess"]
        pattern = entry["pattern"]
        for i, (ch, color) in enumerate(zip(guess, pattern)):
            if color == GREEN:
                greens[i] = ch
                green_letters.add(ch)
            elif color == YELLOW:
                yellows.append((ch, i))
                yellow_letters.add(ch)
            else:  # BLACK
                black_candidates.add(ch)

    # A letter marked black in one position might be green/yellow in another
    # (duplicate letter handling) — only truly exclude if never green/yellow
    confirmed_letters = green_letters | yellow_letters
    blacks = black_candidates - confirmed_letters

    return greens, yellows, blacks


def hard_mode_filter(words, history):
    """
    Filter words to only those satisfying hard mode constraints.
    - Green letters must be at their exact positions
    - Yellow letters must appear in the word (but not at the position they were yellow)
    - Black letters must not appear (unless also green/yellow elsewhere)
    """
    if not history:
        return words  # No constraints on the first guess

    greens, yellows, blacks = derive_hard_mode_constraints(history)

    # Collect required yellow letters (must appear somewhere in the word)
    required_yellows = defaultdict(set)  # letter → set of forbidden positions
    for letter, pos in yellows:
        required_yellows[letter].add(pos)

    filtered = []
    for w in words:
        valid = True

        # Check greens: letter at position must match
        for pos, letter in greens.items():
            if w[pos] != letter:
                valid = False
                break
        if not valid:
            continue

        # Check blacks: letter must not appear at all
        for ch in blacks:
            if ch in w:
                valid = False
                break
        if not valid:
            continue

        # Check yellows: letter must be in word but NOT at the forbidden position
        for letter, forbidden_positions in required_yellows.items():
            if letter not in w:
                valid = False
                break
            # Also ensure the letter isn't ONLY at forbidden positions
            # (it must appear at a non-forbidden position)
            found_valid_pos = False
            for i, ch in enumerate(w):
                if ch == letter and i not in forbidden_positions:
                    found_valid_pos = True
                    break
            # If the letter is green somewhere, that counts
            if not found_valid_pos:
                for pos, gch in greens.items():
                    if gch == letter:
                        found_valid_pos = True
                        break
            if not found_valid_pos:
                valid = False
                break
        if not valid:
            continue

        filtered.append(w)

    return filtered


# ─── Solver helpers ───────────────────────────────────────────────────────────

def top_guesses(matrix, word_to_idx, all_words, possible, top_n=10):
    n = len(all_words)
    possible_set = set(possible)
    mask = np.zeros(n, dtype=bool)
    for w in possible:
        if w in word_to_idx:
            mask[word_to_idx[w]] = True
    ranked = best_guesses_matrix(matrix, mask, n, top_n=top_n)
    return [(all_words[idx], ent, all_words[idx] in possible_set) for idx, ent in ranked]


def top_guesses_hard_mode(matrix, word_to_idx, all_words, possible,
                          valid_guesses, commonality, top_n=10):
    """
    Rank only the valid_guesses (hard-mode-compatible answer words) by entropy,
    using word commonality as a tiebreaker.
    """
    n = len(all_words)
    possible_set = set(possible)

    # Build target mask (which words are still possible answers)
    target_mask = np.zeros(n, dtype=bool)
    for w in possible:
        if w in word_to_idx:
            target_mask[word_to_idx[w]] = True

    n_possible = int(target_mask.sum())
    if n_possible <= 1:
        # Only 0 or 1 word left — entropy is 0 for everything
        results = []
        for w in valid_guesses[:top_n]:
            results.append((w, 0.0, w in possible_set, commonality.get(w, 0)))
        return results

    # Compute entropy for each valid guess
    scored = []
    for w in valid_guesses:
        if w in word_to_idx:
            ent = entropy_from_matrix(matrix, word_to_idx[w], target_mask)
        else:
            ent = compute_entropy(w, possible)
        scored.append((w, ent, w in possible_set, commonality.get(w, 0)))

    # Sort by: entropy desc, then commonality desc (prefer common words as tiebreaker)
    scored.sort(key=lambda x: (-x[1], -x[3]))
    return scored[:top_n]


def entropy_for_guess(matrix, word_to_idx, all_words, guess, possible):
    if guess not in word_to_idx:
        return compute_entropy(guess, possible)
    n = len(all_words)
    mask = np.zeros(n, dtype=bool)
    for w in possible:
        if w in word_to_idx:
            mask[word_to_idx[w]] = True
    return entropy_from_matrix(matrix, word_to_idx[guess], mask)


# ─── Session state helpers ────────────────────────────────────────────────────

def reset_game(all_words):
    st.session_state.possible = list(all_words)
    st.session_state.history = []
    st.session_state.total_info = 0.0
    st.session_state.game_over = False
    st.session_state.solved = False
    st.session_state.attempt = 1


def undo_last_guess(all_words):
    if not st.session_state.history:
        return
    st.session_state.history.pop()
    st.session_state.attempt -= 1
    st.session_state.game_over = False
    st.session_state.solved = False
    # Re-derive possible words by replaying remaining history
    possible = list(all_words)
    total_info = 0.0
    for entry in st.session_state.history:
        possible = filter_words(possible, entry["guess"], entry["pattern"])
        total_info += entry["actual"]
    st.session_state.possible = possible
    st.session_state.total_info = total_info


def ensure_state(all_words):
    if "possible" not in st.session_state:
        reset_game(all_words)
    if "hard_mode" not in st.session_state:
        st.session_state.hard_mode = False


# ─── UI components ────────────────────────────────────────────────────────────

TILE_COLORS = {
    BLACK:  ("#787c7e", "⬛"),
    YELLOW: ("#c9b458", "🟨"),
    GREEN:  ("#6aaa64", "🟩"),
}

COLOR_OPTIONS = ["⬛ Black", "🟨 Yellow", "🟩 Green"]
COLOR_TO_INT   = {"⬛ Black": BLACK, "🟨 Yellow": YELLOW, "🟩 Green": GREEN}


def render_history():
    if not st.session_state.history:
        return
    st.subheader("Guess history")
    for entry in st.session_state.history:
        emoji_row = entry["emoji"]
        word      = entry["guess"].upper()
        delta     = entry["actual"] - entry["expected"]
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"

        with st.container():
            cols = st.columns([2, 3, 3, 3])
            cols[0].markdown(f"### {word}")
            cols[1].markdown(f"### {emoji_row}")
            cols[2].metric("E[I]", f"{entry['expected']:.3f} bits")
            cols[3].metric(
                "Actual I",
                f"{entry['actual']:.3f} bits",
                delta=delta_str,
                delta_color="normal",
            )
            st.caption(
                f"Words: {entry['n_before']} → {entry['n_after']}  |  "
                f"p = {entry['n_after']}/{entry['n_before']} = "
                f"{entry['n_after']/entry['n_before']:.5f}"
            )
        st.divider()


def render_best_guesses_table(ranked, hard_mode=False):
    if hard_mode:
        rows = [
            {
                "Word": w.upper(),
                "E[info] (bits)": round(e, 4),
                "Possible": "✓" if p else "",
                "Commonality": round(c, 2),
            }
            for w, e, p, c in ranked
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Word": st.column_config.TextColumn(width="small"),
                "E[info] (bits)": st.column_config.NumberColumn(format="%.4f"),
                "Possible": st.column_config.TextColumn(width="small"),
                "Commonality": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        rows = [
            {
                "Word": w.upper(),
                "Expected info (bits)": round(e, 4),
                "Answer set": "✓" if p else "",
            }
            for w, e, p in ranked
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Word": st.column_config.TextColumn(width="small"),
                "Expected info (bits)": st.column_config.NumberColumn(format="%.4f"),
                "Answer set": st.column_config.TextColumn(width="small"),
            },
        )


def render_possible_answers(possible, commonality=None):
    """Show a clear list of all remaining candidate words when the set is small."""
    n = len(possible)
    st.subheader(f"🎯 Possible answers ({n})")
    st.caption(
        "The remaining words are few enough that entropy-based ranking isn't very helpful — "
        "any guess from this list will work. Pick whichever word you recognise!"
    )
    # Sort by commonality if available, otherwise alphabetical
    if commonality:
        sorted_words = sorted(possible, key=lambda w: -commonality.get(w, 0))
    else:
        sorted_words = sorted(possible)

    # Display as a neat grid of word chips
    cols_per_row = 5
    for i in range(0, n, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < n:
                col.markdown(
                    f"<div style='background:#1a1a2e; border:1px solid #6aaa64; "
                    f"border-radius:8px; padding:10px 6px; text-align:center; "
                    f"font-weight:700; font-size:1.1em; letter-spacing:2px; "
                    f"color:#6aaa64;'>{sorted_words[idx].upper()}</div>",
                    unsafe_allow_html=True,
                )
    st.write("")


def render_hard_mode_guesses(ranked_hard):
    """Show the hard-mode-compatible guesses with entropy + commonality."""
    if not ranked_hard:
        st.warning("No valid guesses found for hard mode constraints!")
        return

    st.subheader("🔒 Hard mode guesses")
    st.caption(
        "Only answer-list words that satisfy all revealed constraints. "
        "Ranked by entropy, with more common words preferred as tiebreakers."
    )
    render_best_guesses_table(ranked_hard, hard_mode=True)


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.title("🟩 Wordle Assistant")
    st.caption("Entropy-based solver — guides you to the information-optimal guess every turn.")

    all_words, matrix, word_to_idx = load_resources()
    answers, answer_set, commonality = load_answers()
    n_total  = len(all_words)
    max_info = math.log2(n_total)

    ensure_state(all_words)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Game state")
        n_possible = len(st.session_state.possible)

        st.metric("Words remaining", f"{n_possible:,}")
        st.metric("Attempt", f"{st.session_state.attempt} / 6")
        if st.session_state.total_info > 0:
            pct = 100 * st.session_state.total_info / max_info
            st.metric(
                "Cumulative info",
                f"{st.session_state.total_info:.3f} bits",
                help=f"{pct:.1f}% of the theoretical max ({max_info:.2f} bits)",
            )

        st.divider()

        # Hard mode toggle
        hard_mode = st.toggle(
            "🔒 Hard mode",
            value=st.session_state.hard_mode,
            help=(
                "In hard mode, every guess must be from the official answer list "
                "and must satisfy all revealed constraints (green letters stay, "
                "yellow letters must be reused, black letters excluded). "
                "Words are ranked by entropy with common words preferred as tiebreakers."
            ),
        )
        if hard_mode != st.session_state.hard_mode:
            st.session_state.hard_mode = hard_mode
            st.rerun()

        if st.session_state.hard_mode:
            # Show how many answer words are valid under constraints
            valid_hm = hard_mode_filter(answers, st.session_state.history)
            st.caption(f"🔒 {len(valid_hm):,} answer words pass hard mode constraints")

        st.divider()

        if st.button("🔄 Reset game", use_container_width=True):
            reset_game(all_words)
            st.rerun()

        if st.session_state.history and st.button("↩ Undo last guess", use_container_width=True):
            undo_last_guess(all_words)
            st.rerun()

        st.divider()
        if 0 < n_possible <= 30:
            st.subheader(f"Remaining words ({n_possible})")
            st.write(", ".join(w.upper() for w in sorted(st.session_state.possible)))
        elif n_possible > 30:
            st.caption(f"{n_possible:,} words still possible.")

    # ── Compute best guesses for the current state ────────────────────────────
    is_hard = st.session_state.hard_mode
    ranked_hard = None

    with st.spinner("Computing best guesses…"):
        ranked = top_guesses(matrix, word_to_idx, all_words, st.session_state.possible, top_n=10)

        if is_hard:
            valid_hm_guesses = hard_mode_filter(answers, st.session_state.history)
            # Also intersect with possible answers (can't guess a word that's been eliminated)
            # Actually no — in Wordle hard mode you CAN guess any valid word, it just must
            # satisfy the constraints. The answer might not be in your guess.
            # But the user said "you HAVE to use words strictly from the answer list",
            # and the constraints are about letter reuse, not about the word being possible.
            ranked_hard = top_guesses_hard_mode(
                matrix, word_to_idx, all_words, st.session_state.possible,
                valid_hm_guesses, commonality, top_n=15,
            )

    # Determine recommended word
    n_possible = len(st.session_state.possible)
    if is_hard and ranked_hard:
        # In hard mode, recommend the top hard-mode-compatible word
        recommended = ranked_hard[0][0]
    elif 1 < n_possible <= 10:
        # When few candidates remain, recommend from the possible set directly
        recommended = sorted(st.session_state.possible)[0]
    else:
        recommended = ranked[0][0] if ranked else ""

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        render_history()

    with right:
        n_possible = len(st.session_state.possible)

        if is_hard and ranked_hard is not None:
            # Hard mode: show the constrained list
            if 1 < n_possible <= 10:
                render_possible_answers(st.session_state.possible, commonality)
                st.divider()
            render_hard_mode_guesses(ranked_hard)
            with st.expander("Full entropy rankings (all words, for reference)"):
                render_best_guesses_table(ranked)
        else:
            # Normal mode
            if 1 < n_possible <= 10:
                render_possible_answers(st.session_state.possible)
                with st.expander("Entropy rankings (for reference)"):
                    render_best_guesses_table(ranked)
            else:
                st.subheader("Best next guesses")
                render_best_guesses_table(ranked)

    # ── Game-over banner ──────────────────────────────────────────────────────
    if st.session_state.game_over:
        attempt = st.session_state.attempt - 1
        if st.session_state.solved:
            st.success(f"★ Solved in {attempt} guess{'es' if attempt != 1 else ''}!")
        else:
            remaining = ", ".join(w.upper() for w in st.session_state.possible)
            st.error(
                f"✗ Could not solve in 6 guesses.  "
                f"Remaining candidates: {remaining or 'none'}"
            )
        if st.button("Play again"):
            reset_game(all_words)
            st.rerun()
        return

    # ── Input form ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Guess #{st.session_state.attempt}")

    with st.form("guess_form", clear_on_submit=False):
        guess_input = st.text_input(
            f"Word to guess  (recommended: **{recommended.upper()}**)",
            value=recommended.upper(),
            max_chars=5,
            placeholder="5-letter word",
        ).strip().lower()

        st.write("Pattern — set each letter's colour:")
        pat_cols = st.columns(5)
        pattern_vals = []
        for i, col in enumerate(pat_cols):
            letter = guess_input[i].upper() if len(guess_input) > i else f"#{i+1}"
            choice = col.selectbox(letter, COLOR_OPTIONS, key=f"col_{i}", label_visibility="visible")
            pattern_vals.append(COLOR_TO_INT[choice])

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

    if submitted:
        guess = guess_input

        if len(guess) != WORD_LEN or not guess.isalpha():
            st.error("Please enter a valid 5-letter word.")
            st.stop()

        pattern   = tuple(pattern_vals)
        possible  = st.session_state.possible
        n_before  = len(possible)
        expected  = entropy_for_guess(matrix, word_to_idx, all_words, guess, possible)

        # Partition remaining words by pattern
        pattern_dist = defaultdict(list)
        for w in possible:
            pattern_dist[get_pattern(guess, w)].append(w)

        bucket  = pattern_dist.get(pattern, [])
        n_after = len(bucket)

        if n_after == 0:
            st.error(
                "⚠️ No words match this pattern.\n\n"
                "**Common cause with duplicate letters:** if the target has the same letter "
                "at the same position as your guess (e.g. E at position 4 in both TARES and ENTER), "
                "Wordle shows it as **green**, not yellow. "
                "Use the **↩ Undo last guess** button in the sidebar to re-enter any guess with the correct colours."
            )
            st.stop()

        info = actual_information(n_before, n_after)
        st.session_state.total_info += info

        st.session_state.history.append({
            "guess":    guess,
            "pattern":  pattern,
            "emoji":    pattern_to_emoji(pattern),
            "expected": expected,
            "actual":   info,
            "n_before": n_before,
            "n_after":  n_after,
        })

        st.session_state.possible = bucket
        st.session_state.attempt += 1

        if pattern == ALL_GREEN:
            st.session_state.game_over = True
            st.session_state.solved    = True
        elif st.session_state.attempt > 6:
            st.session_state.game_over = True
            st.session_state.solved    = False

        st.rerun()


if __name__ == "__main__":
    main()
