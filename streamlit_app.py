"""
Wordle Assistant — Streamlit UI
--------------------------------
Run with:  streamlit run streamlit_app.py
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from wordle_solver import (
    ALL_GREEN, BLACK, GREEN, WORD_LEN, YELLOW,
    actual_information, build_pattern_matrix, best_guesses_matrix,
    compute_entropy, entropy_from_matrix, get_pattern,
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


def ensure_state(all_words):
    if "possible" not in st.session_state:
        reset_game(all_words)


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


def render_best_guesses_table(ranked):
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


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.title("🟩 Wordle Assistant")
    st.caption("Entropy-based solver — guides you to the information-optimal guess every turn.")

    all_words, matrix, word_to_idx = load_resources()
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

        if st.button("🔄 Reset game", use_container_width=True):
            reset_game(all_words)
            st.rerun()

        st.divider()
        if 0 < n_possible <= 30:
            st.subheader(f"Remaining words ({n_possible})")
            st.write(", ".join(w.upper() for w in sorted(st.session_state.possible)))
        elif n_possible > 30:
            st.caption(f"{n_possible:,} words still possible.")

    # ── Compute best guesses for the current state ────────────────────────────
    with st.spinner("Computing best guesses…"):
        ranked = top_guesses(matrix, word_to_idx, all_words, st.session_state.possible, top_n=10)
    recommended = ranked[0][0] if ranked else ""

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        render_history()

    with right:
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
                "⚠️ No words match this pattern — double-check the colours you entered."
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
