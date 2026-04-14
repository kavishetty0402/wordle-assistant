# Wordle Assistant

An entropy-based Wordle solver with both a command-line interface and a **Streamlit web app**.

It uses information theory (Shannon entropy) to recommend the optimal guess at every step — the word that maximises the expected reduction in the remaining search space.

---

## How it works

For a guess **G** and the set of remaining possible words **P**, the expected information is:

```
E[I] = -Σ p(pattern) · log₂(p(pattern))
```

where `p(pattern) = |words producing that pattern| / |P|`.

When the actual pattern is observed, the information gained is:

```
I = log₂(n_before / n_after)
```

A precomputed **pattern matrix** (cached to disk) stores the colour pattern for every `(guess, target)` pair, making lookups and entropy calculations near-instant.

---

## Getting started

### 1. Install dependencies

```bash
pip install streamlit numpy pandas
```

### 2. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The pattern matrix is built once on first run and cached automatically.

---

## Streamlit app features

| Feature | Detail |
|---|---|
| Best-guess table | Top 10 words ranked by expected information (bits) |
| Colour input | Per-letter drop-downs (Black / Yellow / Green) |
| Live word count | Sidebar shows how many words are still possible |
| Information analysis | Expected vs. actual info, cumulative total, word reduction |
| Guess history | Full record of every guess and its entropy stats |
| Auto-recommendation | Pre-fills the highest-entropy word for you each turn |

---

## Command-line interface

```bash
# Interactive assistant (guides you through a real game)
python app.py

# Simulate solving for a specific word
python app.py --word crane

# Simulate the solver across all words and print stats
python app.py --simulate

# Simulate on a random sample (faster)
python app.py --simulate --sample 500

# Find the best opening words
python app.py --best-opener

# Deep analysis of a guess word
python app.py --analyze slate
```

**Pattern input format** (after each guess):
```
G = green  (right letter, right position)
Y = yellow (right letter, wrong position)
B = black  (letter not in word)
```

---

## Project structure

```
wordle_solver.py   — Core solver: pattern computation, entropy, matrix, simulation
app.py             — Command-line interface
streamlit_app.py   — Streamlit web app
words.txt          — Full word list (~12 k five-letter words)
answers.txt        — Curated answer word list
pattern_matrix.npy — Cached precomputed pattern matrix (auto-generated)
fetch_words.py     — Script to (re)download the word list
```

---

## Performance

| Metric | Value |
|---|---|
| Word list | ~12,972 words |
| Pattern matrix | ~12k × 12k = ~168 M entries (≈ 160 MB on disk) |
| Matrix build time | ~60 s (one-time, multiprocessing) |
| Entropy lookup | < 1 ms per guess (matrix path) |
| Typical solve rate | > 99 % within 6 guesses |
| Average guesses | ~3.5 |
