"""
GhostKey - EDA & ETL (First Review deliverable)

ETL pipeline:
  EXTRACT   Brown corpus (57k sentences) | TamilMixSentiment corpus
            (15,744 code-mixed comments) | self-generated error pairs
  TRANSFORM lowercase + alphabetic filter + length filter | QWERTY
            error simulation (aligned wrong->correct pairs) | vowel &
            frequency guards on mined Tanglish vocabulary
  LOAD      language model (freq + bigrams) | tanglish_mined.json |
            train/test datasets for the correction models

EDA outputs (charts/ folder):
  1. error_types.png        - distribution of simulated error types
  2. keyboard_heatmap.png   - which keys get confused with which
  3. length_vs_error.png    - word length vs corruption rate
  4. corpus_stats.png       - dataset sizes across the three sources
  5. tanglish_top.png       - top mined Tanglish words (if mined)

Run:  python eda.py
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import build_dataset, ADJACENT

os.makedirs("charts", exist_ok=True)
FIG = dict(figsize=(8, 4.5), dpi=110)


def chart_error_types(data):
    types = Counter(e[2] for d in data for e in d["errors"])
    plt.figure(**FIG)
    plt.bar(types.keys(), types.values(), color="#3d7bfd")
    plt.title("Simulated eyes-free typing: error type distribution")
    plt.ylabel("count")
    plt.tight_layout(); plt.savefig("charts/error_types.png"); plt.close()
    return types


def chart_keyboard_heatmap(data):
    """Confusion counts: intended letter -> typed letter (substitutions)."""
    letters = "qwertyuiopasdfghjklzxcvbnm"
    idx = {c: i for i, c in enumerate(letters)}
    grid = [[0] * 26 for _ in range(26)]
    for d in data:
        for clean, bad, etype in d["errors"]:
            if etype == "substitute" and len(clean) == len(bad):
                for a, b in zip(clean, bad):
                    if a != b and a in idx and b in idx:
                        grid[idx[a]][idx[b]] += 1
    plt.figure(figsize=(7.5, 6.5), dpi=110)
    plt.imshow(grid, cmap="Blues")
    plt.xticks(range(26), letters); plt.yticks(range(26), letters)
    plt.xlabel("typed key"); plt.ylabel("intended key")
    plt.title("Keyboard confusion heatmap (QWERTY adjacency)")
    plt.colorbar(shrink=0.8)
    plt.tight_layout(); plt.savefig("charts/keyboard_heatmap.png"); plt.close()


def chart_length_vs_error(data):
    total, wrong = Counter(), Counter()
    for d in data:
        for cw, bw in zip(d["clean"], d["corrupt"]):
            total[len(cw)] += 1
            if cw != bw:
                wrong[len(cw)] += 1
    lengths = sorted(l for l in total if 2 <= l <= 12 and total[l] >= 30)
    rates = [100 * wrong[l] / total[l] for l in lengths]
    plt.figure(**FIG)
    plt.plot(lengths, rates, "o-", color="#e0483e")
    plt.title("Word length vs corruption rate")
    plt.xlabel("word length"); plt.ylabel("% words corrupted")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("charts/length_vs_error.png"); plt.close()


def chart_corpus_stats(data):
    from nltk.corpus import brown
    n_brown = len(brown.sents())
    n_pairs = len(data)
    n_tanglish = 0
    if os.path.exists("tanglish_mined.json"):
        import json
        n_tanglish = json.load(open("tanglish_mined.json"))["comments_mined"]
    names = ["Brown corpus\n(sentences)", "Simulated pairs\n(generated)",
             "TamilMixSentiment\n(comments)"]
    vals = [n_brown, n_pairs, n_tanglish]
    plt.figure(**FIG)
    bars = plt.bar(names, vals, color=["#3d7bfd", "#43a047", "#e0483e"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:,}",
                 ha="center", va="bottom")
    plt.title("ETL: the three data sources of GhostKey")
    plt.ylabel("size")
    plt.tight_layout(); plt.savefig("charts/corpus_stats.png"); plt.close()


def chart_tanglish_top():
    if not os.path.exists("tanglish_mined.json"):
        print("  (tanglish_mined.json not found - run mine_tanglish.py "
              "first for the Tanglish chart)")
        return
    import json
    vocab = json.load(open("tanglish_mined.json"))["vocab"]
    top = Counter(vocab).most_common(15)
    plt.figure(**FIG)
    plt.barh([w for w, _ in reversed(top)],
             [c for _, c in reversed(top)], color="#8e24aa")
    plt.title("Top mined Tanglish words (TamilMixSentiment)")
    plt.xlabel("corpus frequency")
    plt.tight_layout(); plt.savefig("charts/tanglish_top.png"); plt.close()


if __name__ == "__main__":
    print("[EDA] building dataset (EXTRACT + TRANSFORM)...")
    data = build_dataset(n_sentences=2000, error_prob=0.5)

    n_words = sum(len(d["clean"]) for d in data)
    n_errors = sum(len(d["errors"]) for d in data)
    print(f"[EDA] {len(data):,} sentence pairs | {n_words:,} words | "
          f"{n_errors:,} injected errors ({100*n_errors/n_words:.1f}%)")

    types = chart_error_types(data)
    print("[EDA] error types:", dict(types))
    chart_keyboard_heatmap(data)
    chart_length_vs_error(data)
    chart_corpus_stats(data)
    chart_tanglish_top()
    print("[EDA] charts saved to charts/ - open them for the review slides")
