"""
GhostKey - Tanglish Vocabulary Miner
Mines the TamilMixSentiment corpus (Chakravarthi et al., 2020;
15,744 code-mixed Tamil-English YouTube comments) to build:

  1. a large Tanglish vocabulary  (words that are frequent in the corpus
     but are NOT common English words)
  2. real code-mixed bigrams      (for context-aware Tanglish correction)

Output: tanglish_mined.json  - loaded automatically by tanglish.py.

Run once (needs internet the first time):
    python mine_tanglish.py
"""

import json
import re
from collections import Counter

MIN_WORD_FREQ = 5       # word must appear this often in the corpus
MIN_BIGRAM_FREQ = 3
OUT_FILE = "tanglish_mined.json"

WORD_RE = re.compile(r"[a-z]+")


def english_filter():
    """Common English words (from Brown) to exclude from the mined vocab."""
    from nltk.corpus import brown
    freq = Counter(w.lower() for w in brown.words() if w.isalpha())
    return {w for w, c in freq.items() if c >= 5}


def mine():
    from datasets import load_dataset
    print("[miner] loading TamilMixSentiment...")
    ds = load_dataset("community-datasets/tamilmixsentiment")

    texts = []
    for split in ("train", "validation", "test"):
        if split in ds:
            texts += [row["text"] for row in ds[split]]
    print(f"[miner] {len(texts):,} comments loaded")

    english = english_filter()
    word_freq = Counter()
    bigram_freq = Counter()

    for t in texts:
        words = WORD_RE.findall(t.lower())
        word_freq.update(words)
        bigram_freq.update(zip(words, words[1:]))

    # Tanglish vocab: frequent in corpus, not common English, sane length
    VOWELS = set("aeiou")
    vocab = {w: c for w, c in word_freq.items()
             if c >= MIN_WORD_FREQ and w not in english
             and 3 < len(w) < 15
             and set(w) & VOWELS}          # romanized Tamil has vowels

    # keep bigrams where at least one side is a mined Tanglish word
    bigrams = {f"{a} {b}": c for (a, b), c in bigram_freq.items()
               if c >= MIN_BIGRAM_FREQ and (a in vocab or b in vocab)}

    data = {
        "source": "TamilMixSentiment (Dravidian-CodeMix, Chakravarthi et al. 2020)",
        "comments_mined": len(texts),
        "vocab": vocab,
        "bigrams": bigrams,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"[miner] mined vocabulary : {len(vocab):,} Tanglish words")
    print(f"[miner] mined bigrams    : {len(bigrams):,}")
    print(f"[miner] top mined words  : "
          f"{[w for w, _ in Counter(vocab).most_common(15)]}")
    print(f"[miner] saved -> {OUT_FILE}")


if __name__ == "__main__":
    mine()
