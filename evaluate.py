"""
GhostKey - Module 3: Evaluation
Compares three correction models of increasing sophistication on
simulator-generated eyes-free typing data:

  Model A: plain edit distance + word frequency   (classic spell check)
  Model B: keyboard-weighted edit distance + freq (knows QWERTY physics)
  Model C: B + bigram context                     (full GhostKey engine)

Metric: word-level correction accuracy on corrupted words,
plus false-correction rate on clean words (overcorrection).
"""

import math
from collections import Counter

from simulator import build_dataset
from corrector import (GhostKeyCorrector, LanguageModel,
                       weighted_edit_distance, edits1)


def plain_edit_distance(s, t):
    """Uniform-cost Levenshtein (Model A's channel model)."""
    m, n = len(s), len(t)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if s[i-1] == t[j-1] else 1))
        prev = cur
    return prev[n]


class ModelA:
    """Dictionary + plain edit distance + frequency prior."""
    name = "A: plain edit distance"
    def __init__(self, lm): self.lm = lm
    def correct(self, word, prev=None):
        w = word.lower()
        if self.lm.is_word(w):
            return w
        cands = {c for c in edits1(w) if self.lm.is_word(c)}
        if not cands:
            return w
        return max(cands, key=lambda c: (-plain_edit_distance(w, c),
                                         self.lm.freq[c]))


class ModelB:
    """Dictionary + keyboard-weighted edit distance + frequency prior."""
    name = "B: + keyboard weighting"
    def __init__(self, lm): self.lm = lm
    def correct(self, word, prev=None):
        w = word.lower()
        if self.lm.is_word(w):
            return w
        cands = {c for c in edits1(w) if self.lm.is_word(c)}
        if not cands:
            return w
        def score(c):
            return math.exp(-1.6 * weighted_edit_distance(w, c)) * \
                   (self.lm.p_word(c) ** 0.3)
        return max(cands, key=score)


class ModelC:
    """Full GhostKey ensemble (keyboard + bigram context + priors)."""
    name = "C: + bigram context (GhostKey)"
    def __init__(self, lm): self.ck = GhostKeyCorrector(lm)
    def correct(self, word, prev=None):
        best, conf, _ = self.ck.correct_word(word, prev)
        return best


class ModelD:
    """Model C + confidence three-zone policy (the shipping GhostKey)."""
    name = "D: + confidence policy"
    def __init__(self, lm, threshold=0.75): 
        self.ck = GhostKeyCorrector(lm)
        self.threshold = threshold
    def correct(self, word, prev=None):
        w = word.lower()
        best, conf, _ = self.ck.correct_word(w, prev)
        if best != w:
            # changing a dictionary word needs high confidence;
            # changing a non-word needs only moderate confidence
            needed = self.threshold if self.ck.lm.is_word(w) else 0.30
            if conf < needed:
                return w
        return best


def evaluate(n_test=400, error_prob=0.5, seed=99):
    lm = LanguageModel()
    models = [ModelA(lm), ModelB(lm), ModelC(lm), ModelD(lm)]
    data = build_dataset(n_sentences=n_test, error_prob=error_prob, seed=seed)

    results = {}
    for model in models:
        fixed_right = corrupted_total = 0
        broke_clean = clean_total = 0
        for d in data:
            prev = None
            for clean_w, typed_w in zip(d["clean"], d["corrupt"]):
                out = model.correct(typed_w, prev)
                if typed_w != clean_w:                # word was corrupted
                    corrupted_total += 1
                    if out == clean_w:
                        fixed_right += 1
                else:                                  # word was clean
                    clean_total += 1
                    if out != clean_w:
                        broke_clean += 1
                prev = out
        acc = 100 * fixed_right / corrupted_total
        overcorr = 100 * broke_clean / clean_total
        results[model.name] = (acc, overcorr, corrupted_total)
        print(f"{model.name:35s} accuracy={acc:5.1f}%   "
              f"overcorrection={overcorr:4.1f}%   (n={corrupted_total})")
    return results


if __name__ == "__main__":
    print("GhostKey 3-model evaluation on simulated eyes-free typing\n")
    evaluate()
