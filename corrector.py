"""
GhostKey - Module 2: Ensemble Correction Engine
Noisy-channel spelling correction for eyes-free typing.

Techniques voting on every word:
  1. Dictionary check           - is it a valid word?
  2. Keyboard-weighted edit distance - which slips are physically likely?
  3. Bigram language model      - which candidate fits the context?
  4. Word frequency prior       - which words are common at all?
Combined score -> confidence -> three-zone policy (auto / suggest / leave).
"""

import math
import string
from collections import Counter
from functools import lru_cache

from simulator import ADJACENT


def ensure_nltk(packages):
    """Industry rule: download what's missing instead of degrading silently."""
    import nltk
    for pkg in packages:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            print(f"[GhostKey] downloading missing corpus: {pkg} ...")
            nltk.download(pkg, quiet=True)

LETTERS = string.ascii_lowercase

# ---------------- Language model (built once) ----------------

class LanguageModel:
    def __init__(self):
        ensure_nltk(["brown", "words", "nps_chat", "webtext"])
        from nltk.corpus import brown, words as words_corpus
        from nltk.corpus import nps_chat, webtext
        print("[GhostKey] building language model (Brown + chat corpora)...")
        tokens = [w.lower() for w in brown.words() if w.isalpha()]
        # conversational text so "how are you", "im fine" etc. score well
        chat = [w.lower() for w in nps_chat.words() if w.isalpha()]
        web = [w.lower() for w in webtext.words() if w.isalpha()]
        tokens = tokens + chat * 3 + web          # chat weighted up
        self.freq = Counter(tokens)
        self.total = sum(self.freq.values())
        self.bigrams = Counter(zip(tokens, tokens[1:]))
        self.vocab = set(self.freq) | {w.lower() for w in words_corpus.words()}
        print(f"[GhostKey] vocab={len(self.vocab):,}  bigrams={len(self.bigrams):,}")

    def is_word(self, w):
        return w.lower() in self.vocab

    def p_word(self, w):
        """Unigram probability with add-one floor."""
        return (self.freq[w] + 1) / (self.total + len(self.vocab))

    def p_bigram(self, prev, w):
        """P(w | prev) with backoff to unigram."""
        if prev and self.freq[prev]:
            c = self.bigrams[(prev, w)]
            if c:
                return c / self.freq[prev]
        return 0.4 * self.p_word(w)   # stupid-backoff style


# ---------------- Keyboard-weighted edit distance ----------------

def sub_cost(a, b):
    """Substitution cost: adjacent keys are cheap (likely finger slips)."""
    if a == b:
        return 0.0
    if b in ADJACENT.get(a, "") or a in ADJACENT.get(b, ""):
        return 0.5          # neighbouring key - very plausible slip
    return 1.5              # distant key - unlikely slip

def weighted_edit_distance(s, t, max_cost=3.0):
    """Damerau-Levenshtein with eyes-free cost model:
    recovering a deleted letter (0.55) and transpositions (0.5) are
    priced as cheaply as adjacent-key slips, because eyes-free typing
    produces them constantly (validated by error analysis)."""
    DEL, INS, TR = 0.8, 0.55, 0.5
    m, n = len(s), len(t)
    if abs(m - n) * INS > max_cost:
        return max_cost + 1
    prev2 = None
    prev = [j * INS for j in range(n + 1)]
    for i in range(1, m + 1):
        cur = [i * DEL] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(
                prev[j] + DEL,                        # user duplicated
                cur[j - 1] + INS,                     # user deleted
                prev[j - 1] + sub_cost(s[i-1], t[j-1])  # substitution
            )
            if (prev2 is not None and i > 1 and j > 1
                    and s[i-1] == t[j-2] and s[i-2] == t[j-1]):
                cur[j] = min(cur[j], prev2[j - 2] + TR)   # transposition
        prev2, prev = prev, cur
    return prev[n]


# ---------------- Candidate generation ----------------

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in LETTERS]
    inserts = [L + c + R for L, R in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


# ---------------- The corrector ----------------

class GhostKeyCorrector:
    USERDATA = "ghostkey_userdata.json"

    def __init__(self, lm=None):
        self.lm = lm or LanguageModel()
        self.never_correct = set()        # user-protected words
        self.personal_freq = Counter()    # self-learning vocabulary
        self.fingerprint = Counter()      # user's (intended->typed) slips
        self._load_userdata()

    def _load_userdata(self):
        import json, os
        if os.path.exists(self.USERDATA):
            try:
                d = json.load(open(self.USERDATA, encoding="utf-8"))
                self.never_correct = set(d.get("never", []))
                self.personal_freq = Counter(d.get("personal", {}))
                self.fingerprint = Counter(
                    {tuple(k.split("|")): v
                     for k, v in d.get("fingerprint", {}).items()})
                print(f"[GhostKey] loaded user profile "
                      f"({len(self.never_correct)} protected, "
                      f"{len(self.personal_freq)} learned words)")
            except Exception as e:
                print("[GhostKey] user profile unreadable, starting fresh:", e)

    def save_userdata(self):
        import json
        d = {"never": sorted(self.never_correct),
             "personal": dict(self.personal_freq),
             "fingerprint": {f"{a}|{b}": v
                             for (a, b), v in self.fingerprint.items()}}
        with open(self.USERDATA, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)

    # ----- candidate scoring -----
    def candidates(self, word):
        word = word.lower()
        cands = {word} if self.lm.is_word(word) else set()
        e1 = edits1(word)
        cands |= {c for c in e1 if self.lm.is_word(c)}
        # Always explore distance-2 for non-words: the intended common word
        # (e.g. 'good' for 'tgoof') may sit at distance 2 behind a rare
        # distance-1 word ('goof'). Context scoring referees between them.
        if not self.lm.is_word(word):
            budget = 30000
            for e in e1:
                for c in edits1(e):
                    if self.lm.is_word(c):
                        cands.add(c)
                budget -= len(e)
                if budget <= 0:
                    break
        # length guard: a plausible slip never destroys most of the word
        # (blocks e.g. 'kla' -> 'a'); also keeps single-letter junk out
        min_len = max(2, int(len(word) * 0.6)) if len(word) > 2 else 1
        cands = {c for c in cands if len(c) >= min_len}
        return cands or {word}

    def score(self, typed, cand, prev_word, next_word=None):
        """Combined noisy-channel score: keyboard physics x context x prior."""
        dist = weighted_edit_distance(typed.lower(), cand)
        # personal fingerprint bonus: user known to make this exact slip
        if self.fingerprint[(cand, typed.lower())]:
            dist *= 0.7
        keyboard_likelihood = math.exp(-2.8 * dist)
        context = self.lm.p_bigram(prev_word, cand)
        prior = self.lm.p_word(cand) + self.personal_freq[cand] / 1000.0
        score = keyboard_likelihood * (context ** 0.5) * (prior ** 0.30)
        if next_word:      # bidirectional: candidate must also fit what follows
            score *= self.lm.p_bigram(cand, next_word) ** 0.55
        # dictionary words never seen in the corpus (e.g. 'mucin', 'goof')
        # are valid but almost never what an eyes-free typist intended
        if self.lm.freq[cand] == 0 and not self.personal_freq[cand]:
            score *= 0.05
        return score

    # ----- public API -----
    def is_real_word(self, w):
        """Single letters other than a/i are treated as typos, not words."""
        if len(w) == 1:
            return w in ("a", "i", "u")   # u = chat shorthand for you
        return self.lm.is_word(w)

    def correct_word(self, typed, prev_word=None, next_word=None, explain=False):
        """Returns (best_word, confidence, explanation_list)."""
        w = typed.lower()
        if w in self.never_correct or not w.isalpha():
            return typed, 1.0, [("protected/non-alpha", typed, 1.0)]

        cands = self.candidates(w)
        if len(w) == 1:      # a single letter must appear in its correction
            cands = {c for c in cands if w in c and len(c) <= 3} or {w}
        scored = sorted(
            ((self.score(w, c, prev_word, next_word), c) for c in cands),
            reverse=True
        )
        best_score, best = scored[0]
        # Confidence: how decisively the winner beats the runner-up.
        # (Share-of-all-candidates dilutes when many weak candidates exist.)
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        confidence = best_score / (best_score + second_score + 1e-12)
        total = sum(s for s, _ in scored) or 1e-12

        explanation = None
        if explain:
            explanation = [
                {
                    "candidate": c,
                    "edit_distance": round(weighted_edit_distance(w, c), 2),
                    "context_P": round(self.lm.p_bigram(prev_word, c), 6),
                    "frequency": self.lm.freq[c],
                    "share": round(s / total, 3),
                }
                for s, c in scored[:5]
            ]
        return best, confidence, explanation

    def correct_sentence(self, text, auto_threshold=0.95,
                         suggest_threshold=0.60):
        """Three-zone policy over a whole sentence."""
        out = []
        prev = None
        report = []
        tokens = text.split()
        for i, token in enumerate(tokens):
            nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else None
            best, conf, _ = self.correct_word(token, prev, nxt)
            if best != token.lower() and conf >= auto_threshold:
                out.append(best); action = "auto"
            elif best != token.lower() and conf >= suggest_threshold:
                out.append(token); action = f"suggest:{best}"
            else:
                out.append(token); action = "keep"
            report.append((token, best, round(conf, 3), action))
            prev = out[-1].lower()
        return " ".join(out), report

    # ----- learning -----
    def feedback(self, typed, suggested, accepted: bool):
        """Accept/reject loop: learn vocabulary and error fingerprint."""
        if accepted:
            self.personal_freq[suggested] += 1
            self.fingerprint[(suggested, typed.lower())] += 1
        else:
            self.never_correct.add(typed.lower())
            self.personal_freq[typed.lower()] += 2
        self.save_userdata()              # learning survives restarts


if __name__ == "__main__":
    ck = GhostKeyCorrector()
    tests = [
        "i lobe you",
        "the river bamk is beautiful",
        "i wsnt to go hime",
        "she is my best friemd",
    ]
    for t in tests:
        fixed, report = ck.correct_sentence(t, auto_threshold=0.80)
        print(f"\nIN : {t}\nOUT: {fixed}")
        for r in report:
            print("   ", r)