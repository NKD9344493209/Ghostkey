"""
GhostKey - Module 1: Eyes-Free Typing Error Simulator
Generates (wrong, correct) training pairs by corrupting clean text
the way eyes-free typing does: adjacent-key hits, deletions,
duplications, and transpositions.
"""

import random

# QWERTY keyboard geometry - which keys sit next to which
ADJACENT = {
    'q': 'wa',        'w': 'qeas',      'e': 'wrds',      'r': 'etdf',
    't': 'ryfg',      'y': 'tugh',      'u': 'yihj',      'i': 'uojk',
    'o': 'ipkl',      'p': 'ol',
    'a': 'qwsz',      's': 'awedxz',    'd': 'serfcx',    'f': 'drtgvc',
    'g': 'ftyhbv',    'h': 'gyujnb',    'j': 'huikmn',    'k': 'jiolm',
    'l': 'kop',
    'z': 'asx',       'x': 'zsdc',      'c': 'xdfv',      'v': 'cfgb',
    'b': 'vghn',      'n': 'bhjm',      'm': 'njk',
}

ERROR_TYPES = ("substitute", "delete", "duplicate", "transpose")


def corrupt_word(word, error_prob=0.35, seed_rng=None):
    """Corrupt one word with at most one eyes-free typing error."""
    rng = seed_rng or random
    if len(word) < 2 or rng.random() > error_prob:
        return word, None

    error = rng.choice(ERROR_TYPES)
    i = rng.randrange(len(word))
    ch = word[i].lower()

    if error == "substitute" and ch in ADJACENT:
        wrong = rng.choice(ADJACENT[ch])
        return word[:i] + wrong + word[i+1:], "substitute"
    if error == "delete" and len(word) > 2:
        return word[:i] + word[i+1:], "delete"
    if error == "duplicate":
        return word[:i+1] + word[i] + word[i+1:], "duplicate"
    if error == "transpose" and i < len(word) - 1:
        return word[:i] + word[i+1] + word[i] + word[i+2:], "transpose"
    return word, None


def corrupt_sentence(sentence_words, error_prob=0.35, seed=None):
    """Corrupt a list of words; returns (corrupted_words, error_log)."""
    rng = random.Random(seed) if seed is not None else random
    out, log = [], []
    for w in sentence_words:
        bad, etype = corrupt_word(w, error_prob, rng)
        out.append(bad)
        if etype:
            log.append((w, bad, etype))
    return out, log


def build_dataset(n_sentences=2000, error_prob=0.35, seed=42):
    """Build aligned (corrupted, clean) sentence pairs from the Brown corpus."""
    from nltk.corpus import brown
    rng = random.Random(seed)
    clean_sents = [
        [w.lower() for w in s if w.isalpha()]
        for s in brown.sents()
    ]
    clean_sents = [s for s in clean_sents if 4 <= len(s) <= 12]
    rng.shuffle(clean_sents)
    dataset = []
    for s in clean_sents[:n_sentences]:
        corrupted, log = corrupt_sentence(s, error_prob, rng.random())
        dataset.append({"clean": s, "corrupt": corrupted, "errors": log})
    return dataset


if __name__ == "__main__":
    data = build_dataset(n_sentences=5)
    for d in data:
        print("CLEAN  :", " ".join(d["clean"]))
        print("CORRUPT:", " ".join(d["corrupt"]))
        print("ERRORS :", d["errors"])
        print()
