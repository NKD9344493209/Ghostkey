"""
GhostKey - Test Suite
1. Regression tests: every bug ever found in development, locked forever.
2. Bulk iterations: hundreds of simulator-corrupted sentences through the
   FULL pipeline, measuring accuracy and listing any clean word broken.

Run:  python tests.py
"""

from simulator import build_dataset
from pipeline import GhostKeyPipeline

REGRESSIONS = {
    # core correction
    "i lobe you": "i love you",
    "the river bamk is beautiful": "the river bank is beautiful",
    "she is my best friemd": "she is my best friend",
    "i wsnt to go home": "i want to go home",
    # names / entities protected
    "i lobe you nethra": "i love you nethra",
    "John works at Google and he is my best friemd":
        "John works at Google and he is my best friend",
    # acronyms & shorthand
    "i love nlp": "i love nlp",
    "my css and html files": "my css and html files",
    "hello bro i need u to get married immeidatly":
        "hello bro i need u to get married immediately",
    # conversational + merge + single letters
    "Hello hw are yu im fine im doing this o test my app in th e computer":
        "Hello how are you im fine im doing this to test my app in the computer",
    "hello how re you": "hello how are you",
    # punctuation + capitalization
    "Hello, hw are yu? im fine!": "Hello, how are you? im fine!",
    "The river bamk is beautiful.": "The river bank is beautiful.",
    # tanglish
    "dei mapla yenna panrs": "dei mapla yenna panra",
    "na veetla iruken": "na veetla iruken",
}


def run_regressions(pipe):
    failed = []
    for text, want in REGRESSIONS.items():
        got = pipe.process(text)["corrected"]
        if got != want:
            failed.append((text, want, got))
    print(f"[regressions] {len(REGRESSIONS) - len(failed)}"
          f"/{len(REGRESSIONS)} pass")
    for t, w, g in failed:
        print(f"  FAIL: {t!r}\n     want: {w!r}\n     got : {g!r}")
    # tanglish translation gating
    assert pipe.process("naan college poren machan")["translation"]
    assert pipe.process("hello bro how are you")["translation"] is None
    print("[regressions] translation gating OK")
    return not failed


def run_bulk(pipe, n_sentences=150, error_prob=0.45, seed=7):
    data = build_dataset(n_sentences=n_sentences,
                         error_prob=error_prob, seed=seed)
    fixed = tot = broke = clean = 0
    broken_examples = []
    for d in data:
        out = pipe.process(" ".join(d["corrupt"]))["corrected"].split()
        if len(out) != len(d["clean"]):     # merge changed token count
            continue
        for cw, bw, ow in zip(d["clean"], d["corrupt"], out):
            if bw != cw:
                tot += 1
                fixed += (ow.lower() == cw)
            else:
                clean += 1
                if ow.lower() != cw:
                    broke += 1
                    if len(broken_examples) < 10:
                        broken_examples.append((cw, ow))
    print(f"[bulk n={n_sentences}] correction accuracy "
          f"{100*fixed/tot:.1f}% ({fixed}/{tot})   "
          f"clean words broken {100*broke/clean:.2f}% ({broke}/{clean})")
    for cw, ow in broken_examples:
        print(f"   broke clean word: {cw!r} -> {ow!r}")
    return 100 * fixed / tot, 100 * broke / clean


if __name__ == "__main__":
    pipe = GhostKeyPipeline()
    ok = run_regressions(pipe)
    acc, over = run_bulk(pipe)
    print("\n==== SUMMARY ====")
    print(f"regressions : {'ALL PASS' if ok else 'FAILURES - see above'}")
    print(f"bulk        : {acc:.1f}% accuracy, {over:.2f}% overcorrection")