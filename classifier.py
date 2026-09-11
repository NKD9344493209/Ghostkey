"""
GhostKey - Concept 2: Text Classification
Per-word classifier: english / tanglish / name / gibberish.
Routes each token to the right correction path and protects names.
Character n-gram features + Logistic Regression, evaluated with P/R/F1.
"""

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tanglish import TANGLISH_WORDS


def keyboard_mash(rng, n):
    import string
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def build_training_data(seed=7):
    from nltk.corpus import brown, names
    rng = random.Random(seed)

    english = [w.lower() for w, c in
               __import__('collections').Counter(
                   w.lower() for w in brown.words() if w.isalpha()
               ).most_common(4000)]
    name_list = list({n.lower() for n in names.words()})
    rng.shuffle(name_list)
    tanglish = list(TANGLISH_WORDS)
    gibberish = [keyboard_mash(rng, rng.randint(3, 9)) for _ in range(3000)]

    X, y = [], []
    for w in english[:3000]:        X.append(w); y.append("english")
    for w in name_list[:3000]:      X.append(w); y.append("name")
    for w in tanglish * 12:         X.append(w); y.append("tanglish")
    for w in gibberish:             X.append(w); y.append("gibberish")
    return X, y


def train_classifier(seed=7):
    X, y = build_training_data(seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = Pipeline([
        ("feats", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))),
        ("model", LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")),
    ])
    clf.fit(Xtr, ytr)
    report = classification_report(yte, clf.predict(Xte), digits=3)
    return clf, report


class WordRouter:
    """Classifies each word; used by the pipeline to route and protect."""
    def __init__(self):
        self.clf, self.report = train_classifier()

    def classify(self, word):
        return self.clf.predict([word.lower()])[0]

    def classify_proba(self, word):
        probs = self.clf.predict_proba([word.lower()])[0]
        return dict(zip(self.clf.classes_, probs.round(3)))


if __name__ == "__main__":
    router = WordRouter()
    print(router.report)
    for w in ["friend", "semma", "nethra", "xkqzt", "machan", "college"]:
        print(f"{w:10s} -> {router.classify(w)}   {router.classify_proba(w)}")
