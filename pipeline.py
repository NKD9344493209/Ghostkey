"""
GhostKey - Concept 3 + Pipeline Integration
NER-based name/organization protection (spaCy) and the unified
six-concept pipeline:

  classify word (C2) -> NER protect (C3) -> route:
      english   -> ensemble correction (C1)
      tanglish  -> tanglish correction (C5)
      name      -> protect, never correct
      gibberish -> english correction attempt
  -> next-word prediction (C1 LM) -> optional translation (C6)
  Speech in/out (C4) lives in the app layer (browser STT/TTS).
"""

from corrector import GhostKeyCorrector, LanguageModel
from classifier import WordRouter
from tanglish import (correct_tanglish_word, translate_tanglish,
                      TANGLISH_WORDS)


class NERProtector:
    """Concept 3: detect PERSON/ORG in the sentence, protect from correction."""
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def protected_words(self, text):
        doc = self.nlp(text)
        prot = set()
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE"):
                for tok in ent:
                    prot.add(tok.text.lower())
        return prot


class GhostKeyPipeline:
    """The full six-concept GhostKey engine."""

    def __init__(self):
        print("[GhostKey] loading full pipeline...")
        self.ck = GhostKeyCorrector(LanguageModel())   # C1
        self.router = WordRouter()                     # C2
        self.ner = NERProtector()                      # C3
        print("[GhostKey] pipeline ready.")

    def process(self, text, auto_threshold=0.80, translate=False):
        """Correct a sentence through the full pipeline.
        Returns dict with corrected text, per-word report, prediction,
        and optional translation."""
        ner_protected = self.ner.protected_words(text)          # C3
        out, report = [], []
        prev = None
        has_tanglish = False

        for token in text.split():
            w = token.lower()
            if not w.isalpha():
                out.append(token)
                report.append((token, token, 1.0, "non-alpha", "-"))
                continue

            # acronym guard: vowel-less tokens (nlp, css, html) are
            # intentional technical terms, never correction targets
            if not set(w) & set("aeiou"):
                out.append(token)
                report.append((token, token, 1.0, "acronym", "-"))
                prev = w
                continue

            probs = self.router.classify_proba(w)                # C2
            wtype = max(probs, key=probs.get)

            # C3 protect - needs strong evidence: real NER entities are
            # capitalized; the classifier alone must be very sure
            is_cap = token[0].isupper()
            if (w in self.ck.never_correct
                    or (w in ner_protected and is_cap)
                    or (wtype == "name"
                        and (is_cap or probs["name"] > 0.90))):
                out.append(token)
                report.append((token, token, 1.0, "protected", wtype))
                prev = w
                continue

            # C5 - but frequent English words never route to Tanglish
            is_common_english = self.ck.lm.freq[w] >= 50
            if (not is_common_english
                    and (w in TANGLISH_WORDS or probs["tanglish"] > 0.35)):
                best, conf = correct_tanglish_word(w)
                if best != w and conf < 0.70:
                    best, conf = w, 1.0          # too unsure - keep as typed
                has_tanglish = True
                action = "tanglish-fix" if best != w else "tanglish-ok"
                out.append(best)
                report.append((token, best, conf, action, wtype))
                prev = best
                continue

            # english / gibberish -> ensemble correction (C1)
            best, conf, _ = self.ck.correct_word(w, prev)
            # words unknown to English also try the Tanglish lexicon;
            # whichever corrector is more confident wins the word
            t_best, t_conf = (correct_tanglish_word(w)
                              if not self.ck.lm.is_word(w) else (w, 0.0))
            if t_best != w and t_conf >= 0.70 and t_conf > conf:
                has_tanglish = True
                out.append(t_best); action = "tanglish-fix"; conf = t_conf
            elif best != w and conf >= auto_threshold:
                out.append(best); action = "auto"
            elif best != w and conf >= 0.60:
                out.append(token); action = f"suggest:{best}"
            else:
                out.append(token); action = "keep"
            report.append((token, out[-1], round(conf, 3), action, wtype))
            prev = out[-1]

        corrected = " ".join(out)

        prediction = self.predict_next(prev)                     # C1 LM

        translation = None                                       # C6
        if translate or has_tanglish:
            translation = translate_tanglish(corrected.split())

        return {
            "corrected": corrected,
            "report": report,
            "next_words": prediction,
            "translation": translation,
            "protected": sorted(ner_protected),
        }

    def predict_next(self, prev, k=3):
        """Next-word prediction from the bigram model."""
        if not prev:
            return []
        cands = [(c, w2) for (w1, w2), c in self.ck.lm.bigrams.items()
                 if w1 == prev]
        cands.sort(reverse=True)
        return [w for _, w in cands[:k]]


if __name__ == "__main__":
    pipe = GhostKeyPipeline()
    tests = [
        "i lobe you nethra",
        "John works at Google and he is my best friemd",
        "naan college poren machan padam smema mass",
        "the river bamk is beautiful",
    ]
    for t in tests:
        r = pipe.process(t)
        print(f"\nIN  : {t}")
        print(f"OUT : {r['corrected']}")
        print(f"NEXT: {r['next_words']}")
        if r['translation']:
            print(f"TRAN: {r['translation']}")
        if r['protected']:
            print(f"NER : {r['protected']}")
        for row in r["report"]:
            if row[3] not in ("keep", "non-alpha", "tanglish-ok"):
                print("   ", row)
