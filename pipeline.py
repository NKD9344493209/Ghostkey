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

        def split_punct(tok):
            """'hello,' -> ('', 'hello', ','); keeps punctuation safe."""
            i, j = 0, len(tok)
            while i < j and not tok[i].isalnum(): i += 1
            while j > i and not tok[j-1].isalnum(): j -= 1
            return tok[:i], tok[i:j], tok[j:]

        def recase(orig_core, corrected):
            if orig_core[:1].isupper():
                return corrected[:1].upper() + corrected[1:]
            return corrected

        tokens_list = text.split()
        # merge pass: "th e" -> "the" (a space typed mid-word)
        merged, k = [], 0
        while k < len(tokens_list):
            w1 = tokens_list[k].lower()
            w2 = (tokens_list[k + 1].lower()
                  if k + 1 < len(tokens_list) else None)
            if (w2 and w1.isalpha() and w2.isalpha()
                    and self.ck.lm.freq[w1] < 5      # w1 is not real usage
                    and self.ck.lm.freq[w1 + w2] >= 20):
                merged.append(tokens_list[k] + tokens_list[k + 1])
                k += 2
            else:
                merged.append(tokens_list[k])
                k += 1
        tokens_list = merged
        for ti, raw_token in enumerate(tokens_list):
            pre, token, post = split_punct(raw_token)
            w = token.lower()
            if ti + 1 < len(tokens_list):
                _, nx_core, _ = split_punct(tokens_list[ti + 1])
                nxt_tok = nx_core.lower() or None
            else:
                nxt_tok = None
            if not w.isalpha():
                out.append(raw_token)
                report.append((raw_token, raw_token, 1.0, "non-alpha", "-"))
                continue

            # vowel-less tokens (nlp, css) are usually acronyms - but a
            # cheap-slip typo like "wsnt" must still be correctable, so
            # the decision happens after scoring (see below)
            vowelless = not set(w) & set("aeiouy")

            probs = self.router.classify_proba(w)                # C2
            wtype = max(probs, key=probs.get)

            # C3 protect - needs strong evidence: real NER entities are
            # capitalized; the classifier alone must be very sure
            is_cap = token[0].isupper()
            if (w in self.ck.never_correct
                    or (w in ner_protected and is_cap)
                    or (wtype == "name"
                        and (is_cap or probs["name"] > 0.90))):
                out.append(pre + token + post)
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
                out.append(pre + recase(token, best) + post)
                report.append((token, best, conf, action, wtype))
                prev = best
                continue

            # english / gibberish -> ensemble correction (C1)
            best, conf, _ = self.ck.correct_word(w, prev, nxt_tok)
            # words unknown to English also try the Tanglish lexicon;
            # whichever corrector is more confident wins the word
            t_best, t_conf = (correct_tanglish_word(w)
                              if not self.ck.lm.is_word(w) else (w, 0.0))
            # evaluated Model-D policy: changing a real dictionary word
            # needs high confidence; fixing a non-word is low-risk
            from corrector import weighted_edit_distance as _wed
            if vowelless and (_wed(w, best) > 0.6 or conf < 0.45):
                out.append(pre + token + post)    # true acronym: keep
                report.append((token, token, 1.0, "acronym", "-"))
                prev = w
                continue
            if self.ck.is_real_word(w):
                needed = auto_threshold          # real word: high bar
            elif wtype == "name" or probs["name"] > 0.35:
                # name-like unknown: correct only a single-operation slip
                # (0.85 covers one deletion/insertion/transposition/adjacent)
                needed = 0.45 if _wed(w, best) <= 0.85 else auto_threshold
            elif len(w) >= 8 and _wed(w, best) <= 1.3:
                needed = 0.30    # long typo, sparse candidate space: fix it
            else:
                needed = 0.45                    # non-word: low-risk fix
            if t_best != w and t_conf >= 0.70 and t_conf > conf:
                has_tanglish = True
                out.append(pre + recase(token, t_best) + post)
                action = "tanglish-fix"; conf = t_conf
            elif best != w and conf >= needed:
                out.append(pre + recase(token, best) + post); action = "auto"
            elif best != w and conf >= 0.35:
                out.append(pre + token + post); action = f"suggest:{best}"
            else:
                out.append(pre + token + post); action = "keep"
            core_out = out[-1].strip(".,!?;:'\"()[]").lower()
            report.append((token, core_out, round(conf, 3), action, wtype))
            prev = core_out

        corrected = " ".join(out)

        prediction = self.predict_next(prev)                     # C1 LM

        translation = None                                       # C6
        from tanglish import TANGLISH_LEXICON
        really_tanglish = any(
            TANGLISH_LEXICON.get(t.lower(), t.lower()) not in ("", t.lower())
            for t in corrected.split())
        if translate or (has_tanglish and really_tanglish):
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