"""
GhostKey - Concepts 5 & 6: Multilingual NLP + Machine Translation
Tanglish (romanized Tamil-English code-mix) support:
  - Tanglish lexicon with English glosses
  - code-mixed correction (edit distance over the Tanglish vocabulary)
  - dictionary + rule based Tanglish -> English translation
Scope is honest: transfer translation on the lexicon, not a neural MT.
"""

# ---- Tanglish lexicon: romanized Tamil word -> English gloss ----
TANGLISH_LEXICON = {
    # pronouns / people
    "naan": "I", "nee": "you", "avan": "he", "aval": "she",
    "avanga": "they", "namma": "our", "enna": "what", "yaaru": "who",
    "machan": "buddy", "machi": "buddy", "thala": "boss", "anna": "brother",
    "akka": "sister", "amma": "mother", "appa": "father", "thambi": "little-brother",
    # verbs
    "poren": "going", "varen": "coming", "iruken": "am", "iruka": "is",
    "irukku": "is", "vaa": "come", "po": "go", "sollu": "tell",
    "solren": "telling", "paaru": "see", "paren": "seeing", "sapadu": "food",
    "sapten": "ate", "saptiya": "have-you-eaten", "thoongu": "sleep",
    "thoongren": "sleeping", "padikiren": "studying", "padichen": "studied",
    "panren": "doing", "pannu": "do", "mudiyala": "cannot", "mudiyum": "can",
    "theriyum": "know", "theriyala": "dont-know", "venum": "want",
    "vendam": "dont-want", "kudu": "give", "vaangu": "buy",
    # qualifiers / common
    "semma": "awesome", "romba": "very", "konjam": "little", "nalla": "good",
    "chumma": "casually", "seri": "okay", "aama": "yes", "illa": "no",
    "epdi": "how", "enga": "where", "eppo": "when", "ipo": "now",
    "apro": "afterwards", "aana": "but", "kandippa": "definitely",
    "vera": "other", "level": "level", "mass": "mass", "kodumai": "cruelty",
    "super": "super", "sirikira": "laughing", "azhaga": "beautifully",
    "veetla": "at-home", "veedu": "home", "ooru": "town", "padam": "movie",
    "paatu": "song", "vela": "work", "kasu": "money", "pasanga": "boys",
    "ponnu": "girl", "payyan": "boy", "friendu": "friend", "bore": "boring",
    "jolly": "fun", "gethu": "swag", "vandi": "vehicle", "saapdu": "eat",
    "da": "", "di": "", "pa": "", "ma": "",   # fillers - drop in translation
    # greetings / address words
    "dei": "hey", "yov": "hey", "mapla": "bro", "maame": "dude",
    "nanba": "friend", "nanban": "friend", "bro": "bro",
    # question/verb variants (spoken forms)
    "yenna": "what", "yaar": "who", "yean": "why", "een": "why",
    "panra": "doing", "panriya": "are-you-doing", "panreenga": "doing",
    "sollra": "saying", "varriya": "are-you-coming", "poriya": "are-you-going",
    "irukiya": "are-you-there", "sapta": "ate", "mudinchu": "finished",
    "vandhutten": "arrived", "poitten": "went", "paathen": "saw",
    "keten": "asked", "puriyala": "dont-understand", "puriyudhu": "understand",
    "onnum": "nothing", "ellam": "everything", "innum": "still",
    "yepdi": "how", "yen": "why", "unaku": "for-you", "enaku": "for-me",
}

TANGLISH_WORDS = set(TANGLISH_LEXICON)

# ---- mined vocabulary (run mine_tanglish.py to generate) ----
MINED_FREQ = {}
try:
    import json as _json, os as _os
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       "tanglish_mined.json")
    if _os.path.exists(_p):
        _d = _json.load(open(_p, encoding="utf-8"))
        MINED_FREQ = _d.get("vocab", {})
        TANGLISH_WORDS |= set(MINED_FREQ)
        print(f"[tanglish] +{len(MINED_FREQ):,} mined words "
              f"({_d.get('comments_mined', 0):,} comments)")
except Exception as _e:
    print("[tanglish] mined vocab not loaded:", _e)

# very small bigram hints for code-mixed context (hand-built, honest scope)
TANGLISH_BIGRAMS = {
    ("naan", "poren"), ("naan", "varen"), ("semma", "mass"),
    ("romba", "nalla"), ("enna", "panra"), ("veetla", "iruken"),
    ("padam", "semma"), ("nee", "vaa"), ("seri", "da"), ("romba", "kodumai"),
}

# simple reordering rules for the transfer step (Tamil SOV -> English SVO-ish)
PRONOUN_SUBJECTS = {"i", "you", "he", "she", "they", "we"}


def correct_tanglish_word(word, max_dist=1.6):
    """Correct a misspelled Tanglish word against the lexicon."""
    from corrector import weighted_edit_distance
    w = word.lower()
    if w in TANGLISH_WORDS:
        return w, 1.0
    # short words and vowel-less tokens (acronyms like nlp, css) are
    # too risky to fuzzy-match - leave them alone
    if len(w) < 4 or not set(w) & set("aeiouy"):
        return w, 0.0
    best, bd = None, 99
    for t in TANGLISH_WORDS:
        d = weighted_edit_distance(w, t, max_cost=max_dist + 0.5)
        # frequent mined words win ties (small frequency bonus)
        d -= min(0.3, MINED_FREQ.get(t, 0) / 2000.0)
        if d < bd:
            best, bd = t, d
    if best is not None and bd <= max_dist:
        conf = max(0.0, 1.0 - bd / 3.0)
        return best, round(conf, 2)
    return w, 0.0


def translate_tanglish(sentence_words):
    """Dictionary + rule based transfer translation to English."""
    glosses = []
    for w in sentence_words:
        lw = w.lower()
        if lw in TANGLISH_LEXICON:
            g = TANGLISH_LEXICON[lw]
            if g:                     # fillers translate to nothing
                glosses.append(g)
        else:
            glosses.append(w)          # English word passes through

    AUX = {"i": "am", "you": "are", "he": "is", "she": "is",
           "they": "are", "we": "are"}
    ING = {"going", "coming", "studying", "sleeping", "doing", "telling",
           "seeing", "eating", "laughing"}
    out = glosses[:]
    # rule 1: subject pronoun moves to front (Tamil allows it later)
    for i in range(1, len(out)):
        if out[i].lower() in AUX:
            out.insert(0, out.pop(i))
            break
    # rule 2: SOV -> SVO: sentence-final -ing verb moves after the subject
    if (len(out) >= 2 and out[0].lower() in AUX
            and out[-1].lower() in ING):
        verb = out.pop()
        out.insert(1, verb)
    # rule 3: insert the auxiliary: "I going" -> "I am going"
    if (len(out) >= 2 and out[0].lower() in AUX
            and out[1].lower() in ING):
        out.insert(1, AUX[out[0].lower()])
    # rule 4: "is/am/are" gloss directly after subject stays; final "am/is"
    # gloss moves after subject: "I at-home am" -> "I am at-home"
    if len(out) >= 2 and out[-1].lower() in {"am", "is", "are"}:
        aux = out.pop()
        if out[0].lower() in AUX:
            out.insert(1, AUX[out[0].lower()])
        else:
            out.insert(1, aux)
    text = " ".join(out).replace("-", " ")
    # spoken-question smoothing
    text = text.replace("what doing", "what are you doing")
    text = text.replace("hey bro what", "hey bro, what")
    return text


def evaluate_tanglish():
    """Small honest evaluation of correction + translation."""
    correction_tests = [
        ("smema", "semma"), ("romva", "romba"), ("machna", "machan"),
        ("porne", "poren"), ("nala", "nalla"), ("kandipa", "kandippa"),
    ]
    right = 0
    print("Tanglish correction:")
    for typed, gold in correction_tests:
        got, conf = correct_tanglish_word(typed)
        ok = got == gold
        right += ok
        print(f"  {typed:10s} -> {got:10s} ({conf:.2f}) {'OK' if ok else 'expected ' + gold}")
    print(f"  accuracy: {right}/{len(correction_tests)}")

    translation_tests = [
        ("naan college poren", "I am going college"),
        ("padam semma mass", "movie awesome mass"),
        ("naan veetla iruken", "I am at home"),
        ("nee romba nalla iruka", "you very good is"),
    ]
    print("\nTanglish -> English translation:")
    for src, _ in translation_tests:
        print(f"  {src:28s} -> {translate_tanglish(src.split())}")


if __name__ == "__main__":
    evaluate_tanglish()
