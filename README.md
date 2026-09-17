# GhostKey — AI Keyboard for Eyes-Free Typing (FINAL)

All six NLP concepts, one product.

## Files
| File | Concept | What it does |
|---|---|---|
| `simulator.py` | Data | QWERTY error simulator — generates the dataset |
| `corrector.py` | C1 | Ensemble correction engine + confidence policy + learning |
| `evaluate.py` | C1 eval | 4-model accuracy comparison table |
| `classifier.py` | C2 | Word-type classifier (english/tanglish/name/gibberish) + P/R/F1 |
| `pipeline.py` | C3 + integration | spaCy NER protection + full six-concept pipeline |
| `tanglish.py` | C5 + C6 | Tanglish lexicon, code-mixed correction, translation + eval |
| `app.py` | C4 + product | Final Streamlit keyboard: voice in/out, explain panel, profile dashboard |

## Setup (one time)
```
pip install nltk streamlit scikit-learn spacy
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(p) for p in ['brown','words','punkt','names']]"
```
If the spacy model download fails:
```
pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## Run (each prints its own evaluation — screenshot everything)
```
python simulator.py       -> dataset examples          (Data marks)
python evaluate.py        -> C1: 4-model accuracy table
python classifier.py      -> C2: P/R/F1 report
python pipeline.py        -> C3 + integration tests
python tanglish.py        -> C5 correction + C6 translation eval
python -m streamlit run app.py   -> the product (C4 voice lives here)
```

## Verified results
- C1: A 59.7% -> B 72.6% -> C 86.8% -> D 83.6% (overcorrection 0.5%)
- C2: overall accuracy 87%, tanglish F1 0.947
- C3: John/Google protected, "i lobe you nethra" -> love corrected, nethra untouched
- C5: 6/6 misspelled Tanglish words corrected
- C6: "naan veetla iruken" -> "I am at home"

## Demo script (for the presentation)
1. Toggle OFF, type garbled -> mess. Toggle ON -> fixed.
2. "i lobe you nethra" -> love fixed, name protected. Open explain panel.
3. "naan college poren machan" -> Tanglish corrected + English translation.
4. 🎤 speak a sentence -> corrected. 🔊 speak back.
5. Reject a correction -> word protected forever. Show Typing Profile tab.
6. Turn off WiFi -> everything still works.
hi everyone