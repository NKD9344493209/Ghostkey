# ⌨️ GhostKey — AI Keyboard for Eyes-Free Typing

**An adaptive, explainable, privacy-first AI keyboard that corrects the errors
of typing without looking — and tells you *why* it corrected.**

`Python` `NLTK` `spaCy` `scikit-learn` `Streamlit` · 100% offline · MIT License

---

## Why

Typing without visual attention — visually impaired users, walking, low light —
produces a specific error pattern: adjacent-key slips, dropped and doubled
letters, transpositions. Commercial autocorrect assumes you're glancing at the
screen, corrects silently (useless if you can't see it), and never explains
itself. GhostKey is built for exactly this gap.

## What it does

| Capability | How |
|---|---|
| Real-time correction (88.8% on eyes-free errors, 0.13% overcorrection) | Noisy-channel ensemble: keyboard-weighted edit distance × **bidirectional** bigram context × frequency priors |
| **Explain My Correction** | Every replacement shows its evidence: edit ops, key adjacency, context probability, confidence |
| Confidence policy | Auto-fix / suggest / leave-alone zones; names & acronyms protected (spaCy NER + classifier) |
| Self-learning | Accept/reject feedback updates a persistent personal dictionary & error fingerprint |
| Conversational English | Language model trained on Brown + NPS Chat + Webtext (257k vocab, 550k bigrams) |
| Tanglish | Corrects Tamil-English code-mix (1,857 words mined from TamilMixSentiment) and translates to English |
| Speech loop | Voice in (browser STT) → correct → sentence spoken back (TTS) |
| Word surgery | Merges split words (`th e → the`), fixes single letters (`o → to`), keeps punctuation & case |
| Product features | 4 themes, mobile-responsive (use from a phone browser, zero install), apply/copy, typing-profile dashboard |

## The six NLP concepts inside

1. **Spelling correction & language modelling** — the ensemble engine + next-word prediction
2. **Text classification** — char-n-gram LogReg routes english / tanglish / name / gibberish (Tanglish F1 0.95)
3. **Named Entity Recognition** — spaCy protects PERSON/ORG/GPE from overcorrection
4. **Speech processing** — STT input, sentence-level TTS confirmation
5. **Multilingual NLP** — code-mixed Tanglish correction
6. **Machine translation** — dictionary + rule-based Tanglish → English

## Results

| Model | Accuracy | Overcorrection |
|---|---|---|
| A — plain edit distance | 56.8% | 0.0% |
| B — + keyboard weighting | 70.4% | 0.0% |
| C — + bidirectional context | **91.6%** | 1.6% |
| D — + confidence policy (shipping) | 88.4% | **0.2%** |

Full pipeline on 150 fresh corrupted sentences: **88.8% / 0.13%** (`python tests.py`).

## Quick start

```bash
git clone <this repo>
cd ghostkey
setup.bat                # one-time: installs everything (Windows)
python -m streamlit run app.py
```

Manual setup: `pip install -r requirements.txt`, the spaCy model, then the
engine auto-downloads any missing NLTK corpora on first run.

**Use from a phone (no app):**
`python -m streamlit run app.py -- --server.address 0.0.0.0`
then open `http://<laptop-ip>:8501` on the phone (same WiFi).

## Project layout

```
app.py            Streamlit product (keyboard, themes, explain panel, dashboard)
pipeline.py       Six-concept pipeline + routing guards
corrector.py      Ensemble engine, confidence policy, persistent learning
classifier.py     Word-type classifier (concept 2)
tanglish.py       Tanglish lexicon, correction, translation (concepts 5-6)
simulator.py      Eyes-free error generator (the dataset)
mine_tanglish.py  Mines TamilMixSentiment into tanglish_mined.json
evaluate.py       4-model comparison        tests.py  regression + bulk suite
eda.py            Review charts (heatmap, error types, corpus stats)
```

## Data

- Brown corpus (57,340 sentences) + NPS Chat + Webtext — language model
- **TamilMixSentiment** (Chakravarthi et al., 2020; 15,744 code-mixed comments) — Tanglish
- Self-generated: QWERTY-adjacency error simulator producing aligned pairs

## Honest limitations / future work

n-gram context ≪ transformer context (an optional neural re-ranker is the
natural next step) · translation is transfer-based, long mixed sentences come
out rough · voice input uses the browser's online STT · Android IME packaging.