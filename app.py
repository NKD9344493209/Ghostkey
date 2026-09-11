"""
GhostKey - FINAL APP: all six NLP concepts in one keyboard.

  C1 ensemble correction + next-word prediction
  C2 word-type classification routing
  C3 NER name protection
  C4 speech: voice input (browser STT) + sentence TTS
  C5 Tanglish code-mixed correction
  C6 Tanglish -> English translation

Plus: Explain-My-Correction, accept/reject learning, never-correct list,
on/off toggle, Typing Profile dashboard.

Run:  python -m streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
from collections import Counter

from pipeline import GhostKeyPipeline

st.set_page_config(page_title="GhostKey", page_icon="⌨️", layout="wide")


@st.cache_resource
def load_pipeline():
    return GhostKeyPipeline()

pipe = load_pipeline()

ss = st.session_state
ss.setdefault("never", set())
ss.setdefault("log", [])          # (typed, corrected, action) history
pipe.ck.never_correct = ss.never

def apply_corrections():
    """Rewrite the input box with the corrected sentence."""
    ss.input_box = ss.get("last_corrected", ss.get("input_box", ""))


st.markdown(
    "<h1 style='margin-bottom:0'>⌨️ GhostKey</h1>"
    "<p style='color:#888;margin-top:2px'>Six NLP concepts. One keyboard. "
    "Corrects eyes-free typing — and explains itself. 100% offline.</p>",
    unsafe_allow_html=True,
)

tab_kb, tab_profile, tab_about = st.tabs(
    ["Keyboard", "Typing Profile", "How it works"])

# ================= KEYBOARD TAB =================
with tab_kb:
    col_in, col_out = st.columns([1.05, 1])

    with col_in:
        st.subheader("Type (or speak) — don't look!")

        typed = st.text_area(
            "Input", height=100, label_visibility="collapsed",
            placeholder="i lobe you  |  naan college poren machan  |  "
                        "John is my best friemd",
            key="input_box")

        # ---- voice input (C4: browser speech-to-text) ----
        components.html("""
        <button id="mic" style="padding:9px 20px;border-radius:8px;border:none;
          background:#e0483e;color:white;font-size:15px;cursor:pointer">
          🎤 Voice input</button>
        <span id="heard" style="font-family:system-ui;color:#8b95a5;
          margin-left:10px;font-size:14px"></span>
        <script>
          const btn = document.getElementById("mic");
          const out = document.getElementById("heard");
          const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (!SR) { btn.disabled = true; out.textContent = "STT needs Chrome/Edge"; }
          else {
            const rec = new SR(); rec.lang = "en-IN";
            btn.onclick = () => { out.textContent = "listening..."; rec.start(); };
            rec.onresult = e => {
              const text = e.results[0][0].transcript;
              out.textContent = '"' + text + '" — copy into the box above';
            };
            rec.onerror = e => { out.textContent = "mic error: " + e.error; };
          }
        </script>""", height=55)

        # ---- on-screen QWERTY (eyes-free typing surface) ----
        components.html("""
        <style>
          .kb { user-select:none; font-family:system-ui; }
          .row { display:flex; justify-content:center; margin:3px 0; }
          .key { width:42px;height:50px;margin:3px;border-radius:8px;
            background:#1e2530;color:#e8edf5;border:1px solid #313b4b;
            display:flex;align-items:center;justify-content:center;
            font-size:17px;cursor:pointer;transition:all .06s }
          .key:active { background:#3d7bfd; transform:scale(1.08) }
          .wide { width:86px;font-size:12px } .space { width:210px }
        </style>
        <div class="kb" id="kb"></div>
        <p style="color:#8b95a5;font-family:system-ui;font-size:13px">
          buffer: <b id="echo">&nbsp;</b></p>
        <script>
          const rows = ["qwertyuiop","asdfghjkl","zxcvbnm"];
          const kb = document.getElementById("kb");
          rows.forEach(r => {
            const div = document.createElement("div"); div.className = "row";
            [...r].forEach(ch => {
              const k = document.createElement("div");
              k.className = "key"; k.textContent = ch;
              k.onclick = () => tap(ch); div.appendChild(k); });
            kb.appendChild(div); });
          const last = document.createElement("div"); last.className = "row";
          const mk = (l,c,f) => { const k = document.createElement("div");
            k.className = "key "+c; k.textContent = l; k.onclick = f;
            last.appendChild(k); };
          mk("⌫","wide",()=>tap("BKSP"));
          mk("space","space",()=>tap(" "));
          mk("🔊","wide",()=>speechSynthesis.speak(
              new SpeechSynthesisUtterance(buffer)));
          kb.appendChild(last);
          let buffer = "";
          function tap(ch){
            if (ch==="BKSP") buffer = buffer.slice(0,-1); else buffer += ch;
            document.getElementById("echo").textContent = buffer || "\\u00A0"; }
        </script>""", height=290)

    with col_out:
        st.subheader("GhostKey output")
        c1, c2 = st.columns([1, 1.4])
        enabled = c1.toggle("Correction", value=True)
        auto_th = c2.slider("Auto-correct threshold", 0.50, 0.99, 0.80, 0.01)

        if typed.strip():
            if enabled:
                r = pipe.process(typed, auto_threshold=auto_th)
            else:
                r = {"corrected": typed, "report": [], "next_words": [],
                     "translation": None, "protected": []}

            ss.last_corrected = r["corrected"]
            st.markdown(
                f"<div style='font-size:24px;padding:14px;border-radius:10px;"
                f"background:#10151d;border:1px solid #2a3444'>"
                f"{r['corrected']}</div>", unsafe_allow_html=True)

            ba, bc = st.columns(2)
            ba.button("✅ Apply corrections to my text",
                      on_click=apply_corrections,
                      use_container_width=True,
                      help="Rewrites your input with the corrected sentence")
            with bc:
                components.html(f"""
                <button style="width:100%;padding:9px;border-radius:8px;
                  border:1px solid #3d7bfd;background:transparent;
                  color:#3d7bfd;font-size:15px;cursor:pointer"
                  onclick="navigator.clipboard.writeText({r['corrected']!r})
                    .then(()=>{{this.textContent='✓ Copied!';
                    setTimeout(()=>this.textContent='📋 Copy corrected text',1200)}})">
                  📋 Copy corrected text</button>""", height=50)

            # C6: translation (appears when Tanglish detected)
            if r["translation"] and r["translation"].lower() != r["corrected"].lower():
                st.markdown(f"**🌐 English:** {r['translation']}")

            # C1: next-word prediction chips
            if r["next_words"]:
                st.markdown("**Next word:** " + " · ".join(
                    f"`{w}`" for w in r["next_words"]))

            # C4: sentence-level TTS
            components.html(
                f"""<button style="padding:8px 18px;border-radius:8px;border:none;
                background:#3d7bfd;color:white;font-size:15px;cursor:pointer"
                onclick="speechSynthesis.speak(
                  new SpeechSynthesisUtterance({r['corrected']!r}))">
                🔊 Speak sentence</button>""", height=55)

            if r["protected"]:
                st.caption("🛡️ NER-protected: " + ", ".join(r["protected"]))

            # Explain-My-Correction + feedback
            changed = [x for x in r["report"]
                       if x[3].startswith(("auto", "suggest", "tanglish-fix"))]
            if enabled and changed:
                st.markdown("#### Why these corrections?")
                for token, best, conf, action, wtype in changed:
                    label = f"'{token}' → '{best}'  ({conf:.0%}, {action}, type: {wtype})"
                    with st.expander(label):
                        _, _, expl = pipe.ck.correct_word(
                            token.lower(), explain=True)
                        if expl:
                            st.table(expl)
                        b1, b2 = st.columns(2)
                        if b1.button("✓ Accept", key=f"a{token}{best}"):
                            pipe.ck.feedback(token, best, True)
                            ss.log.append((token, best, "accepted"))
                            st.success("Learned.")
                        if b2.button("✗ Never correct this",
                                     key=f"r{token}{best}"):
                            pipe.ck.feedback(token, best, False)
                            ss.never.add(token.lower())
                            ss.log.append((token, best, "rejected"))
                            st.info(f"'{token}' protected.")
                    ss.log.append((token, best, action))

        with st.expander("🛡️ Protected words"):
            nw = st.text_input("Protect a word", key="pin")
            if st.button("Protect") and nw.strip():
                ss.never.add(nw.strip().lower())
            st.write(sorted(ss.never) or "*(none yet)*")

# ================= TYPING PROFILE TAB =================
with tab_profile:
    st.subheader("📊 Your Typing Profile")
    if not ss.log:
        st.info("Type something in the Keyboard tab first — "
                "your correction history builds this page.")
    else:
        actions = Counter(a for _, _, a in ss.log)
        pairs = Counter((t.lower(), b) for t, b, a in ss.log
                        if a.startswith(("auto", "tanglish-fix")))
        c1, c2, c3 = st.columns(3)
        c1.metric("Corrections", sum(v for k, v in actions.items()
                                     if k.startswith(("auto", "tanglish"))))
        c2.metric("Accepted", actions.get("accepted", 0))
        c3.metric("Protected words", len(ss.never))
        if pairs:
            st.markdown("**Your most common slips:**")
            st.table([{"you typed": t, "GhostKey wrote": b, "times": n}
                      for (t, b), n in pairs.most_common(8)])

# ================= ABOUT TAB =================
with tab_about:
    st.markdown("""
#### The six NLP concepts inside GhostKey
| # | Concept | Where it lives |
|---|---|---|
| 1 | Spelling correction & language modelling | Ensemble: keyboard-weighted edit distance × bigram context × frequency prior; three-zone confidence policy; next-word prediction |
| 2 | Text classification | Char-n-gram Logistic Regression routes each word: english / tanglish / name / gibberish |
| 3 | Named Entity Recognition | spaCy detects PERSON/ORG/GPE → automatic never-correct protection |
| 4 | Speech processing | Browser speech-to-text input; sentence-level speech synthesis output |
| 5 | Multilingual NLP | Tanglish code-mixed correction against a romanized-Tamil lexicon |
| 6 | Machine translation | Dictionary + rule based Tanglish → English transfer |

**Privacy:** everything runs locally. The only network access ever needed
is the one-time NLTK corpus download.
""")

st.divider()
st.caption("GhostKey — eyes-free typing, explained corrections, offline by design.")
