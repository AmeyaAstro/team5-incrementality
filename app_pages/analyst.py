"""AI component 2: ask questions in plain English; every figure is traced back to the data."""

import streamlit as st

from incrementality import ai
from incrementality.ui import DEFAULT_AOV, DEFAULT_COST, assumptions

b = st.session_state["bundle"]
aov, cost = assumptions()
tb = ai.Toolbox(b, aov, cost)
illustrative = aov == DEFAULT_AOV and cost == DEFAULT_COST

st.title("Ask the analyst")
st.caption("A Gemini LLM that can only answer by calling the same functions that draw this app's charts. "
           "After each answer, every figure it quotes is checked against those function outputs.")

key = ai.api_key()
with st.expander("AI settings", expanded=not key):
    st.text_input("Google Gemini API key (kept in this browser session only)", type="password", key="user_api_key",
                  help="Free key at aistudio.google.com/apikey. Or set GEMINI_API_KEY in .streamlit/secrets.toml.")
    st.caption(f"Model: `{ai.model_name()}` (override with GEMINI_MODEL in secrets).")
    key = ai.api_key()

# ---------------------------------------------------------------- brief (works without a key)
st.subheader("CFO brief at your current assumptions")
brief, _ = ai.deterministic_brief(tb)
st.markdown(brief)
if illustrative:
    st.caption("⚠️ Dollar figures use the illustrative $85 / $0.35 defaults. Enter Vantage's numbers in the sidebar.")

st.divider()


# ---------------------------------------------------------------- chat
def render_answer(ans: dict) -> None:
    st.markdown(ans["text"])
    g = ans["grounding"]
    if g.total:
        label = f"✅ All {g.total} figures traced to the data" if g.ok == g.total else \
            f"⚠️ {g.total - g.ok} of {g.total} figures could not be traced. Treat them with caution."
    else:
        label = "No figures quoted"
    with st.expander(label):
        if g.checked:
            st.write(" · ".join(f"{'✅' if ok else '⚠️'} `{t}`" for t, ok in g.checked))
        for name, inp, out in ans["calls"]:
            st.markdown(f"**{name}** `{inp}`")
            st.json(out, expanded=False)


st.subheader("Ask a question")
if not key:
    st.info("Add a Google Gemini API key above to enable questions. The brief above is always available.", icon="🔑")
    st.stop()

st.session_state.setdefault("chat", [])
suggestions = [
    "Should we renew the $6M budget as-is?",
    "If our real AOV is $140, does the program pay for itself?",
    "Which users should we stop targeting, and what do we save?",
    "Move 70% of the Weak Responder budget into Persuadables on $6M. What happens?",
    "How sure are you the Persuadable segment is real?",
]
cols = st.columns(len(suggestions))
picked = None
for c, s in zip(cols, suggestions):
    if c.button(s, use_container_width=True):
        picked = s

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        render_answer(turn["answer"]) if turn["role"] == "assistant" else st.markdown(turn["content"])

q = st.chat_input("e.g. What's our true ROAS if cost per user drops to $0.20?") or picked
if q:
    with st.chat_message("user"):
        st.markdown(q)
    history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat][-6:]
    with st.chat_message("assistant"):
        with st.status("Querying the experiment…", expanded=True) as status:
            try:
                ans = ai.ask(q, history, tb, illustrative, key, on_step=status.write)
            except Exception as e:
                status.update(label="The AI call failed", state="error")
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    st.error("Gemini's free-tier rate limit was hit. Wait a minute and ask again.")
                else:
                    st.error(f"The AI call failed ({type(e).__name__}: {msg[:300]}). The brief above still reflects the data.")
                st.stop()
            status.update(label=f"Answered with {ans['model']} · {len(ans['calls'])} data lookups",
                          state="complete", expanded=False)
        render_answer(ans)
    st.session_state.chat += [{"role": "user", "content": q},
                              {"role": "assistant", "content": ans["text"], "answer": ans}]
