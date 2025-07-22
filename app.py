import json
import tempfile

import requests
import streamlit as st
from langchain_core.runnables import RunnableConfig

from config import Config
from run import graph
from states import SummaryStateInput


def save_uploaded_file(uploaded_file) -> str:
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def render_reflection(reflection_json: str):
    try:
        data = json.loads(reflection_json)
        st.markdown(f"**🧾 Summary:** {data['summary']}")
        st.markdown(f"**📊 Convergence Score:** {data['convergence_score']}/10")
        st.markdown("**💡 Recommendations:**")
        for item in data.get("recommendations", []):
            st.markdown(f"- {item}")
    except Exception:
        st.warning("⚠️ Reflection was not in expected JSON format.")
        st.code(reflection_json)


def render_final_answer_bundle(data: dict):
    try:
        st.markdown(f"**🧠 Final Answer:**\n\n{data['answer']}")
        with st.expander("📄 PDF Chunks Used"):
            chunks = data.get("pdf_chunks", [])
            if chunks:
                for i, chunk in enumerate(chunks):
                    st.markdown(f"{chunk}")
            else:
                st.info("No relevant PDF chunks found.")
        with st.expander("🌐 Web Sources Used"):
            sources = data.get("web_sources", [])
            if sources:
                for i, src in enumerate(sources):
                    st.markdown(f"{src}")
            else:
                st.info("No web sources used.")
    except Exception as e:
        st.warning("Could not parse or display final answer bundle.")
        st.code(data)


def list_ollama_models(base_url: str):
    try:
        res = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        return [m["name"] for m in res.json().get("models", [])]
    except Exception:
        return [f"⚠️ Could not connect to Ollama ({base_url})"]


def list_lmstudio_models(base_url: str):
    try:
        res = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
        return [m["id"] for m in res.json().get("data", [])]
    except Exception:
        return [f"⚠️ Could not connect to LMStudio ({base_url})"]


# 🔧 Load default config from .env
default_config = Config.from_runnable_config()

st.set_page_config(page_title="ThinkFusionAgent", layout="wide")
st.title("ThinkFusionAgent: Hybrid PDF + Web Research Agent")

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Runtime Configuration")

llm_provider = st.sidebar.selectbox("LLM Provider", ["ollama", "lmstudio"],
                                    index=["ollama", "lmstudio"].index(default_config.llm_provider))
ollama_base_url = st.sidebar.text_input("Ollama Base URL", value=default_config.ollama_base_url)
lmstudio_base_url = st.sidebar.text_input("LMStudio Base URL", value=default_config.lmstudio_base_url)

if llm_provider == "ollama":
    available_models = list_ollama_models(ollama_base_url)
else:
    available_models = list_lmstudio_models(lmstudio_base_url)

local_llm = st.sidebar.selectbox("Local LLM", options=available_models,
                                 index=0 if default_config.local_llm not in available_models else available_models.index(
                                     default_config.local_llm))

search_api = st.sidebar.selectbox("Search API", ["duckduckgo", "tavily", "perplexity", "searxng"],
                                  index=["duckduckgo", "tavily", "perplexity", "searxng"].index(
                                      default_config.search_api))
fetch_full_page = st.sidebar.checkbox("Fetch Full Page Content", value=default_config.fetch_full_page)
strip_thinking_tokens = st.sidebar.checkbox("Strip <think> Tokens", value=default_config.strip_thinking_tokens)
local_em = st.sidebar.text_input("Embedding Model", value=default_config.local_em)
max_attempts = st.sidebar.slider("Max Web Research Attempts", 1, 5, value=default_config.max_web_research_attempts)
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

# Build override config from sidebar values
user_config = RunnableConfig(configurable={
    "llm_provider": llm_provider,
    "local_llm": local_llm,
    "search_api": search_api,
    "fetch_full_page": fetch_full_page,
    "strip_thinking_tokens": strip_thinking_tokens,
    "local_em": local_em,
    "ollama_base_url": ollama_base_url,
    "lmstudio_base_url": lmstudio_base_url,
    "max_web_research_attempts": max_attempts,
})

# --- Main Input Form ---
st.markdown("Ask a question and optionally upload a PDF. The system will answer using both document and web search.")

with st.form("research_form"):
    user_question = st.text_area("🔎 Your research question:", height=150)
    uploaded_pdf = st.file_uploader("📁 Upload a PDF (optional)", type=["pdf"])
    submitted = st.form_submit_button("🚀 Run Agent")

if submitted:
    if not user_question.strip():
        st.error("Please enter a valid question.")
        st.stop()

    pdf_path = save_uploaded_file(uploaded_pdf) if uploaded_pdf else None
    st.subheader("🧪 Debug Info" if debug_mode else "⚙️ Configuration Summary")
    st.code({
        "llm_provider": llm_provider,
        "local_llm": local_llm,
        "search_api": search_api,
        "local_em": local_em,
        "fetch_full_page": fetch_full_page,
        "strip_thinking_tokens": strip_thinking_tokens,
        "max_attempts": max_attempts,
        "ollama_base_url": ollama_base_url,
        "lmstudio_base_url": lmstudio_base_url,
        "pdf_path": pdf_path,
    }, language="json")

    st.subheader("🧪Agent Progress")
    progress_placeholder = st.empty()
    log_lines = []

    final_output = {}
    step_outputs = {}

    for event in graph.stream(SummaryStateInput(user_question=user_question, pdf_path=pdf_path), config=user_config):
        if not isinstance(event, dict) or not event:
            continue

        step_name, step_output = next(iter(event.items()))
        step_outputs[step_name] = step_output

        if debug_mode:
            if step_output:
                formatted_output = json.dumps(step_output, indent=2)
                log_lines.append(f"✅ **Step: `{step_name}`**\n```json\n{formatted_output}\n```")
            else:
                log_lines.append(f"✅ **Step: `{step_name}`**\nℹ️ No output produced.")

            progress_placeholder.markdown(
                f"<div style='height: 300px; overflow-y: scroll; padding: 1em; border: 1px solid #ccc; border-radius: 5px;'>"
                + "<br>".join(log_lines)
                + "</div>",
                unsafe_allow_html=True
            )

        if step_name in {
            "post_reflect",
            "generate_answer",
            "generate_answer_from_pdf",
            "generate_answer_from_web",
        }:
            final_output.update(step_output)

    st.success("🎉 Agent run complete!")

    if debug_mode:
        with st.expander("🧾 Full Output (Debug Dump)"):
            st.code(json.dumps(final_output, indent=2), language="json")

    if "final_answer_bundle" in final_output:
        st.subheader("📦 Final Answer Bundle")
        render_final_answer_bundle(final_output["final_answer_bundle"])
    else:
        st.subheader("🧠 Final Answer")
        st.markdown(final_output.get("final_answer", "No answer produced."))

    st.subheader("🪞 Post-Finalization Reflection")
    render_reflection(final_output.get("post_final_reflection", ""))
