import difflib
import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from config import Config
from prompts import (
    query_writer_instructions, openbook_answer_instructions, openbook_human_prompt,
    answer_quality_reflection_instructions, answer_quality_reflection_prompt,
    web_reflection_instructions, web_reflection_prompt,
    post_finalization_reflection_instructions, post_finalization_reflection_prompt
)
from states import SummaryState, SummaryStateInput, SummaryStateOutput
from utils import (
    get_current_date, get_config_value, tavily_search, deduplicate_and_format_sources, format_sources,
    perplexity_search, duckduckgo_search, searxng_search, strip_thinking_tokens, extract_text_from_pdf,
    find_relevant_pdf_chunks, get_llm_from_config, deduplicate_sources
)


def load_document_text(state: SummaryState, config: RunnableConfig):
    """Extracts raw text from the PDF file path if a path is provided.

    Args:
        state (SummaryState): Current pipeline state.
        config (RunnableConfig): Not used here

    Returns:
        dict: {'pdf_text': str}
    """

    if not state.pdf_path:
        return {"extracted_text": "", "relevant_chunks": []}
    return {"extracted_text": extract_text_from_pdf(state.pdf_path)}


def extract_relevant_chunks(state: SummaryState, config: RunnableConfig):
    """Uses semantic search to select relevant chunks from the extracted PDF text.

    Args:
        state (SummaryState): Current pipeline state.
        config (RunnableConfig): Runtime config.

    Returns:
        dict: {'pdf_chunks': list of str}
    """
    if not state.extracted_text.strip():
        return {"pdf_chunks": []}
    return {"relevant_chunks": find_relevant_pdf_chunks(config, state.extracted_text, state.user_question)}


def generate_answer(state: SummaryState, config: RunnableConfig):
    """Uses LLM to answer the user question using PDF + Web content.

    Args:
        state (SummaryState): Contains research_topic and relevant_chunks.
        config (RunnableConfig): Runtime configuration (LLM-related).

    Returns:
        dict: {'final_answer': str}
    """
    cfg, llm = get_llm_from_config(config)

    result = llm.invoke([
        SystemMessage(content=openbook_answer_instructions),
        HumanMessage(
            content=openbook_human_prompt.format(
                query=state.user_question,
                pdf_context="\n\n".join(state.relevant_chunks or []),
                web_context="\n\n".join(state.web_context or []),
            )
        )
    ])

    # Track answer evolution
    new_answer = result.content.strip()
    previous_answer = state.final_answer or ""
    new_history = (state.answer_history or []) + [new_answer]
    similarity = difflib.SequenceMatcher(None, previous_answer.strip(), new_answer).ratio()
    has_converged = similarity >= 0.95

    return {
        "final_answer": new_answer,
        "answer_history": new_history,
        "has_converged": has_converged,
        "final_answer_bundle": {
            "answer": new_answer,
            "pdf_chunks": state.relevant_chunks,
            "web_sources": deduplicate_sources(state.web_sources)
        }
    }


def generate_search_query(state: SummaryState, config: RunnableConfig):
    """Uses LLM to rewrite the user question into a search-optimized query.

    Args:
        state (SummaryState): Contains user_question
        config (RunnableConfig): Includes LLM format=json

    Returns:
        dict: {'search_query': str}
    """

    current_date = get_current_date()
    cfg, llm = get_llm_from_config(config, format="json")

    result = llm.invoke([
        SystemMessage(
            content=query_writer_instructions.format(
                current_date=current_date,
                research_topic=state.user_question,
            )
        ),
        HumanMessage(content=f"Generate a query for web search:")
    ])

    try:
        parsed = json.loads(result.content)
        search_query = parsed["query"]
    except (json.JSONDecodeError, KeyError):
        # If parsing fails or the key is not found, use a fallback query
        search_query = strip_thinking_tokens(result.content) if cfg.strip_thinking_tokens else result.content
    return {"search_query": search_query}


def web_research(state: SummaryState, config: RunnableConfig):
    """Executes a web search using the configured search API (tavily, perplexity, duckduckgo, or searxng)
    and formats the results for further processing.

    Args:
        state (SummaryState): Contains search_query
        config (RunnableConfig): Controls engine and max tokens

    Returns:
        dict: Updated state with web_context and sources
    """
    cfg = Config.from_runnable_config(config)
    engine = get_config_value(cfg.search_api)

    # Search the web
    if engine == "tavily":
        results = tavily_search(state.search_query, fetch_full_page=cfg.fetch_full_page, max_results=1)
    elif engine == "perplexity":
        results = perplexity_search(state.search_query, state.search_attempts)
    elif engine == "duckduckgo":
        results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=cfg.fetch_full_page)
    elif engine == "searxng":
        results = searxng_search(state.search_query, max_results=3, fetch_full_page=cfg.fetch_full_page)
    else:
        raise ValueError(f"Unsupported search API: {cfg.search_api}")

    formatted = deduplicate_and_format_sources(results, max_tokens_per_source=1000, fetch_full_page=cfg.fetch_full_page)

    return {
        "web_sources": [format_sources(results)],
        "search_attempts": state.search_attempts + 1,
        "web_context": [formatted],
        # "web_context_empty": not any(chunk.strip() for chunk in formatted)
    }


def reflect_on_answer_quality(state: SummaryState, config: RunnableConfig):
    """Determines whether the current answer is sufficient to finalize, or if more web info is needed.

    Uses an LLM to reflect on the sufficiency of the answer given the PDF and (if any) web results.

    Args:
        state (SummaryState): Contains the current answer and sources.
        config (RunnableConfig): Runtime config.

    Returns:
        str: Next node to transition to ("web_research" or "finalize")
    """
    cfg, llm = get_llm_from_config(config, format="json")
    result = llm.invoke([
        SystemMessage(content=answer_quality_reflection_instructions),
        HumanMessage(
            content=answer_quality_reflection_prompt.format(
                query=state.user_question,
                answer=state.final_answer,
                sources="\n\n".join(
                    state.relevant_chunks +
                    (state.web_context or [])
                ),
            )
        )
    ])

    try:
        parsed = json.loads(result.content)
        is_sufficient = parsed.get("sufficient", False)
        reasoning = parsed.get("reasoning", "")
    except Exception:
        is_sufficient = len(state.final_answer.strip()) >= 200
        reasoning = "Fallback heuristic used due to parse failure"

    return {
        "answer_is_sufficient": is_sufficient,
        "answer_reflection_reasoning": reasoning
    }


def reflect_on_web_context(state: SummaryState, config: RunnableConfig):
    """Evaluate whether the web context is useful or needs refinement.

    Args:
        state (SummaryState): Current state with web search results.
        config (RunnableConfig): Runtime config.

    Returns:
        dict: {"knowledge_gap": "...", "follow_up_query": "..." }
    """
    cfg, llm = get_llm_from_config(config, format="json")
    result = llm.invoke([
        SystemMessage(content=web_reflection_instructions),
        HumanMessage(
            content=web_reflection_prompt.format(
                query=state.user_question,
                search_query=state.search_query,
                context="\n\n".join(state.web_context or []),
                attempt=state.search_attempts
            )
        )
    ])

    try:
        parsed = json.loads(result.content)
        search_query = parsed.get('follow_up_query', f"Tell me more about {state.user_question}")
        search_action = parsed.get('action', "regenerate_query")
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        search_query = f"Tell me more about {state.user_question}"
        search_action = "regenerate_query"

    return {"search_query": search_query, "search_action": search_action}


def post_reflect(state: SummaryState, config: RunnableConfig):
    """Run a final reflection on the evolution of answers throughout the pipeline.

    This function performs a post-finalization analysis using an LLM to evaluate
    the quality, progression, and convergence of all intermediate and final answers
    given a user's question. It uses a dedicated prompt and instructions to assess
    whether the reasoning improved, stabilized, or remained inconsistent.

    The reflection generates a structured summary that includes:
    - a narrative of how the answer evolved,
    - a convergence score (0–10),
    - and a list of recommendations for improving future runs.

    Args:
        state (SummaryState): The current agent state, including the user question and
                              all previously generated answers in `answer_history`.
        config (RunnableConfig): Runtime configuration object used to determine which LLM to use.

    Returns:
        dict: A dictionary with a single key:
            - "post_final_reflection" (str): The JSON-formatted string returned by the LLM, which includes:
                - summary (str): Narrative description of answer evolution
                - convergence_score (int): Stability score between 0 and 10
                - recommendations (List[str]): Suggestions for improving the pipeline
    """
    cfg, llm = get_llm_from_config(config)
    result = llm.invoke([
        SystemMessage(content=post_finalization_reflection_instructions),
        HumanMessage(
            content=post_finalization_reflection_prompt.format(
                query=state.user_question,
                history="\n\n".join(state.answer_history)
            )
        )
    ])
    return {"post_final_reflection": result.content}


def route_on_source_availability(state: SummaryState, config: RunnableConfig) -> str:
    """
    Routes based on whether relevant PDF content is available.

    Args:
        state (SummaryState): Current pipeline state.
        config (RunnableConfig): Runtime config object.

    Returns:
        str: "generate_answer" if PDF chunks exist, else "generate_search_query".
    """
    if state.relevant_chunks:
        return "generate_answer"

    return "generate_search_query"


def route_on_answer_reflection(state: SummaryState, config: RunnableConfig) -> str:
    """Route based on answer quality or convergence status.

    If the answer is marked as sufficient OR the answer has converged
    (i.e., high similarity to previous answer), the pipeline will finalize.

    If not, the system will decide whether to do more web research.

    Args:
        state (SummaryState): Current state with answer quality flags.
        config (RunnableConfig): Runtime config, including max attempts.

    Returns:
        str: Next step in the pipeline ("post_reflect" or "web_research").
    """
    cfg = Config.from_runnable_config(config)
    if state.answer_is_sufficient or state.has_converged or state.search_attempts >= cfg.max_web_research_attempts:
        return "post_reflect"

    return "web_research"


def route_on_web_reflection(state: SummaryState, config: RunnableConfig) -> str:
    """Route decision based on web context reflection.

    Args:
        state (SummaryState): Current state with web search results.
        config (RunnableConfig): Runtime config.

    Returns:
        str: Next node to transition to ("web_research" or "finalize")
    """
    # TODO: Short-circuit only if we've already tried and it's still useless
    if state.search_attempts >= 1 and (
            state.web_context_empty or not any(state.web_context)
    ):
        return "finalize"

    cfg = Config.from_runnable_config(config)
    if state.search_action == "answer":
        return "finalize"

    if state.search_action == "regenerate_query":
        return "web_research" if state.search_attempts <= cfg.max_web_research_attempts else "finalize"
    return "web_research"


builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Config)
builder.add_node("load_document_text", load_document_text)
builder.add_node("extract_relevant_chunks", extract_relevant_chunks)
builder.add_node("generate_answer_from_pdf", generate_answer)
builder.add_node("generate_answer_from_web", generate_answer)
builder.add_node("reflect_answer_from_pdf", reflect_on_answer_quality)
builder.add_node("reflect_answer_from_web", reflect_on_answer_quality)
builder.add_node("generate_search_query", generate_search_query)
builder.add_node("web_research", web_research)
builder.add_node("reflect_on_web_context", reflect_on_web_context)
builder.add_node("post_reflect", post_reflect)

# Flow
builder.add_edge(START, "load_document_text")
builder.add_edge("load_document_text", "extract_relevant_chunks")

# After extraction, decide if PDF gives anything
builder.add_conditional_edges("extract_relevant_chunks", route_on_source_availability, path_map={
    "generate_answer": "generate_answer_from_pdf",
    "generate_search_query": "generate_search_query"
})

# PDF path
builder.add_edge("generate_answer_from_pdf", "reflect_answer_from_pdf")
builder.add_conditional_edges("reflect_answer_from_pdf", route_on_answer_reflection, path_map={
    "web_research": "generate_search_query",
    "post_reflect": "post_reflect"
})

# Web path
builder.add_edge("generate_search_query", "web_research")
builder.add_edge("web_research", "reflect_on_web_context")
builder.add_conditional_edges("reflect_on_web_context", route_on_web_reflection, path_map={
    "web_research": "generate_search_query",
    "finalize": "generate_answer_from_web"
})

builder.add_edge("generate_answer_from_web", "reflect_answer_from_web")
builder.add_conditional_edges("reflect_answer_from_web", route_on_answer_reflection, path_map={
    "web_research": "generate_search_query",
    "post_reflect": "post_reflect"
})

# Final
builder.add_edge("post_reflect", END)
# builder.add_edge("generate_answer", END)
graph = builder.compile()
